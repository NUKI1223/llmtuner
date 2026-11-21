"""The eval loop: run a target on every case, grade each output, store the result."""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from llmtuner.config import TARGET_GRADER, Config, ConfigError
from llmtuner.dataset import Case, load_cases, select_split
from llmtuner.graders import Grade, check_case, grade
from llmtuner.store import Store
from llmtuner.targets import META_ENV, TRIAL_ENV, CommandResult, run_command

# Upper bound on repeats of one run: each trial costs one target call per case.
MAX_TRIALS = 50

_MAX_META_BYTES = 65_536


def _reject_constant(name: str):
    raise ValueError(f"{name} is not a finite number")


def _finite_float(text: str) -> float:
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):  # 1e999 parses to infinity
        raise ValueError(f"{text} is not a finite number")
    return value


def _read_meta(path: Path) -> dict | None:
    """What the target reported about its call (tokens, cost). Missing or malformed is None."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_META_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                          parse_float=_finite_float)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None

# TARGET_GRADER (the pseudo-grader recorded when the target itself failed) lives in config.py
# so that config can reserve the name; it stays available here.


def _tool_version(config: Config, command) -> str | None:
    if command is None:
        return None
    result = run_command(command, "", 15.0, config.root)
    return result.output.strip().splitlines()[0][:200] if result.ok and result.output.strip() else None


def run_eval(
    config: Config,
    store: Store,
    target_name: str,
    dataset_name: str,
    split: str = "test",
    note: str | None = None,
    jobs: int = 1,
    limit: int | None = None,
    on_case: Callable[[Case, bool], None] | None = None,
    trials: int = 1,
    on_trial: Callable[[int, int], None] | None = None,
) -> int:
    """Run one target on one dataset split and return the id of the stored run.

    With `trials` above 1 every case is run that many times, to see how stable the target is.
    Each trial is stored as an ordinary run; the runs share a trial group and the id of the
    first one is returned. The dataset is read once, so every trial records the same data.
    """
    if not isinstance(trials, int) or isinstance(trials, bool) or not 1 <= trials <= MAX_TRIALS:
        raise ConfigError(f"trials must be a whole number from 1 to {MAX_TRIALS}")
    target = config.target(target_name)
    dataset = config.dataset(dataset_name)
    graders = config.graders_for(dataset)
    if not any(g.required for g in graders):
        raise ConfigError(f"dataset '{dataset.name}' has no required grader, so nothing decides pass or fail")

    cases, dataset_hash = load_cases(dataset.path)
    cases = select_split(cases, split)
    if limit is not None:
        if limit < 1:
            raise ConfigError("limit must be at least 1")
        cases = cases[:limit]
    # Check the data before spending any model calls.
    for case in cases:
        for grader in graders:
            check_case(grader, case)

    def safe_grade(g, case: Case, output: str) -> Grade:
        try:
            return grade(g, case, output, config.root)
        except Exception as exc:  # a grader bug must not abort the whole run
            return Grade(g.name, False, f"grader crashed: {type(exc).__name__}: {exc}", error=True)

    def execute(item: tuple[int, Case], trial_no: int = 1
                ) -> tuple[Case, CommandResult, list[Grade], bool, dict | None]:
        index, case = item
        meta = None
        try:
            # The target may write token and cost figures for this call into its own file.
            meta_path = Path(meta_dir) / f"trial-{trial_no}-case-{index}.json"
            result = run_command(target.command, case.input_text, target.timeout, config.root,
                                 env={META_ENV: str(meta_path), TRIAL_ENV: str(trial_no)})
            meta = _read_meta(meta_path)
            if not result.ok:
                # A timeout or crash is a failed case, not a crashed eval.
                return case, result, [Grade(TARGET_GRADER, False, result.error or "", error=True)], False, meta
            grades = [safe_grade(g, case, result.output) for g in graders]
            required = {g.name for g in graders if g.required}
            passed = all(g.passed for g in grades if g.grader in required)
            return case, result, grades, passed, meta
        except Exception as exc:  # keep the run going; record this case as failed
            message = f"harness error: {type(exc).__name__}: {exc}"
            result = CommandResult("", None, False, 0.0, message)
            return case, result, [Grade(TARGET_GRADER, False, message, error=True)], False, meta

    tool_version = _tool_version(config, target.version_command)
    grouped = trials > 1
    first_id = None
    with tempfile.TemporaryDirectory(prefix="llmtuner-meta-") as meta_dir, \
            ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for trial_no in range(1, trials + 1):
            run_id = store.create_run(
                project=config.project,
                dataset=dataset.name,
                dataset_hash=dataset_hash,
                target=target.name,
                command=target.command,
                model=target.model,
                tool_version=tool_version,
                split=split,
                graders=[g.describe() for g in graders],
                note=note,
                trial_group=first_id if grouped else None,
                trial=trial_no if grouped else None,
                trials_planned=trials if grouped else None,
            )
            first_id = first_id or run_id
            if on_trial:
                on_trial(trial_no, trials)
            # map() keeps dataset order; results are written from this thread only.
            results = pool.map(lambda item, number=trial_no: execute(item, number), enumerate(cases))
            for case, result, grades, passed, meta in results:
                store.add_case(run_id, case, result, passed, grades, meta)
                if on_case:
                    on_case(case, passed)
            store.finish_run(run_id)
    return first_id
