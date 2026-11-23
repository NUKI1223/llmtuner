"""What an agent may do with the holdout split, and what it may see of it.

The holdout is the part of a dataset nobody tunes against. Through MCP an agent may run it
once per rolling day and sees only totals: no case ids, inputs, outputs or reasons. People using
the CLI or the web page are not restricted. This is only a guard for agents that are given
nothing but MCP: an agent that can read files can read the dataset file itself.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from llmtuner import report
from llmtuner.config import ConfigError
from llmtuner.dataset import DatasetError
from llmtuner.store import Store

WINDOW_SECONDS = 24 * 60 * 60


class HoldoutError(ConfigError):
    """The request would break a holdout rule."""


def check_agent_run(store: Store, project: str, dataset: str, split: str, limit: int | None,
                    now: float | None = None, reserve: bool = False) -> int | None:
    """Refuse an agent's holdout run that has a `limit` or comes within 24 hours of the last one.

    Every earlier agent run of this holdout counts, finished or not, a group of trials once.
    Nothing is checked for other splits. With `reserve` the attempt is also taken, in the same
    database transaction as the check, and its reservation id is returned (release it with
    `store.release_holdout` when the run is over); this is what stops parallel callers, even in
    other processes, from all getting through.
    """
    if split != "holdout":
        return None
    if limit is not None:
        raise HoldoutError("a holdout run cannot use limit: it must run every holdout case")
    now = time.time() if now is None else now
    if reserve:
        token, latest = store.reserve_holdout(project, dataset, now, WINDOW_SECONDS)
    else:
        recent = store.holdout_attempts(project, dataset, now - WINDOW_SECONDS)
        token, latest = None, (recent[0] if recent else None)
    if latest is not None:
        allowed = datetime.fromtimestamp(latest + WINDOW_SECONDS, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        raise HoldoutError(f"the holdout of {project} / {dataset} was already run through MCP in the last 24"
                           f" hours (a run that has started counts, finished or not); the next attempt is allowed"
                           f" at {allowed}")
    return token


def run_for_agent(store: Store, config, target: str, dataset: str, split: str, note, limit, trials) -> int:
    """Start a run for an agent (origin "mcp"), applying the holdout rules. Returns the run id.

    A holdout run that fails its checks reports no case ids: what is wrong with holdout data is for
    a person to read."""
    from llmtuner import runner

    def start() -> int:
        return runner.run_eval(config, store, target, dataset, split=split, note=note, limit=limit,
                               trials=trials, origin="mcp")

    if split != "holdout":
        return start()
    token = check_agent_run(store, config.project, dataset, split, limit, reserve=True)
    try:
        return start()
    except DatasetError:  # its message may name a holdout case
        raise HoldoutError("the holdout run could not start (its data failed a check); details are not shown"
                           " through MCP, ask the owner to run it from the CLI") from None
    finally:
        # By now the run row, if the run got as far as starting, carries the attempt. A run refused
        # before it started (bad data, unknown target) did not use it.
        store.release_holdout(token)


def is_holdout(store: Store, run_id: int) -> bool:
    return store.get_run(run_id)["split"] == "holdout"


def check_show_cases(store: Store, run_id: int) -> None:
    """Agents cannot read the cases of a holdout run."""
    if is_holdout(store, run_id):
        raise HoldoutError("run %d is a holdout run: its cases, inputs, outputs and failures are not shown"
                           " through MCP, only the totals in get_report" % run_id)


# --- redaction ---------------------------------------------------------------

_RUN_KEYS = ("id", "project", "dataset", "target", "model", "split", "started_at", "finished_at")


def _run(run: dict) -> dict:
    return {k: run[k] for k in _RUN_KEYS}


def hide_holdout_changes(changes: list[dict] | None) -> list[dict] | None:
    """For a list of changed cases: swap the ones that involve the holdout for a bare marker, so an
    agent never sees which holdout case changed, or its values."""
    if changes is None:
        return None
    return [{"kind": c["kind"], "hidden": True} if "holdout" in (c["split_before"], c["split_after"]) else c
            for c in changes]


def redact_compare(comparison: dict | None, whole: bool = True) -> dict | None:
    """A comparison with only totals. `whole` is True when a holdout run is one of the two; if not,
    only the changed cases that involve the holdout are hidden."""
    if comparison is None:
        return None
    if not whole:
        return {**comparison, "changed_cases": hide_holdout_changes(comparison["changed_cases"])}
    keep = ("n", "baseline_rate", "candidate_rate", "diff", "p_value", "measurable", "share_worse",
            "detection_floor", "verdict", "warnings", "baseline_trials", "candidate_trials",
            "changes_log_available")
    out = {k: comparison[k] for k in keep if k in comparison}
    out["baseline"], out["candidate"] = _run(comparison["baseline"]), _run(comparison["candidate"])
    out["n_regressions"], out["n_improvements"] = len(comparison["regressions"]), len(comparison["improvements"])
    out["n_changed_cases"] = len(comparison["changed_cases"])
    out["redacted"] = "holdout: case ids, per-case results and details are not shown"
    return out


def redact_report(rep: dict) -> dict:
    """A report on a holdout run: the totals and nothing that could identify a case."""
    trials = rep["trials"]
    if trials is not None:
        trials = {k: trials[k] for k in ("group", "planned", "finished", "complete", "n", "passed_all",
                                         "passed_any", "per_trial", "usage")}
        trials["runs"] = len(rep["trials"]["runs"])  # how many, not which
        trials["n_unstable_cases"] = len(rep["trials"]["unstable_cases"])
    changed = rep["changed_since_previous"]
    return {
        "run": _run(rep["run"]),
        "figures_describe": rep["figures_describe"],
        "n": rep["n"],
        "n_passed": rep["n_passed"],
        "pass_rate": rep["pass_rate"],
        "detection_floor": rep["detection_floor"],
        "usage": rep["usage"],
        "is_baseline": rep["is_baseline"],
        "baseline": redact_compare(rep["baseline"]),
        "trials": trials,
        "n_changed_since_previous": None if changed is None else len(changed),
        "redacted": "holdout: case ids, failures, grader details, inputs and outputs are not shown",
    }


def redact_gate(result: dict, whole: bool = True) -> dict:
    """A gate result for an agent. Holdout ids are never named in the reason."""
    comparison = result["comparison"]
    out = dict(result)
    out["comparison"] = redact_compare(comparison, whole)
    blocking = out.pop("blocking_changes", [])
    if blocking and comparison is not None:
        involved = [c for c in comparison["changed_cases"] if c["case_id"] in blocking]
        out["reason"] = report.blocking_reason(involved, comparison["baseline"]["id"],
                                               comparison["candidate"]["id"], "all" if whole else "holdout")
    out["n_blocking_changes"] = len(blocking)
    return out


def agent_report(store: Store, run_id: int, changes_log: list[dict] | None = None) -> dict:
    """get_report as an agent sees it."""
    run = store.get_run(run_id)
    base_id = store.get_baseline(run["project"], run["dataset"])
    involved = "holdout" in (run["split"], store.get_run(base_id)["split"] if base_id is not None else None)
    rep = report.build_report(store, run_id, changes_log, holdout_only=involved)
    if rep["run"]["split"] == "holdout":
        return redact_report(rep)
    rep["baseline"] = redact_compare(rep["baseline"], whole=_baseline_is_holdout(store, rep["baseline"]))
    rep["changed_since_previous"] = hide_holdout_changes(rep["changed_since_previous"])
    return rep


def _baseline_is_holdout(store: Store, comparison: dict | None) -> bool:
    return comparison is not None and comparison["baseline"]["split"] == "holdout"


def agent_compare(store: Store, baseline_id: int, candidate_id: int,
                  changes_log: list[dict] | None = None) -> dict:
    whole = "holdout" in (store.get_run(baseline_id)["split"], store.get_run(candidate_id)["split"])
    comparison = report.compare_runs(store, baseline_id, candidate_id, changes_log, holdout_only=whole)
    return redact_compare(comparison, whole)


def agent_gate(store: Store, run_id: int, baseline_id: int | None = None, max_drop: float = 0.03,
               changes_log: list[dict] | None = None) -> dict:
    run = store.get_run(run_id)
    if baseline_id is None:
        baseline_id = store.get_baseline(run["project"], run["dataset"])
    whole = baseline_id is not None and "holdout" in (run["split"], store.get_run(baseline_id)["split"])
    result = report.gate(store, run_id, baseline_id, max_drop, changes_log, holdout_only=whole)
    comparison = result["comparison"]
    if comparison is None:
        return result
    return redact_gate(result, whole)
