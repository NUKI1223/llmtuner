"""MCP server: exposes the llmtuner library to AI agents over stdio.

This is a thin wrapper. Every tool calls an existing library function and holds
no eval logic of its own. No tool takes a command to execute: commands come only
from the project's own `llmtuner.toml`. Human labels and baselines are ground
truth and can only be set by a person through the CLI, so no tool writes them.

The `mcp` package is optional. This module imports without it; `main()` explains
how to install it.
"""

from __future__ import annotations

import sys
from typing import Any

from llmtuner import report, runner
from llmtuner.config import ConfigError, load_config
from llmtuner.dataset import DatasetError
from llmtuner.store import Store, StoreError

INSTALL_HINT = (
    "llmtuner mcp needs the 'mcp' package: run it with "
    "`uv run --with 'mcp>=2,<3' llmtuner mcp` or install it with `pip install 'llmtuner[mcp]'`"
)

# Errors the library raises on purpose; they reach the agent as readable tool errors.
_LIBRARY_ERRORS = (ConfigError, DatasetError, StoreError)

MAX_TEXT_CHARS = 4000

# An agent may repeat a run at most this many times; a person can ask for more through the CLI.
MAX_TOOL_TRIALS = 10


def _clip(text: str | None) -> str | None:
    if text is not None and len(text) > MAX_TEXT_CHARS:
        return text[:MAX_TEXT_CHARS] + f"... [truncated, {len(text) - MAX_TEXT_CHARS} more characters]"
    return text


def build_server(store_home=None):
    """Build the MCP server. Raises ImportError if `mcp` 2.x is not installed.

    `store_home` is the store folder every tool uses (default: the usual store location).
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(
        "llmtuner",
        instructions=(
            "Local LLM eval harness. Call describe_project first to see a project's targets, datasets"
            " and graders, then run_eval, then read get_report and show_cases. Pass names from"
            " describe_project; commands are never accepted here."
        ),
    )

    def guarded(fn):
        """Turn the library's expected errors into tool errors the agent can read."""
        def wrapper(*args, **kwargs):
            try:
                return report.json_safe(fn(*args, **kwargs))
            except _LIBRARY_ERRORS as exc:
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__annotations__ = fn.__annotations__
        wrapper.__wrapped__ = fn
        return wrapper

    def tool(fn):
        server.tool()(guarded(fn))
        return fn

    @tool
    def describe_project(project_dir: str) -> dict[str, Any]:
        """Show what a project can be evaluated with. Use this first.

        `project_dir` is the directory holding the project's llmtuner.toml. Returns the project
        name, its targets (the systems under test, with the command each one runs and its
        timeout), its datasets (with the graders that apply to each) and its graders. Use the
        names it returns as `target` and `dataset` in run_eval.
        """
        return load_config(project_dir).describe()

    @tool
    def run_eval(
        project_dir: str,
        target: str,
        dataset: str,
        split: str = "test",
        note: str | None = None,
        limit: int | None = None,
        trials: int = 1,
    ) -> dict[str, Any]:
        """Run one target on one dataset split and return the report for the new run.

        WARNING: this runs the target's command once per case (plus any judge graders), so it
        can be slow and can spend model usage. Try `limit` (run only the first N cases) first
        when unsure. `trials` (1 to 10) repeats every case that many times to measure how stable
        the target is; it MULTIPLIES the number of calls and the cost, so keep it at 1 unless
        the user asked for repeats; `trials` must be a whole number from 1 to 10. With repeats
        the result has a "trials" section: read `trials.passed_all` (cases that passed every
        trial), `trials.passed_any` and `trials.per_trial`. The top-level `pass_rate` and
        `n_passed` then describe a single trial (`figures_describe` says which), not the group.
        `target` and `dataset` are names from describe_project; the command comes
        from the project's llmtuner.toml, never from this call. `split` is "test" (for
        reporting), "dev" (for tuning) or "all". The result is the same as get_report: the pass
        rate with its 95% interval, how each grader agrees with human labels, the detection
        floor (the smallest change this many cases can show; smaller moves are noise), and the
        failed case ids. The run id is in result["run"]["id"]; read failures with show_cases.
        """
        if not isinstance(trials, int) or isinstance(trials, bool) or not 1 <= trials <= MAX_TOOL_TRIALS:
            raise ConfigError(f"trials must be a whole number from 1 to {MAX_TOOL_TRIALS}")
        config = load_config(project_dir)
        with Store(store_home) as store:
            run_id = runner.run_eval(config, store, target, dataset, split=split, note=note, limit=limit,
                                     trials=trials)
            return report.build_report(store, run_id)

    @tool
    def get_report(run_id: int) -> dict[str, Any]:
        """Report on a stored run: use it to read results without re-running anything.

        Returns the pass rate (point estimate and 95% interval), per-grader pass counts, the
        detection floor (changes smaller than this are noise at this case count), how each grader
        agrees with human labels (kappa near 1 means it can be trusted; "not measured" means no
        human has labelled yet), the failed case ids, and a comparison against the project's
        baseline run when one has been set. For a run in a group of repeated trials, read
        `trials.passed_all` (cases that passed every trial), `trials.passed_any` and
        `trials.per_trial`; the top-level `pass_rate` and `n_passed` then describe a single
        trial, and `figures_describe` says which one.
        """
        with Store(store_home) as store:
            return report.build_report(store, run_id)

    @tool
    def compare_runs(baseline_run_id: int, candidate_run_id: int) -> dict[str, Any]:
        """Compare two runs case by case: did the candidate get better or worse?

        Only cases present in both runs count. Returns both pass rates, the difference with a
        95% interval, the case ids that got worse and better, a significance p-value, the share
        that got worse, the detection floor, a plain verdict ("candidate is better", "candidate
        is worse" or "no measurable difference"), and warnings (for example the dataset or
        graders changed between runs, which makes the comparison unreliable).
        """
        with Store(store_home) as store:
            return report.compare_runs(store, baseline_run_id, candidate_run_id)

    @tool
    def check_gate(run_id: int, baseline_run_id: int | None = None, max_drop: float = 0.03) -> dict[str, Any]:
        """Check whether a run regressed: use this as a pass/fail release check.

        Fails (ok=false) when the pass rate fell more than `max_drop` (a fraction, 0.03 = 3
        points) below the baseline. The baseline is `baseline_run_id` if given, otherwise the one
        a person set for this project and dataset; with neither, it fails with a reason. It also
        fails (ok=false) when the candidate's or the baseline's repeated trials did not all
        finish, and the comparison with its warnings is still returned. The "note" field warns when
        there are too few cases for the gate to see drops that small.
        """
        with Store(store_home) as store:
            return report.gate(store, run_id, baseline_run_id, max_drop)

    @tool
    def list_runs(project: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """List recent stored runs, newest first, to find a run id.

        Optionally filter by project name. Each row has the run id, project, dataset, target,
        model, split, note, start and finish times (unix seconds), the number of cases and how
        many passed.
        """
        with Store(store_home) as store:
            return store.list_runs(project, limit)

    @tool
    def show_cases(run_id: int, failed_only: bool = True, limit: int = 20) -> list[dict[str, Any]]:
        """Read the individual cases of a run, to see why something failed.

        By default only failed cases are shown. For each case: case_id, passed, input, output
        (both cut to 4000 characters), error (set when the target crashed or timed out), and its
        grades (grader name, passed, error, detail). A grade with error=true means the grader
        itself failed to produce a verdict; it is not a verdict on the output.
        """
        with Store(store_home) as store:
            store.get_run(run_id)  # raises StoreError for an unknown run
            grades: dict[str, list[dict]] = {}
            for g in store.get_grades(run_id):
                grades.setdefault(g["case_id"], []).append({
                    "grader": g["grader"],
                    "passed": bool(g["passed"]),
                    "error": bool(g["error"]),
                    "detail": g["detail"],
                })
            rows = []
            for case in store.get_cases(run_id):
                if len(rows) >= max(0, limit):
                    break
                if failed_only and case["passed"]:
                    continue
                rows.append({
                    "case_id": case["case_id"],
                    "passed": bool(case["passed"]),
                    "input": _clip(case["input"]),
                    "output": _clip(case["output"]),
                    "error": case["error"],
                    "grades": grades.get(case["case_id"], []),
                })
            return rows

    return server


def main(store_home=None) -> int:
    try:
        server = build_server(store_home)
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
