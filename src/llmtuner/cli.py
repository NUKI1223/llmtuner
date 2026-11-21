"""Command line interface. A thin wrapper over the library functions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

from llmtuner import __version__, report
from llmtuner.config import ConfigError, load_config
from llmtuner.dataset import DatasetError
from llmtuner.runner import run_eval
from llmtuner.store import CASE_LABEL, Store, StoreError

_SHOW_CHARS = 2000


def _emit(args, data, text: str) -> None:
    print(json.dumps(report.json_safe(data), indent=2, default=str, allow_nan=False) if args.json else text)


def _resolve_run(store: Store, args) -> int:
    """The run given on the command line, else the latest run of the project in this folder."""
    if getattr(args, "run", None) is not None:
        return args.run
    try:
        project = load_config(args.config).project
    except ConfigError:
        project = None
    return store.latest_run_id(project)


def _clip(text: str | None, limit: int = _SHOW_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


# --- commands ----------------------------------------------------------------


def cmd_info(args, store: Store) -> int:
    config = load_config(args.config)
    data = {"store": str(store.path), **config.describe()}
    lines = [f"project  {config.project}", f"store    {store.path}", "targets"]
    lines += [f"  {t.name:<16} {' '.join(t.command)}" for t in config.targets.values()]
    lines.append("datasets")
    for d in config.datasets.values():
        lines.append(f"  {d.name:<16} {d.path.name}  graders: {', '.join(g.name for g in config.graders_for(d))}")
    _emit(args, data, "\n".join(lines))
    return 0


def cmd_run(args, store: Store) -> int:
    config = load_config(args.config)

    def progress(case, passed: bool) -> None:
        if not args.json:
            print("." if passed else "x", end="", file=sys.stderr, flush=True)

    def next_trial(number: int, total: int) -> None:
        if not args.json and total > 1:
            print(("\n" if number > 1 else "") + f"trial {number}/{total} ", end="", file=sys.stderr, flush=True)

    run_id = run_eval(config, store, args.target, args.dataset, split=args.split, note=args.note,
                      jobs=args.jobs, limit=args.limit, on_case=progress, trials=args.trials,
                      on_trial=next_trial)
    if not args.json:
        print(file=sys.stderr)
    if args.set_baseline:
        store.set_baseline(run_id)
    result = report.build_report(store, run_id)
    _emit(args, result, report.format_report(result))
    return 0


def cmd_report(args, store: Store) -> int:
    result = report.build_report(store, _resolve_run(store, args))
    _emit(args, result, report.format_report(result))
    return 0


def cmd_compare(args, store: Store) -> int:
    result = report.compare_runs(store, args.baseline, args.candidate)
    _emit(args, result, report.format_compare(result))
    return 0


def cmd_gate(args, store: Store) -> int:
    result = report.gate(store, _resolve_run(store, args), args.baseline, args.max_drop)
    _emit(args, result, report.format_gate(result))
    return 0 if result["ok"] else 1


def cmd_baseline(args, store: Store) -> int:
    run = store.set_baseline(args.run)
    # A group of repeated trials is stored as its first run, whichever trial was named.
    stored = store.get_baseline(run["project"], run["dataset"])
    first = store.get_run(stored)
    trials = (first["trials_planned"] or len(store.group_runs(stored))) if first["trial_group"] is not None else 1
    where = f"{run['project']} / {run['dataset']}"
    if trials > 1:
        named = f" (run {args.run} is one trial of it)" if stored != args.run else ""
        text = f"run {stored}, the whole group of {trials} trials, is now the baseline for {where}{named}"
    else:
        text = f"run {stored} is now the baseline for {where}"
    _emit(args, {"project": run["project"], "dataset": run["dataset"], "baseline_run": stored, "trials": trials}, text)
    return 0


def cmd_runs(args, store: Store) -> int:
    project = args.project
    if project is None and not args.all:
        try:
            project = load_config(args.config).project
        except ConfigError:
            project = None
    runs = store.list_runs(project, args.limit)
    lines = []
    for r in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started_at"]))
        rate = f"{r['n_passed']}/{r['n']}" if r["n"] else "empty"
        state = "" if r["finished_at"] else "  (unfinished)"
        # Repeated trials are ordinary runs; say which trial each one is.
        trial = f"  trial {r['trial']}/{r['trials_planned']} of group {r['trial_group']}" if r["trial_group"] else ""
        lines.append(f"{r['id']:>4}  {when}  {r['project']:<18} {r['dataset']:<14} {r['target']:<16}"
                     f" {r['split']:<5} {rate:>8}{state}{trial}" + (f"  {r['note']}" if r["note"] else ""))
    _emit(args, runs, "\n".join(lines) or "no runs recorded yet")
    return 0


def cmd_show(args, store: Store) -> int:
    run_id = _resolve_run(store, args)
    store.get_run(run_id)  # raises StoreError for an unknown run
    grades: dict[str, list[dict]] = {}
    for g in store.get_grades(run_id):
        grades.setdefault(g["case_id"], []).append(g)
    cases = store.get_cases(run_id)
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            raise StoreError(f"run {run_id} has no case '{args.case}'")
    if args.failed:
        cases = [c for c in cases if not c["passed"]]
    cases = cases[: args.limit]
    data = [{**c, "grades": grades.get(c["case_id"], [])} for c in cases]
    blocks = []
    for c in data:
        lines = [f"=== {c['case_id']}  {'PASS' if c['passed'] else 'FAIL'}  ({c['duration_s']:.1f}s)",
                 "--- input", _clip(c["input"])]
        if c["expected"] is not None:
            lines += ["--- expected", _clip(c["expected"])]
        lines += ["--- output", _clip(c["output"]) or "(empty)", "--- grades"]
        for g in c["grades"]:
            mark = "error" if g["error"] else ("pass" if g["passed"] else "FAIL")
            lines.append(f"  {g['grader']:<20} {mark}" + (f"  {g['detail']}" if g["detail"] else ""))
        blocks.append("\n".join(lines))
    _emit(args, data, "\n\n".join(blocks) or "no cases to show")
    return 0


def cmd_label(args, store: Store) -> int:
    run_id = _resolve_run(store, args)
    run = store.get_run(run_id)
    grader = args.grader or CASE_LABEL
    spec = next((g for g in run["graders"] if g["name"] == grader), None)
    if args.grader and spec is None:
        names = ", ".join(g["name"] for g in run["graders"])
        raise StoreError(f"run {run_id} has no grader '{args.grader}' (it has: {names})")

    if args.case:  # one label, no prompt: usable from scripts
        if args.verdict is None:
            raise ConfigError("give --pass or --fail together with --case")
        store.set_label(run_id, args.case, args.verdict, grader, args.note)
        _emit(args, {"run": run_id, "case": args.case, "grader": grader, "passed": args.verdict, "note": args.note},
              f"labelled {args.case} as {'pass' if args.verdict else 'fail'}")
        return 0

    done = {l["case_id"] for l in store.get_labels(run_id) if l["grader"] == grader}
    cases = [c for c in store.get_cases(run_id) if c["case_id"] not in done and c["error"] is None]
    random.Random(run_id).shuffle(cases)  # a fixed order per run, not sorted by grader verdict
    if args.sample:
        cases = cases[: args.sample]
    if not cases:
        print("nothing left to label for this run")
        return 0

    question = "Is this output acceptable overall?"
    if spec and spec["type"] == "judge":
        question = f"The grader asks: \"{spec['question']}\"  Does this output PASS that check?"
    elif spec:
        question = f"Should this output PASS the '{spec['name']}' check?"
    print(f"Labelling run {run_id}: {len(cases)} outputs. The grader's own verdict is hidden on purpose.\n")
    labelled = 0
    for number, case in enumerate(cases, start=1):
        print(f"=== {number}/{len(cases)}  case {case['case_id']}\n--- input\n{_clip(case['input'])}")
        if case["context"]:
            print(f"--- context\n{_clip(case['context'])}")
        if case["expected"] is not None:
            print(f"--- expected\n{_clip(case['expected'])}")
        print(f"--- output\n{_clip(case['output']) or '(empty)'}\n")
        try:
            answer = input(f"{question}\n[p]ass / [f]ail / [s]kip / [q]uit: ").strip().lower()[:1]
            if answer == "q":
                break
            if answer not in ("p", "f"):
                continue
            note = input("note (enter to skip): ").strip() or None
        except EOFError:
            break
        store.set_label(run_id, case["case_id"], answer == "p", grader, note)
        labelled += 1
        print()
    print(f"\nsaved {labelled} labels. See the agreement with: llmtuner report {run_id}")
    return 0


def cmd_rank(args, store: Store) -> int:
    project = args.project or load_config(args.config).project
    result = report.rank(store, project, args.dataset, args.split)
    _emit(args, result, report.format_rank(result))
    return 0


def cmd_ui(args, store: Store) -> int:
    from llmtuner import ui

    return ui.main(args.store, args.port, not args.no_open)


def cmd_mcp(args, store: Store) -> int:
    from llmtuner import mcp_server

    return mcp_server.main(args.store)


# --- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", metavar="PATH",
                        help="project config file or folder (default: ./llmtuner.toml)")
    common.add_argument("--store", metavar="DIR", help="store folder (default: $LLMTUNER_HOME or ~/.local/share/llmtuner)")
    common.add_argument("--json", action="store_true", help="print machine-readable JSON")

    parser = argparse.ArgumentParser(prog="llmtuner", description="A small eval harness for LLM systems.")
    parser.add_argument("--version", action="version", version=f"llmtuner {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    add("info", cmd_info, "Show this project's targets, datasets and graders.")

    p = add("run", cmd_run, "Run a target on a dataset, grade every output and store the result.")
    p.add_argument("target")
    p.add_argument("dataset")
    p.add_argument("--split", choices=("test", "dev", "all"), default="test",
                   help="which cases to run (default: test; tune on dev, report on test)")
    p.add_argument("--note", help="a short note stored with the run, e.g. what changed")
    p.add_argument("--jobs", type=_positive_int, default=1, help="cases to run at the same time (default: 1)")
    p.add_argument("--limit", type=_positive_int, help="run only the first N cases")
    p.add_argument("--trials", type=_positive_int, default=1,
                   help="run every case this many times to measure stability (default: 1); "
                        "multiplies the number of target calls")
    p.add_argument("--set-baseline", action="store_true", help="make this run the baseline for its dataset")

    p = add("report", cmd_report, "Show the four numbers for a run (default: the latest run).")
    p.add_argument("run", nargs="?", type=int)

    p = add("compare", cmd_compare, "Compare two runs case by case.")
    p.add_argument("baseline", type=int)
    p.add_argument("candidate", type=int)

    p = add("gate", cmd_gate, "Exit 1 if a run's pass rate fell too far below the baseline. For CI.")
    p.add_argument("run", nargs="?", type=int)
    p.add_argument("--baseline", type=int, help="baseline run id (default: the stored baseline)")
    p.add_argument("--max-drop", type=float, default=0.03, help="allowed drop in pass rate (default: 0.03)")

    p = add("baseline", cmd_baseline, "Make a run the baseline for its project and dataset.")
    p.add_argument("run", type=int)

    p = add("runs", cmd_runs, "List recorded runs (default: this folder's project).")
    p.add_argument("--project")
    p.add_argument("--all", action="store_true", help="list runs from every project")
    p.add_argument("--limit", type=_positive_int, default=20)

    p = add("show", cmd_show, "Read the inputs, outputs and grades of a run.")
    p.add_argument("run", nargs="?", type=int)
    p.add_argument("--failed", action="store_true", help="only cases that failed")
    p.add_argument("--case", help="only this case id")
    p.add_argument("--limit", type=_positive_int, default=50)

    p = add("label", cmd_label, "Record human pass/fail labels for a run, to check the graders against.")
    p.add_argument("run", nargs="?", type=int)
    p.add_argument("--grader", help="label one grader's question instead of the overall verdict")
    p.add_argument("--sample", type=_positive_int, help="label only N outputs")
    p.add_argument("--case", help="label this one case without prompting (needs --pass or --fail)")
    verdict = p.add_mutually_exclusive_group()
    verdict.add_argument("--pass", dest="verdict", action="store_const", const=True, default=None)
    verdict.add_argument("--fail", dest="verdict", action="store_const", const=False)
    p.add_argument("--note")

    p = add("rank", cmd_rank, "Rank this project's targets on a dataset by pass rate and cost per passed case.")
    p.add_argument("dataset")
    p.add_argument("--project", help="project name (default: this folder's project)")
    p.add_argument("--split", choices=("test", "dev", "all"), default="test")

    p = add("ui", cmd_ui, "Open the local web page for browsing runs and labelling outputs.")
    p.add_argument("--port", type=_port, default=8765, help="port on 127.0.0.1 (default: 8765; 0 picks a free one)")
    p.add_argument("--no-open", action="store_true", help="do not open the browser")

    add("mcp", cmd_mcp, "Start the MCP server on stdio, so agents can use the harness.")
    return parser


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port number") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("the port must be between 0 and 65535")
    return port


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with Store(args.store) as store:
            return args.func(args, store)
    except (ConfigError, DatasetError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
