"""Reports: the four numbers a result needs before it means anything.

1. the pass rate, with its interval
2. how well each grader agrees with human labels
3. the detection floor (the smallest change this many cases can show)
4. the share of cases that got worse than the baseline

All functions return plain dicts so the CLI and the MCP server share them.
"""

from __future__ import annotations

import json
import math
import statistics

from llmtuner import stats
from llmtuner.config import TARGET_GRADER
from llmtuner.store import CASE_LABEL, Store

# Cohen's kappa bars for trusting a grader (Landis & Koch style thresholds).
KAPPA_GATE = 0.6
KAPPA_UNSUPERVISED = 0.8
MIN_LABELS = 20


def json_safe(value):
    """Replace NaN/inf (undefined statistics) with None so the result is valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _summary(run: dict) -> dict:
    keys = ("id", "project", "dataset", "dataset_hash", "target", "command", "model",
            "tool_version", "split", "note", "started_at", "finished_at")
    return {k: run[k] for k in keys}


def _trial_runs(store: Store, run_id: int) -> list[dict]:
    """The finished trials of the group `run_id` belongs to; the run alone if it has no group."""
    runs = store.group_runs(run_id)
    if runs[0]["trial_group"] is None:
        return runs
    return [r for r in runs if r["finished_at"] is not None]


def _case_counts(store: Store, run_id: int) -> dict[str, tuple[int, int]]:
    """For each case: (trials it passed, trials it ran), over the finished trials of the group."""
    counts: dict[str, tuple[int, int]] = {}
    for run in _trial_runs(store, run_id):
        for case in store.get_cases(run["id"]):
            passes, trials = counts.get(case["case_id"], (0, 0))
            counts[case["case_id"]] = (passes + int(bool(case["passed"])), trials + 1)
    return counts


def _passed(store: Store, run_id: int) -> dict[str, bool]:
    """Whether each case passed. In a group of repeated trials a case passes only if it passed
    every trial, so comparisons, the gate and the ranking measure reliability, not luck."""
    return {case_id: passes == trials for case_id, (passes, trials) in _case_counts(store, run_id).items()}


def trial_summary(store: Store, run_id: int) -> dict | None:
    """How stable a target was over repeated trials. None for a run that is not part of a group.

    The case is the unit throughout: `passed_all` and `passed_any` count cases, and the interval
    on the per-trial rate treats each case as one observation, so repeating a run never makes
    a result look more certain than its number of cases allows.
    """
    runs = store.group_runs(run_id)
    if runs[0]["trial_group"] is None:
        return None
    finished = [r for r in runs if r["finished_at"] is not None]
    planned = runs[0]["trials_planned"] or len(runs)
    rows: dict[str, dict] = {}
    every_case: list[dict] = []
    for run in finished:
        cases = store.get_cases(run["id"])
        every_case += cases
        errored = {g["case_id"] for g in store.get_grades(run["id"]) if g["grader"] == TARGET_GRADER}
        for case in cases:
            row = rows.setdefault(case["case_id"], {"case_id": case["case_id"], "passes": 0,
                                                    "trials": 0, "target_errors": 0})
            row["trials"] += 1
            row["passes"] += int(bool(case["passed"]))
            row["target_errors"] += int(case["case_id"] in errored)
    cases = list(rows.values())
    n = len(cases)
    passed_all = sum(1 for r in cases if r["passes"] == r["trials"])
    passed_any = sum(1 for r in cases if r["passes"] > 0)
    passes, total = sum(r["passes"] for r in cases), sum(r["trials"] for r in cases)
    return {
        "group": runs[0]["trial_group"],
        "runs": [r["id"] for r in runs],
        "planned": planned,
        "finished": len(finished),
        "complete": len(finished) == planned,
        "n": n,
        "cases": cases,
        "passed_all": {"n_passed": passed_all, "rate": stats.wilson_interval(passed_all, n).as_dict()},
        "passed_any": {"n_passed": passed_any, "rate": stats.wilson_interval(passed_any, n).as_dict()},
        "per_trial": {"n_passed": passes, "n": total,
                      "rate": stats.wilson_interval(passes / total * n if total else 0, n).as_dict()},
        # A trial that ended in a target error (timeout, budget) says nothing about stability:
        # a case is unstable only when its graded trials disagree.
        "unstable_cases": [r["case_id"] for r in cases if 0 < r["passes"] < r["trials"] - r["target_errors"]],
        "error_cases": [r["case_id"] for r in cases if r["target_errors"]],
        "target_error_trials": sum(r["target_errors"] for r in cases),
        "usage": usage(every_case, passes),
    }


def compare_runs(store: Store, baseline_id: int, candidate_id: int) -> dict:
    """Paired comparison of two runs on the cases they share."""
    base, cand = store.get_run(baseline_id), store.get_run(candidate_id)
    base_passed, cand_passed = _passed(store, baseline_id), _passed(store, candidate_id)
    paired = stats.paired_compare(base_passed, cand_passed)

    warnings = []
    if (base["project"], base["dataset"]) != (cand["project"], cand["dataset"]):
        warnings.append("the runs are from different projects or datasets")
    elif base["dataset_hash"] != cand["dataset_hash"]:
        warnings.append("the dataset file changed between the two runs")
    if base["graders"] != cand["graders"]:
        warnings.append("the graders changed between the two runs")
    if paired.n < max(len(base_passed), len(cand_passed)):
        warnings.append(f"only {paired.n} cases are in both runs; the rest were ignored")
    base_trials, cand_trials = len(_trial_runs(store, baseline_id)), len(_trial_runs(store, candidate_id))
    # A group whose trials did not all finish is compared on what it has, and says so.
    for side, run_id in (("baseline", baseline_id), ("candidate", candidate_id)):
        members = store.group_runs(run_id)
        if members[0]["trial_group"] is not None:
            planned = members[0]["trials_planned"] or len(members)
            finished = sum(1 for m in members if m["finished_at"] is not None)
            if finished < planned:
                warnings.append(f"only {finished} of the {side}'s {planned} planned trials finished,"
                                " so its result is provisional")
    if base_trials != cand_trials:
        warnings.append(f"the baseline ran {base_trials} trial(s) per case and the candidate {cand_trials};"
                        " a case must pass every trial, which is stricter with more trials,"
                        " so the two sides are not like for like")
    # With repeated trials each case can also be tested on its own: did its behaviour change?
    per_case = None
    if max(base_trials, cand_trials) > 1:
        before, after = _case_counts(store, baseline_id), _case_counts(store, candidate_id)
        per_case = [{"case_id": case_id,
                     "baseline_passes": before[case_id][0], "baseline_trials": before[case_id][1],
                     "candidate_passes": after[case_id][0], "candidate_trials": after[case_id][1],
                     "p_value": stats.fisher_exact(*before[case_id], *after[case_id])}
                    for case_id in sorted(set(before) & set(after))]

    if paired.n == 0:
        verdict = "no shared cases"
    elif not paired.measurable:
        verdict = "no measurable difference"
    else:
        verdict = "candidate is better" if paired.diff.point > 0 else "candidate is worse"

    result = paired.as_dict()
    result.update(
        baseline=_summary(base),
        candidate=_summary(cand),
        share_worse=len(paired.regressions) / paired.n if paired.n else None,
        # The smallest change the exact McNemar test could show on this many shared cases.
        detection_floor=stats.paired_detection_floor(paired.n),
        verdict=verdict,
        warnings=warnings,
        baseline_trials=base_trials,
        candidate_trials=cand_trials,
        per_case=per_case,
    )
    return result


def _trust(check: stats.Agreement) -> str:
    if check.n < MIN_LABELS:
        return f"too few labels ({check.n}) to say; label at least {MIN_LABELS}"
    if check.kappa is None:
        return "undefined: every label is on one side; label some failing outputs too"
    if check.kappa >= KAPPA_UNSUPERVISED:
        return "reliable enough to run unsupervised"
    if check.kappa >= KAPPA_GATE:
        return "reliable enough to gate a release"
    return "not reliable: do not gate on this grader"


def grader_checks(store: Store, run_id: int) -> list[dict]:
    """Agreement between each grader and the human labels recorded for this run."""
    verdicts = {(g["case_id"], g["grader"]): bool(g["passed"])
                for g in store.get_grades(run_id) if not g["error"]}
    for case in store.get_cases(run_id):
        verdicts[(case["case_id"], CASE_LABEL)] = bool(case["passed"])

    pairs: dict[str, list[tuple[bool, bool]]] = {}
    for label in store.get_labels(run_id):
        key = (label["case_id"], label["grader"])
        if key in verdicts:
            pairs.setdefault(label["grader"], []).append((bool(label["passed"]), verdicts[key]))

    checks = []
    for grader, rows in pairs.items():
        check = stats.agreement([h for h, _ in rows], [g for _, g in rows])
        checks.append({
            "grader": "overall verdict" if grader == CASE_LABEL else grader,
            **check.as_dict(),
            "trust": _trust(check),
        })
    return checks


def _number(meta: dict, key: str) -> float | None:
    value = meta.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:  # an integer too large for a float
        return None


def usage(cases: list[dict], n_passed: int) -> dict:
    """Cost, tokens and speed of a run, from what its target reported for each case.

    Anything the target did not report is None, never 0. `cost_per_pass` is the cost of the
    whole run divided by the cases that passed: what one correct answer costs. Costs in
    different units are never added: `mixed_units` is then true and the cost fields are None.
    """
    metas = [c.get("meta") or {} for c in cases]
    costs = [v for m in metas if (v := _number(m, "cost")) is not None]
    units = sorted({str(m.get("cost_unit") or "unknown") for m in metas if _number(m, "cost") is not None})
    mixed = len(units) > 1
    prompt = [v for m in metas if (v := _number(m, "prompt_tokens")) is not None]
    completion = [v for m in metas if (v := _number(m, "completion_tokens")) is not None]
    cost = sum(costs) if costs and not mixed else None
    return {
        "cost": cost,
        "cost_unit": None if mixed or not units or units == ["unknown"] else units[0],
        "cost_units": units,
        "mixed_units": mixed,
        "cases_with_cost": len(costs),
        "cost_per_pass": cost / n_passed if cost is not None and n_passed else None,
        "prompt_tokens": sum(prompt) if prompt else None,
        "completion_tokens": sum(completion) if completion else None,
        "median_duration_s": statistics.median(c["duration_s"] for c in cases) if cases else None,
    }


def build_report(store: Store, run_id: int) -> dict:
    run = store.get_run(run_id)
    cases = store.get_cases(run_id)
    grades = store.get_grades(run_id)
    passed = [bool(c["passed"]) for c in cases]
    n, n_passed = len(passed), sum(passed)

    graders = []
    for spec in run["graders"]:
        rows = [g for g in grades if g["grader"] == spec["name"]]
        graders.append({
            **spec,
            "n": len(rows),
            "n_passed": sum(g["passed"] for g in rows),
            "errors": sum(g["error"] for g in rows),
        })

    checks = grader_checks(store, run_id)
    checked = {c["grader"] for c in checks}
    unchecked_judges = [s["name"] for s in run["graders"] if s["type"] == "judge" and s["name"] not in checked]

    baseline_id = store.get_baseline(run["project"], run["dataset"])
    # Every trial of the baseline's group is the baseline, not something to compare with it.
    own_group = {r["id"] for r in store.group_runs(run_id)}
    baseline = compare_runs(store, baseline_id, run_id) if baseline_id is not None and baseline_id not in own_group else None

    group = trial_summary(store, run_id)
    describes = "this run" if group is None else (
        f"trial {group['runs'].index(run_id) + 1} of {group['planned']} (run {run_id}) alone;"
        " the figures for the whole group of trials are under 'trials'")
    return {
        "run": _summary(run),
        # What the top-level figures below (n, n_passed, pass_rate, failed_cases, usage, graders) cover.
        "figures_describe": describes,
        "n": n,
        "n_passed": n_passed,
        "pass_rate": stats.pass_rate(passed).as_dict(),
        # The same rule compare and gate use, so one run never shows two different floors.
        "detection_floor": stats.paired_detection_floor(n),
        "graders": graders,
        "target_errors": sum(1 for g in grades if g["grader"] == TARGET_GRADER),
        "timeouts": sum(c["timed_out"] for c in cases),
        "failed_cases": [c["case_id"] for c in cases if not c["passed"]],
        "usage": usage(cases, n_passed),
        "grader_checks": checks,
        "unchecked_judges": unchecked_judges,
        "is_baseline": baseline_id in own_group,
        "baseline": baseline,
        # Set when this run is one of several repeated trials; everything above is this trial alone.
        "trials": group,
    }


def rank(store: Store, project: str, dataset: str, split: str = "test") -> dict:
    """Rank the targets of one project on one dataset, each by its newest run on the reference cases.

    The reference follows the newest finished run: its dataset file and graders, and among
    runs on those the set of case ids that holds the most cases. Targets with no run in that
    group are listed as skipped. Each row carries a paired verdict against the leader, so
    a higher position alone is never read as "better".
    """
    result = {"project": project, "dataset": dataset, "split": split, "rows": [], "skipped": []}
    # A group of repeated trials is one entry, represented by its first run.
    runs, incomplete, seen = [], [], set()
    for first in store.groups_without_finished_trial(project, dataset, split):
        incomplete.append({**first, "trials": first["trials_planned"] or 1, "case_ids": frozenset()})
    for run in store.finished_runs(project, dataset, split):  # newest first
        if run["trial_group"] is None:
            runs.append({**run, "trials": 1})
            continue
        if run["trial_group"] in seen:
            continue
        seen.add(run["trial_group"])
        members = store.group_runs(run["id"])
        planned = members[0]["trials_planned"] or len(members)
        entry = {**members[0], "trials": planned, "case_ids": frozenset(_passed(store, members[0]["id"]))}
        if sum(1 for m in members if m["finished_at"] is not None) == planned:
            runs.append(entry)
        else:
            incomplete.append(entry)
    if not runs:
        result["skipped"] = [{"run": r["id"], "target": r["target"],
                              "reason": "its repeated trials did not all finish"} for r in incomplete]
        return result

    def group(run: dict):
        return (run["dataset_hash"], json.dumps(run["graders"], sort_keys=True), run["trials"], run["case_ids"])

    # The newest run says which dataset file, graders and number of trials are current, so
    # editing the dataset moves the ranking on to the new file. Among runs on that setup the
    # largest case set wins, so a quick --limit run does not hide a target's full run.
    current = runs[0]
    same_setup = [r for r in runs if r["dataset_hash"] == current["dataset_hash"]
                  and r["graders"] == current["graders"] and r["trials"] == current["trials"]]
    reference = max(same_setup, key=lambda r: (len(r["case_ids"]), r["id"]))
    ref_group = group(reference)
    newest: dict[str, dict] = {}
    for run in runs:  # newest first
        if group(run) == ref_group:
            newest.setdefault(run["target"], run)
    rows = []
    for run in newest.values():
        rep = build_report(store, run["id"])
        row = {"run": run["id"], "target": run["target"], "model": run["model"], "trials": run["trials"]}
        if rep["trials"]:  # repeated trials: a case counts only if it passed every one
            summary = rep["trials"]
            row.update(n=summary["n"], n_passed=summary["passed_all"]["n_passed"],
                       pass_rate=summary["passed_all"]["rate"], usage=summary["usage"])
        else:
            row.update(n=rep["n"], n_passed=rep["n_passed"], pass_rate=rep["pass_rate"], usage=rep["usage"])
        rows.append(row)
    for run in runs + incomplete:  # the newest run of each target that has none in the reference group
        if run["target"] in newest or any(s["target"] == run["target"] for s in result["skipped"]):
            continue
        if run in incomplete:
            reason = "its repeated trials did not all finish"
        elif run["dataset_hash"] != reference["dataset_hash"]:
            reason = "ran on a different version of the dataset file"
        elif run["graders"] != reference["graders"]:
            reason = "ran with different graders"
        elif run["trials"] != reference["trials"]:
            reason = "ran a different number of trials per case"
        else:
            reason = "ran on a different set of cases (for example with --limit)"
        result["skipped"].append({"run": run["id"], "target": run["target"], "reason": reason})

    # Costs order rows only when they are all in one unit; a row that reported no cost just sorts last.
    costed = {tuple(row["usage"]["cost_units"]) for row in rows if row["usage"]["cost_units"]}
    mixed = any(row["usage"]["mixed_units"] for row in rows)
    by_cost = not mixed and len(costed) <= 1

    def order(row: dict):
        rate = -row["pass_rate"]["point"]
        if not by_cost:
            return (rate, row["run"])
        cost = row["usage"]["cost_per_pass"]
        return (rate, cost is None, cost or 0.0, row["run"])

    rows.sort(key=order)
    if not by_cost:
        result["cost_note"] = "costs are in different units, so they were not used for ordering"
    verdicts = {"candidate is worse": "measurably worse than the leader",
                "candidate is better": "measurably better than the leader"}
    leader = rows[0]["run"]
    for row in rows:
        if row["run"] == leader:
            row["vs_leader"] = "leader"
            continue
        cmp = compare_runs(store, leader, row["run"])
        row["vs_leader"] = verdicts.get(cmp["verdict"], "not measurably different from the leader")
        row["p_value"] = cmp["p_value"]
    result["rows"] = rows
    return result


def gate(store: Store, run_id: int, baseline_id: int | None = None, max_drop: float = 0.03) -> dict:
    """Fail when the pass rate fell more than `max_drop` below the baseline."""
    run = store.get_run(run_id)
    if baseline_id is None:
        baseline_id = store.get_baseline(run["project"], run["dataset"])
    if baseline_id is None:
        return {"ok": False, "reason": "no baseline set for this project and dataset", "comparison": None}
    comparison = compare_runs(store, baseline_id, run_id)
    # A group whose trials did not all finish is provisional: the gate never passes one.
    for side, side_id, what in (("candidate", run_id, "it"), ("baseline", baseline_id, "the candidate")):
        members = store.group_runs(side_id)
        if members[0]["trial_group"] is None:
            continue
        planned = members[0]["trials_planned"] or len(members)
        finished = sum(1 for m in members if m["finished_at"] is not None)
        if finished < planned:
            return {"ok": False, "note": None, "comparison": comparison,
                    "reason": f"the {side}'s repeated trials did not all finish ({finished} of {planned}),"
                              f" so the gate cannot pass {what}"}
    if comparison["n"] == 0:
        return {"ok": False, "reason": "the run shares no cases with the baseline", "comparison": comparison}
    drop = -comparison["diff"]["point"]
    ok = drop <= max_drop
    repeated = max(comparison["baseline_trials"], comparison["candidate_trials"]) > 1
    measure = "share of cases passing every trial" if repeated else "pass rate"
    reason = (f"{measure} {'fell' if drop > 0 else 'rose'} {abs(drop) * 100:.1f} points against baseline run"
              f" {baseline_id}; allowed drop is {max_drop * 100:.1f}")
    floor = comparison["detection_floor"]
    note = None
    if floor is None:
        note = (f"with {comparison['n']} shared cases (fewer than 6), no difference can be shown,"
                " so this gate cannot tell a real regression from noise")
    elif floor > max_drop:
        note = (f"with {comparison['n']} cases, drops smaller than about {floor * 100:.0f} points"
                " cannot be told from noise, so this gate only catches large regressions")
    return {"ok": ok, "reason": reason, "note": note, "comparison": comparison}


# --- text formatting ---------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None or value != value else f"{value * 100:.1f}%"


def _ids(ids: list[str], limit: int = 8) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def _header(run: dict, n: int) -> list[str]:
    lines = [f"Run {run['id']} · {run['project']} · {run['dataset']} ({run['split']}, {n} cases)"
             f" · target {run['target']}"]
    meta = [f"command: {' '.join(run['command'])}"]
    if run["model"]:
        meta.append(f"model: {run['model']}")
    if run["tool_version"]:
        meta.append(f"tool: {run['tool_version']}")
    lines.append("  " + " · ".join(meta))
    if run["note"]:
        lines.append(f"  note: {run['note']}")
    return lines


def _cost(value: float | None, unit: str | None) -> str:
    if value is None:
        return "not reported"
    return f"{value:.6f}".rstrip("0").rstrip(".") + (f" {unit}" if unit else "")


def _usage_line(use: dict, n: int, group: bool = False) -> str:
    """One line of usage. `group` is for repeated trials, where each count is a case result
    (one case in one trial), not a case."""
    noun, per = ("case results", "passing case result") if group else ("cases", "passed case")
    parts = []
    if use["mixed_units"]:
        parts.append(f"cost reported in different units ({', '.join(use['cost_units'])}), so it is not added up")
    elif use["cost"] is None:
        parts.append("cost not reported by the target")
    else:
        partial = "" if use["cases_with_cost"] == n else f" (reported for {use['cases_with_cost']} of {n} {noun})"
        parts.append(f"cost {_cost(use['cost'], use['cost_unit'])}{partial}")
        parts.append(f"per {per} " + _cost(use["cost_per_pass"], use["cost_unit"])
                     if use["cost_per_pass"] is not None
                     else f"no case passed, so no cost per {per}" if not group
                     else f"no case result passed, so no cost per {per}")
    if use["prompt_tokens"] is not None or use["completion_tokens"] is not None:
        side = lambda v, name: f"{v:.0f} {name}" if v is not None else f"{name} not reported"  # noqa: E731
        parts.append(f"tokens {side(use['prompt_tokens'], 'in')}, {side(use['completion_tokens'], 'out')}")
    if use["median_duration_s"] is not None:
        parts.append(f"median {use['median_duration_s']:.1f}s per {noun[:-1]}")
    return " · ".join(parts)


def format_rank(result: dict) -> str:
    if not result["rows"]:
        return "\n".join([f"no finished runs for {result['project']} / {result['dataset']} ({result['split']})"]
                         + [f"skipped run {s['run']} ({s['target']}): {s['reason']}" for s in result["skipped"]])
    lines = [f"{result['project']} · {result['dataset']} ({result['split']}): newest comparable run of each target", ""]
    for place, row in enumerate(result["rows"], start=1):
        rate, use = row["pass_rate"], row["usage"]
        cost = (_cost(use["cost_per_pass"], use["cost_unit"]) if use["cost_per_pass"] is not None
                else "in different units" if use["mixed_units"] else "not reported")
        speed = f"{use['median_duration_s']:.1f}s" if use["median_duration_s"] is not None else "n/a"
        repeated = row.get("trials", 1) > 1
        lines.append(f"{place}. {row['target']:<22} {_pct(rate['point']):>6} ({row['n_passed']} of {row['n']},"
                     f" {_pct(rate['low'])} to {_pct(rate['high'])})  run {row['run']}"
                     + (f", passed all {row['trials']} trials" if repeated else ""))
        lines.append(f"   per {'passing case result' if repeated else 'passed case'}: {cost} · median {speed}"
                     f" · {row['vs_leader']}")
    if result.get("cost_note"):
        lines += ["", result["cost_note"]]
    lines += [f"skipped run {s['run']} ({s['target']}): {s['reason']}" for s in result["skipped"]]
    return "\n".join(lines)


def format_compare(cmp: dict) -> str:
    base, cand, diff = cmp["baseline"], cmp["candidate"], cmp["diff"]
    lines = [
        f"Baseline run {base['id']} ({base['target']}) vs candidate run {cand['id']} ({cand['target']})"
        f" on {cmp['n']} shared cases",
        ("  all trials   " if cmp.get("per_case") else "  pass rate    ")
        + f"{_pct(cmp['baseline_rate'])} -> {_pct(cmp['candidate_rate'])}"
        + (" of cases passed every trial" if cmp.get("per_case") else ""),
    ]
    if cmp["n"]:
        lines += [
            f"  difference   {diff['point'] * 100:+.1f} points, 95% bootstrap interval"
            f" {diff['low'] * 100:+.1f} to {diff['high'] * 100:+.1f}",
            f"  got worse    {len(cmp['regressions'])} of {cmp['n']}"
            + (f": {_ids(cmp['regressions'])}" if cmp["regressions"] else ""),
            f"  got better   {len(cmp['improvements'])} of {cmp['n']}"
            + (f": {_ids(cmp['improvements'])}" if cmp["improvements"] else ""),
            f"  McNemar p    {cmp['p_value']:.3f}",
        ]
    verdict = cmp["verdict"]
    if verdict == "no measurable difference":
        verdict += f" (exact McNemar p = {cmp['p_value']:.3f}, needs p < 0.05)"
    lines.append(f"  verdict      {verdict}")
    if cmp.get("per_case"):
        lines += ["", f"  A case counts as passed above only if it passed every trial"
                      f" ({cmp['baseline_trials']} before, {cmp['candidate_trials']} after).",
                  "  Each case on its own: trials passed before -> after, Fisher exact p"]
        changed = [c for c in cmp["per_case"] if c["p_value"] < 0.05]
        shown = sorted(cmp["per_case"], key=lambda c: (c["p_value"], c["case_id"]))[:20]
        for c in shown:
            lines.append(f"    {c['case_id']:<28} {c['baseline_passes']} of {c['baseline_trials']} ->"
                         f" {c['candidate_passes']} of {c['candidate_trials']}   p {c['p_value']:.3f}")
        if len(cmp["per_case"]) > len(shown):
            lines.append(f"    ... and {len(cmp['per_case']) - len(shown)} more cases with larger p")
        lines.append(f"  {len(changed)} of {len(cmp['per_case'])} cases changed with p < 0.05."
                     " Testing many cases one by one, about 1 in 20 crosses 0.05 by chance.")
    lines += [f"  warning: {w}" for w in cmp["warnings"]]
    return "\n".join(lines)


def _trials_phrase(finished: int, planned: int) -> str:
    """"all 5 trials", or when some did not finish "all 3 finished trials (5 planned)"."""
    phrase = "the 1 finished trial" if finished == 1 else f"all {finished} trials"
    if finished < planned:
        phrase = (phrase if finished == 1 else f"all {finished} finished trials") + f" ({planned} planned)"
    return phrase


def _trial_lines(trials: dict) -> list[str]:
    """The stability block of a report on repeated trials. Every figure says what it counts."""
    n, k = trials["n"], trials["finished"]
    if k == 0:
        return [f"1. Repeated trials      none of the {trials['planned']} planned trials has finished yet,"
                " so there is nothing to report"]
    all_k, any_k, each = trials["passed_all"], trials["passed_any"], trials["per_trial"]
    counts = {c["case_id"]: c for c in trials["cases"]}
    lines = []
    if not trials["complete"]:
        lines.append(f"   only {k} of {trials['planned']} planned trials finished, so these figures are provisional")
    lines.append(f"1. Passed {_trials_phrase(k, trials['planned'])}   {_pct(all_k['rate']['point'])} ({all_k['n_passed']} of {n} cases),"
                 f" 95% interval {_pct(all_k['rate']['low'])} to {_pct(all_k['rate']['high'])}")
    lines.append(f"   Passed at least once  {_pct(any_k['rate']['point'])} ({any_k['n_passed']} of {n} cases)")
    lines.append(f"   Per-trial pass rate   {_pct(each['rate']['point'])} ({each['n_passed']} of {each['n']} case results),"
                 f" 95% interval {_pct(each['rate']['low'])} to {_pct(each['rate']['high'])}"
                 " (each case counted once)")
    unstable = [f"{cid} ({counts[cid]['passes']} of {counts[cid]['trials']})" for cid in trials["unstable_cases"]]
    lines.append(f"   Unstable cases        {len(unstable)}" + (": " + _ids(unstable) if unstable else
                                                              ": every case gave the same result each time"))
    if trials["target_error_trials"]:
        lines.append(f"   Target errors         {trials['target_error_trials']} case results on"
                     f" {len(trials['error_cases'])} cases: {_ids(trials['error_cases'])}"
                     " (each counts as a failed case result; not counted as instability)")
    return lines


def format_report(report: dict) -> str:
    run, rate, n = report["run"], report["pass_rate"], report["n"]
    trials = report.get("trials")
    lines = _header(run, n)
    if trials:
        lines.insert(1, f"  repeated trials: runs {_ids([str(i) for i in trials['runs']], 12)}"
                        f" ({trials['finished']} of {trials['planned']} finished)")
    lines.append("")
    if trials and trials["finished"] == 0:  # nothing to report, even if an unfinished trial recorded cases
        return "\n".join(lines + _trial_lines(trials))
    if n == 0 and not (trials and trials["n"]):
        if trials and trials["finished"] == 0:
            lines += _trial_lines(trials)
        else:
            lines.append("This run recorded no cases (it was probably interrupted), so there are no numbers to report.")
        return "\n".join(lines)

    if trials:
        lines += _trial_lines(trials)
    else:
        lines.append(f"1. Pass rate         {_pct(rate['point'])} ({report['n_passed']} of {n}),"
                     f" 95% interval {_pct(rate['low'])} to {_pct(rate['high'])}")

    if report["grader_checks"]:
        for i, check in enumerate(report["grader_checks"]):
            kappa = "n/a" if check["kappa"] is None else f"{check['kappa']:.2f}"
            prefix = "2. Grader vs human  " if i == 0 else "                     "
            lines.append(f"{prefix}{check['grader']}: agrees on {_pct(check['agreement'])} of {check['n']}"
                         f" labels, kappa {kappa}, catches {_pct(check['catch_rate'])} of bad outputs"
                         f" - {check['trust']}")
    else:
        lines.append(f"2. Grader vs human   not measured: no human labels (llmtuner label {run['id']})")
    if report["unchecked_judges"]:
        lines.append("                     unchecked judge graders: " + ", ".join(report["unchecked_judges"]))

    floor = report["detection_floor"]
    if floor is None:
        lines.append(f"3. Detection floor   none: {n} cases are too few to show any change (at least 6 needed)")
    else:
        lines.append(f"3. Detection floor   about {floor * 100:.0f} points: at least 6 of {n} cases must change"
                     " one way before a difference can be shown")

    baseline = report["baseline"]
    if baseline:
        worse = baseline["regressions"]
        lines.append(f"4. Got worse         {len(worse)} of {baseline['n']} cases vs baseline run"
                     f" {baseline['baseline']['id']}" + (f": {_ids(worse)}" if worse else "")
                     + f" ({baseline['verdict']})")
        lines += [f"   warning: {w}" for w in baseline["warnings"]]
    elif report["is_baseline"]:
        lines.append("4. Got worse         this run is the baseline")
    else:
        lines.append(f"4. Got worse         not measured: no baseline (llmtuner baseline {run['id']})")

    if trials:
        lines += ["", f"Usage ({_trials_phrase(trials['finished'], trials['planned'])} together;"
                      " counts are case results, one case in one trial)",
                  "  " + _usage_line(trials["usage"], trials["per_trial"]["n"], group=True)]
        lines += ["", f"This trial alone (run {run['id']}): {report['n_passed']} of {n} cases passed"]
    else:
        lines += ["", "Usage", "  " + _usage_line(report["usage"], n)]

    lines += ["", "Graders" + (f" (run {run['id']})" if trials else "")]
    for g in report["graders"]:
        flags = "" if g["required"] else " (informational)"
        errors = f", {g['errors']} grader errors" if g["errors"] else ""
        lines.append(f"  {g['name']:<20} {g['type']:<13} {g['n_passed']} of {g['n']} pass{errors}{flags}")
    if report["target_errors"]:
        lines.append(f"  target failed on {report['target_errors']} cases ({report['timeouts']} timeouts)")
    if report["failed_cases"]:
        lines.append(f"\nFailed cases: {_ids(report['failed_cases'], 12)}"
                     f"  (read them: llmtuner show {run['id']} --failed)")
    return "\n".join(lines)


def format_gate(result: dict) -> str:
    lines = [("PASS: " if result["ok"] else "FAIL: ") + result["reason"]]
    if result.get("note"):
        lines.append("note: " + result["note"])
    if result["comparison"]:
        lines += [f"warning: {w}" for w in result["comparison"]["warnings"]]
    if result["comparison"] and result["comparison"]["regressions"]:
        lines.append("got worse: " + _ids(result["comparison"]["regressions"], 12))
    return "\n".join(lines)
