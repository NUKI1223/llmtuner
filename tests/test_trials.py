"""Tests for repeated trials (`run --trials K`): storage, running, group semantics, reports, CLI, MCP.

Offline and fast: targets are stub scripts run with sys.executable. Every store is a temp dir.
The stub reads the case id on stdin and LLMTUNER_TRIAL from the environment, and answers by a rule
file in the temp project: per case a fixed outcome or a list with one outcome per trial.
Outcomes: "yes", "no", "crash" (exit 3), "sleep" (runs into the timeout).
"""

import asyncio
import contextlib
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from llmtuner import mcp_server, report, stats
from llmtuner.cli import main
from llmtuner.config import ConfigError, load_config
from llmtuner.dataset import load_cases
from llmtuner.runner import MAX_TRIALS, run_eval
from llmtuner.store import _SCHEMA, Store

try:
    from mcp import Client, StdioServerParameters
except ImportError:  # the core package has no dependencies; mcp is optional
    Client = StdioServerParameters = None

PY = sys.executable
SRC = str(Path(__file__).resolve().parent.parent / "src")

STUB = """\
import json, os, sys, time
case = sys.stdin.read().strip()
trial = int(os.environ["LLMTUNER_TRIAL"])
rules = json.load(open(sys.argv[1]))
with open("seen.log", "a") as log:
    log.write(f"{sys.argv[1]} {case} {trial}\\n")
plan = rules.get("cases", {}).get(case, rules.get("default", "yes"))
outcome = (plan[min(trial, len(plan)) - 1]) if isinstance(plan, list) else plan
if rules.get("rewrite") is not None and trial == 1:
    open("cases.jsonl", "w").write(rules["rewrite"])
if rules.get("cost") is not None:
    open(os.environ["LLMTUNER_META"], "w").write(json.dumps({"cost": rules["cost"], "cost_unit": "credits"}))
if outcome == "crash":
    sys.exit(3)
if outcome == "sleep":
    time.sleep(60)
print(outcome)
"""

GRADERS = [{"name": "same", "type": "exact", "required": True}]


def strict_loads(text):
    def refuse(token):
        raise ValueError(f"not valid JSON: {token}")

    return json.loads(text, parse_constant=refuse)


def fail(*trials):
    """A per-trial plan: 'no' on the listed trial numbers (1-based), 'yes' elsewhere, for 5 trials."""
    return ["no" if t in trials else "yes" for t in range(1, 6)]


class TrialBase(unittest.TestCase):
    N = 6

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.root = self.tmp / "project"
        self.root.mkdir()
        self.store = Store(home=self.tmp / "store")
        self.addCleanup(self.store.close)
        (self.root / "stub.py").write_text(STUB)
        self.targets = {}
        self.write_cases(self.N)

    def write_cases(self, n):
        rows = "\n".join(json.dumps({"id": f"c{i}", "input": f"c{i}", "expected": "yes"})
                         for i in range(1, n + 1)) + "\n"
        (self.root / "cases.jsonl").write_text(rows)

    def target(self, name, cases=None, default="yes", cost=None, version=False, rewrite=None, timeout=15):
        """Define (or redefine) a stub target whose answers follow `cases` (case id -> outcome or list)."""
        rules = {"cases": cases or {}, "default": default, "cost": cost, "rewrite": rewrite}
        (self.root / f"rules_{name}.json").write_text(json.dumps(rules))
        self.targets[name] = (timeout, version)
        text = 'project = "evaldemo"\n'
        for tname, (tmo, ver) in self.targets.items():
            text += (f"[targets.{tname}]\ncommand = {json.dumps([PY, 'stub.py', f'rules_{tname}.json'])}\n"
                     f"timeout = {tmo}\n")
            if ver:
                text += f"version_command = {json.dumps([PY, '-c', 'print(1.5)'])}\n"
        text += '[graders.same]\ntype = "exact"\n[datasets.main]\npath = "cases.jsonl"\n'
        (self.root / "llmtuner.toml").write_text(text)

    def run_target(self, name, trials=1, jobs=4, **kw):
        return run_eval(load_config(self.root), self.store, name, "main", trials=trials, jobs=jobs, **kw)

    def group_ids(self, first):
        return [r["id"] for r in self.store.group_runs(first)]

    def seen(self):
        path = self.root / "seen.log"
        return [line.split() for line in path.read_text().splitlines()] if path.exists() else []

    def fake_group(self, target, planned, finished, passes, case_ids=None):
        """A group written straight to the store. `passes[i]` is the set of case ids passing in trial i+1;
        only the first `finished` runs are finished. Returns the run ids."""
        case_ids = case_ids or [f"c{i}" for i in range(1, self.N + 1)]
        digest = load_cases(self.root / "cases.jsonl")[1]
        ids = []
        for trial in range(1, planned + 1):
            run_id = self.store.create_run(
                project="evaldemo", dataset="main", dataset_hash=digest, target=target,
                command=[PY], model=None, tool_version=None, split="test", graders=GRADERS, note=None,
                trial_group=ids[0] if ids else None, trial=trial, trials_planned=planned)
            ids.append(run_id)
            ok = passes[trial - 1] if trial - 1 < len(passes) else set()
            for cid in case_ids:
                passed = cid in ok
                self.store.add_case(
                    run_id, SimpleNamespace(id=cid, input_text=cid, expected="yes", context=None),
                    SimpleNamespace(output="yes" if passed else "no", exit_code=0, timed_out=False,
                                    duration_s=0.1, error=None),
                    passed, [SimpleNamespace(grader="same", passed=passed, error=False, detail=None)])
            if trial <= finished:
                self.store.finish_run(run_id)
        return ids


# --- statistics ---------------------------------------------------------------


class FisherTests(unittest.TestCase):
    def test_known_answers_for_five_trials_a_side(self):
        self.assertAlmostEqual(stats.fisher_exact(0, 5, 5, 5), 2 / 252, places=4)  # 0.0079
        self.assertAlmostEqual(stats.fisher_exact(0, 5, 5, 5), 0.0079, places=4)
        self.assertAlmostEqual(stats.fisher_exact(1, 5, 5, 5), 0.0476, places=4)
        self.assertAlmostEqual(stats.fisher_exact(2, 5, 5, 5), 0.1667, places=4)

    def test_identical_counts_give_one(self):
        for passes, n in ((0, 5), (3, 5), (5, 5), (1, 1), (7, 10)):
            self.assertEqual(stats.fisher_exact(passes, n, passes, n), 1.0, (passes, n))

    def test_symmetric_and_never_above_one(self):
        for n_a in range(1, 7):
            for n_b in range(1, 7):
                for a in range(n_a + 1):
                    for b in range(n_b + 1):
                        p = stats.fisher_exact(a, n_a, b, n_b)
                        self.assertGreater(p, 0.0)
                        self.assertLessEqual(p, 1.0)
                        self.assertAlmostEqual(p, stats.fisher_exact(b, n_b, a, n_a), places=12)

    def test_a_side_with_no_trials_gives_one(self):
        self.assertEqual(stats.fisher_exact(0, 0, 3, 5), 1.0)
        self.assertEqual(stats.fisher_exact(3, 5, 0, 0), 1.0)
        self.assertEqual(stats.fisher_exact(0, 0, 0, 0), 1.0)


# --- storage and running ------------------------------------------------------


class KnownAnswerTests(TrialBase):
    def test_five_trials_over_three_cases(self):
        self.write_cases(3)
        self.target("t", cases={"c1": "yes", "c2": fail(2, 4), "c3": "no"})
        first = self.run_target("t", trials=5)
        rep = report.build_report(self.store, first)
        tr = rep["trials"]
        self.assertEqual((tr["planned"], tr["finished"], tr["complete"], tr["n"]), (5, 5, True, 3))
        self.assertEqual(tr["passed_all"]["n_passed"], 1)
        self.assertEqual(tr["passed_any"]["n_passed"], 2)
        self.assertEqual((tr["per_trial"]["n_passed"], tr["per_trial"]["n"]), (8, 15))
        self.assertEqual(tr["unstable_cases"], ["c2"])
        self.assertEqual(tr["error_cases"], [])
        self.assertEqual(tr["runs"], list(range(first, first + 5)))
        self.assertEqual(tr["group"], first)
        self.assertEqual({c["case_id"]: (c["passes"], c["trials"]) for c in tr["cases"]},
                         {"c1": (5, 5), "c2": (3, 5), "c3": (0, 5)})
        # Trial 3 is an ordinary run with its own numbers: c1 and c2 passed, c3 did not.
        mid = report.build_report(self.store, first + 2)
        self.assertEqual((mid["n"], mid["n_passed"]), (3, 2))
        self.assertEqual(mid["trials"]["group"], first)
        self.assertEqual(mid["trials"]["passed_all"]["n_passed"], 1)


class StorageAndRunningTests(TrialBase):
    def test_single_run_has_no_trial_fields(self):
        self.target("t")
        run_id = self.run_target("t")
        run = self.store.get_run(run_id)
        self.assertEqual((run["trial_group"], run["trial"], run["trials_planned"]), (None, None, None))
        self.assertIsNone(report.build_report(self.store, run_id)["trials"])
        self.assertEqual(self.group_ids(run_id), [run_id])
        self.assertEqual({t for _, _, t in self.seen()}, {"1"})

    def test_k_trials_make_k_linked_runs(self):
        self.target("t", version=True, cases={f"c{i}": ["no", "yes", "no"] for i in range(1, 7)})
        first = self.run_target("t", trials=3, note="hello")
        ids = self.group_ids(first)
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[0], first)
        runs = [self.store.get_run(i) for i in ids]
        self.assertEqual([r["trial"] for r in runs], [1, 2, 3])
        self.assertEqual({r["trial_group"] for r in runs}, {first})
        self.assertEqual({r["trials_planned"] for r in runs}, {3})
        self.assertEqual(len({r["dataset_hash"] for r in runs}), 1)
        self.assertEqual({r["tool_version"] for r in runs}, {"1.5"})
        self.assertEqual({r["note"] for r in runs}, {"hello"})
        self.assertTrue(all(r["finished_at"] is not None for r in runs))
        # Any trial finds the same group.
        self.assertEqual(self.group_ids(ids[2]), ids)

    def test_each_trial_gets_its_own_number(self):
        # c_i passes only in trial i, so each stored run proves which number its target saw.
        self.target("t", default="no", cases={"c1": ["yes", "no", "no"], "c2": ["no", "yes", "no"],
                                              "c3": ["no", "no", "yes"]})
        self.write_cases(3)
        first = self.run_target("t", trials=3)
        for number, run_id in enumerate(self.group_ids(first), start=1):
            passed = [c["case_id"] for c in self.store.get_cases(run_id) if c["passed"]]
            self.assertEqual(passed, [f"c{number}"])
        seen = self.seen()
        self.assertEqual(len(seen), 9)
        self.assertEqual({(case, trial) for _, case, trial in seen},
                         {(f"c{c}", str(t)) for c in (1, 2, 3) for t in (1, 2, 3)})

    def test_parallel_jobs_give_the_same_results(self):
        plan = {f"c{i}": ["yes" if (i + t) % 3 else "no" for t in range(1, 4)] for i in range(1, 7)}
        self.target("t", cases=plan)

        def results(jobs):
            first = self.run_target("t", trials=3, jobs=jobs)
            return [[(c["case_id"], bool(c["passed"]), c["output"].strip()) for c in self.store.get_cases(i)]
                    for i in self.group_ids(first)]

        serial, parallel = results(1), results(4)
        self.assertEqual(serial, parallel)
        self.assertTrue(any(not ok for trial in serial for _, ok, _ in trial))

    def test_invalid_trial_counts_are_rejected_before_anything_runs(self):
        self.target("t")
        for bad in (0, -1, MAX_TRIALS + 1, 51, True, 2.0, "3", None):
            with self.subTest(trials=bad):
                with self.assertRaises(ConfigError):
                    self.run_target("t", trials=bad)
        self.assertEqual(self.store.list_runs(), [])
        self.assertEqual(self.seen(), [])

    def test_the_cli_rejects_zero_trials(self):
        self.target("t")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as raised:
            main(["run", "t", "main", "--trials", "0", "--store", str(self.tmp / "s2"), "--config", str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--trials", err.getvalue())
        self.assertEqual(self.seen(), [])

    def test_the_dataset_is_read_once(self):
        original = (self.root / "cases.jsonl").read_text()
        digest = load_cases(self.root / "cases.jsonl")[1]
        replacement = json.dumps({"id": "zz", "input": "zz", "expected": "yes"}) + "\n"
        self.target("t", rewrite=replacement)
        first = self.run_target("t", trials=2)
        self.assertEqual((self.root / "cases.jsonl").read_text(), replacement)  # the stub did rewrite it
        one, two = [self.store.get_run(i) for i in self.group_ids(first)]
        self.assertEqual(one["dataset_hash"], two["dataset_hash"])
        self.assertEqual(one["dataset_hash"], digest)
        ids = [[c["case_id"] for c in self.store.get_cases(i)] for i in (one["id"], two["id"])]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], [f"c{i}" for i in range(1, self.N + 1)])
        self.assertNotEqual(original, replacement)


# --- migration -----------------------------------------------------------------

OLD_RUNS = _SCHEMA.replace("  finished_at REAL,\n  trial_group INTEGER,\n  trial INTEGER,\n  trials_planned INTEGER\n",
                           "  finished_at REAL\n")

OPEN_SCRIPT = """\
import os, sys, time
from llmtuner.store import Store
while not os.path.exists(sys.argv[2]):
    time.sleep(0.001)
Store(sys.argv[1]).close()
"""


def make_old_store(home: Path) -> None:
    """A store as the version before repeated trials wrote it: one run with one case."""
    assert OLD_RUNS != _SCHEMA and "trial" not in OLD_RUNS
    home.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(home / "store.db")
    db.executescript(OLD_RUNS)
    db.execute("INSERT INTO runs (project, dataset, dataset_hash, target, command, model, tool_version, split,"
               " graders, note, started_at, finished_at) VALUES ('old','main','h','t','[\"x\"]',NULL,NULL,'test',"
               "'[]','n',1.0,2.0)")
    db.execute("INSERT INTO cases (run_id, case_id, input, expected, context, output, exit_code, timed_out,"
               " duration_s, error, passed, meta) VALUES (1,'a','in',NULL,NULL,'out',0,0,0.1,NULL,1,NULL)")
    db.commit()
    db.close()


def columns(home: Path, table="runs") -> list[str]:
    db = sqlite3.connect(home / "store.db")
    try:
        return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
    finally:
        db.close()


class MigrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_an_old_store_is_upgraded_on_open(self):
        home = self.tmp / "old"
        make_old_store(home)
        self.assertNotIn("trial_group", columns(home))
        with Store(home) as store:
            run = store.get_run(1)
            self.assertEqual((run["trial_group"], run["trial"], run["trials_planned"]), (None, None, None))
            self.assertEqual(store.group_runs(1), [run])
            row = store.list_runs()[0]
            self.assertEqual((row["trial_group"], row["trial"], row["trials_planned"]), (None, None, None))
            self.assertEqual(len(store.get_cases(1)), 1)
            rep = report.build_report(store, 1)
            self.assertIsNone(rep["trials"])
            # A grouped run can be written on the upgraded store.
            first = store.create_run(project="old", dataset="main", dataset_hash="h", target="t", command=["x"],
                                     model=None, tool_version=None, split="test", graders=[], note=None,
                                     trials_planned=2, trial=1)
            second = store.create_run(project="old", dataset="main", dataset_hash="h", target="t", command=["x"],
                                      model=None, tool_version=None, split="test", graders=[], note=None,
                                      trial_group=first, trial=2, trials_planned=2)
            self.assertEqual(store.get_run(first)["trial_group"], first)
            self.assertEqual([r["id"] for r in store.group_runs(second)], [first, second])

    def test_opening_twice_changes_nothing(self):
        home = self.tmp / "old"
        make_old_store(home)
        Store(home).close()
        after_first = columns(home)
        for name in ("trial_group", "trial", "trials_planned"):
            self.assertEqual(after_first.count(name), 1)
        Store(home).close()
        self.assertEqual(columns(home), after_first)
        self.assertEqual(columns(home, "cases").count("meta"), 1)
        with Store(home) as store:
            self.assertEqual(len(store.list_runs()), 1)

    def test_eight_processes_open_an_old_store_together(self):
        env = dict(os.environ, PYTHONPATH=SRC)
        for round_no in range(2):
            home = self.tmp / f"race{round_no}"
            make_old_store(home)
            go = self.tmp / f"go{round_no}"
            procs = [subprocess.Popen([PY, "-c", OPEN_SCRIPT, str(home), str(go)], env=env,
                                      stderr=subprocess.PIPE, text=True) for _ in range(8)]
            time.sleep(0.5)  # let every process reach the starting line
            go.write_text("go")
            for proc in procs:
                _, err = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, 0, err)
            names = columns(home)
            for name in ("trial_group", "trial", "trials_planned"):
                self.assertEqual(names.count(name), 1, name)


# --- group semantics -----------------------------------------------------------


class CopyingAddsNoCertaintyTests(TrialBase):
    N = 8

    def test_same_answers_every_trial_do_not_narrow_the_interval(self):
        self.target("a", cases={"c1": "no", "c2": "no"})
        single = self.run_target("a")
        group = self.run_target("a", trials=5)
        one, many = report.build_report(self.store, single), report.build_report(self.store, group)["trials"]
        self.assertEqual(many["passed_all"]["n_passed"], one["n_passed"])
        self.assertEqual(many["passed_any"]["n_passed"], one["n_passed"])
        self.assertEqual(many["per_trial"]["n_passed"], 5 * one["n_passed"])
        self.assertEqual(many["unstable_cases"], [])
        for key in ("passed_all", "per_trial"):
            rate = many[key]["rate"]
            self.assertLessEqual(rate["low"], one["pass_rate"]["low"] + 1e-9, key)
            self.assertGreaterEqual(rate["high"], one["pass_rate"]["high"] - 1e-9, key)
            self.assertAlmostEqual(rate["point"], one["pass_rate"]["point"], places=9)

    def test_compare_between_groups_gives_the_verdict_of_the_single_runs(self):
        self.target("a")
        self.target("b", cases={f"c{i}": "no" for i in range(1, 7)})
        singles = report.compare_runs(self.store, self.run_target("a"), self.run_target("b"))
        groups = report.compare_runs(self.store, self.run_target("a", trials=5), self.run_target("b", trials=5))
        self.assertEqual(singles["verdict"], "candidate is worse")
        for key in ("verdict", "regressions", "improvements", "n", "p_value", "warnings"):
            self.assertEqual(groups[key], singles[key], key)
        self.assertEqual(groups["baseline_trials"], 5)
        self.assertEqual(groups["candidate_trials"], 5)


class GroupSemanticsTests(TrialBase):
    def test_four_of_five_counts_as_failed_for_the_group(self):
        self.target("a")
        self.target("b", cases={"c1": fail(3), "c2": fail(1, 2)})
        cmp = report.compare_runs(self.store, self.run_target("a", trials=5), self.run_target("b", trials=5))
        self.assertEqual(cmp["regressions"], ["c1", "c2"])
        self.assertEqual(cmp["improvements"], [])
        self.assertAlmostEqual(cmp["baseline_rate"], 1.0)
        self.assertAlmostEqual(cmp["candidate_rate"], (self.N - 2) / self.N)

    def test_trial_count_warning(self):
        self.target("a")
        group_a = self.run_target("a", trials=5)
        group_b = self.run_target("a", trials=5)
        single = self.run_target("a")
        mixed = report.compare_runs(self.store, group_a, single)
        self.assertTrue(any("trial(s) per case" in w for w in mixed["warnings"]), mixed["warnings"])
        self.assertEqual((mixed["baseline_trials"], mixed["candidate_trials"]), (5, 1))
        both = report.compare_runs(self.store, group_a, group_b)
        self.assertFalse(any("trial" in w for w in both["warnings"]), both["warnings"])

    def test_per_case_is_none_for_single_runs_and_a_list_otherwise(self):
        self.write_cases(3)
        self.target("before", cases={"c1": "no", "c2": fail(1, 2)})
        self.target("after")
        singles = report.compare_runs(self.store, self.run_target("before"), self.run_target("after"))
        self.assertIsNone(singles["per_case"])
        self.assertEqual((singles["baseline_trials"], singles["candidate_trials"]), (1, 1))

        cmp = report.compare_runs(self.store, self.run_target("before", trials=5), self.run_target("after", trials=5))
        per = {c["case_id"]: c for c in cmp["per_case"]}
        self.assertEqual(list(per), ["c1", "c2", "c3"])
        self.assertEqual((per["c1"]["baseline_passes"], per["c1"]["baseline_trials"],
                          per["c1"]["candidate_passes"], per["c1"]["candidate_trials"]), (0, 5, 5, 5))
        self.assertEqual((per["c2"]["baseline_passes"], per["c2"]["candidate_passes"]), (3, 5))
        self.assertEqual((per["c3"]["baseline_passes"], per["c3"]["candidate_passes"]), (5, 5))
        self.assertAlmostEqual(per["c1"]["p_value"], 0.0079, places=4)
        self.assertLess(per["c1"]["p_value"], 0.05)
        self.assertGreaterEqual(per["c2"]["p_value"], 0.05)
        self.assertAlmostEqual(per["c2"]["p_value"], stats.fisher_exact(3, 5, 5, 5), places=12)
        self.assertEqual(per["c3"]["p_value"], 1.0)

        # A group against a single run still tests each case, with one trial on one side.
        mixed = report.compare_runs(self.store, self.run_target("before", trials=5), self.run_target("after"))
        self.assertEqual({c["candidate_trials"] for c in mixed["per_case"]}, {1})

    def test_an_unfinished_trial_is_ignored(self):
        everyone = {"c1", "c2", "c3", "c4", "c5", "c6"}
        ids = self.fake_group("slow", planned=3, finished=2, passes=[everyone, everyone, set()])
        rep = report.build_report(self.store, ids[0])
        tr = rep["trials"]
        self.assertFalse(tr["complete"])
        self.assertEqual((tr["finished"], tr["planned"]), (2, 3))
        self.assertLess(tr["finished"], tr["planned"])
        self.assertEqual(tr["runs"], ids)
        self.assertEqual(tr["passed_all"]["n_passed"], 6)  # the unfinished third trial (all failed) is ignored
        self.assertEqual((tr["per_trial"]["n_passed"], tr["per_trial"]["n"]), (12, 12))
        self.assertEqual(tr["unstable_cases"], [])
        text = report.format_report(rep)
        self.assertIn("provisional", text)
        self.assertIn("only 2 of 3 planned trials finished", text)
        # compare counts the same two trials only
        self.target("good")
        cmp = report.compare_runs(self.store, ids[0], self.run_target("good"))
        self.assertEqual(cmp["baseline_trials"], 2)
        self.assertEqual(cmp["regressions"], [])
        self.assertEqual(cmp["baseline_rate"], 1.0)
        self.assertTrue(any("trial(s) per case" in w for w in cmp["warnings"]))

    def test_a_complete_group_is_not_provisional(self):
        self.target("t")
        text = report.format_report(report.build_report(self.store, self.run_target("t", trials=3)))
        self.assertNotIn("provisional", text)

    def test_target_errors_are_not_instability(self):
        self.write_cases(5)
        self.target("t", timeout=1.5, cases={
            "c1": ["yes", "sleep", "yes"],  # a timeout between two passes
            "c2": ["yes", "crash", "yes"],  # a crash between two passes
            "c3": ["yes", "no", "crash"],   # pass, fail and error: really unstable
            "c4": ["no", "crash", "no"],    # failing, with an error
            "c5": "yes",
        })
        first = self.run_target("t", trials=3, jobs=4)
        tr = report.build_report(self.store, first)["trials"]
        self.assertEqual(tr["target_error_trials"], 4)
        self.assertEqual(tr["error_cases"], ["c1", "c2", "c3", "c4"])
        self.assertEqual(tr["unstable_cases"], ["c3"])
        self.assertEqual(tr["passed_all"]["n_passed"], 1)  # an errored trial still stops a case passing every one
        self.assertEqual(tr["passed_any"]["n_passed"], 4)  # c4 never passed
        text = report.format_report(report.build_report(self.store, first))
        self.assertIn("Target errors", text)
        self.assertIn("not counted as instability", text)
        self.assertIn("Unstable cases        1: c3 (1 of 3)", text)

    def test_usage_adds_up_over_all_trials(self):
        self.write_cases(4)
        self.target("t", cost=0.01)
        first = self.run_target("t", trials=3)
        use = report.build_report(self.store, first)["trials"]["usage"]
        self.assertAlmostEqual(use["cost"], 0.12)
        self.assertEqual(use["cases_with_cost"], 12)
        self.assertEqual(use["cost_unit"], "credits")
        self.assertAlmostEqual(use["cost_per_pass"], 0.01)  # 0.12 over 12 passing case results
        # the trial's own usage is just its own cases
        self.assertEqual(report.build_report(self.store, first)["usage"]["cases_with_cost"], 4)


class BaselineAndGateTests(TrialBase):
    def test_baseline_on_any_trial_stores_the_first_run(self):
        self.target("a")
        self.target("b", cases={"c1": "no"})
        ids = self.group_ids(self.run_target("a", trials=3))
        returned = self.store.set_baseline(ids[1])
        self.assertEqual(returned["id"], ids[1])
        self.assertEqual(self.store.get_baseline("evaldemo", "main"), ids[0])
        for run_id in ids:
            rep = report.build_report(self.store, run_id)
            self.assertTrue(rep["is_baseline"], run_id)
            self.assertIsNone(rep["baseline"], run_id)
        later = report.build_report(self.store, self.run_target("b", trials=3))
        self.assertFalse(later["is_baseline"])
        self.assertEqual(later["baseline"]["baseline"]["id"], ids[0])
        self.assertEqual(later["baseline"]["regressions"], ["c1"])
        # a single run is compared against the group too, with a warning
        single = report.build_report(self.store, self.run_target("a"))
        self.assertEqual(single["baseline"]["baseline"]["id"], ids[0])
        self.assertTrue(any("trial(s) per case" in w for w in single["baseline"]["warnings"]))

    def test_baseline_of_a_single_run_is_unchanged(self):
        self.target("a")
        run_id = self.run_target("a")
        self.store.set_baseline(run_id)
        self.assertEqual(self.store.get_baseline("evaldemo", "main"), run_id)

    def test_gate_on_groups_names_the_stricter_measure(self):
        self.target("good")
        self.target("flaky", cases={"c1": fail(2), "c2": fail(5), "c3": "no"})
        self.store.set_baseline(self.run_target("good", trials=5))
        result = report.gate(self.store, self.run_target("flaky", trials=5))
        self.assertFalse(result["ok"])
        self.assertIn("passing every trial", result["reason"])
        self.assertIn("fell", result["reason"])
        self.assertEqual(result["comparison"]["regressions"], ["c1", "c2", "c3"])
        again = report.gate(self.store, self.run_target("good", trials=5))
        self.assertTrue(again["ok"], again["reason"])
        self.assertIn("passing every trial", again["reason"])

    def test_gate_on_single_runs_still_says_pass_rate(self):
        self.target("good")
        self.target("bad", cases={"c1": "no", "c2": "no"})
        self.store.set_baseline(self.run_target("good"))
        result = report.gate(self.store, self.run_target("bad"))
        self.assertFalse(result["ok"])
        self.assertIn("pass rate", result["reason"])
        self.assertNotIn("every trial", result["reason"])


class RankTests(TrialBase):
    def test_groups_are_one_row_each_and_mismatches_are_skipped(self):
        self.target("single", cases={"c1": "no"})
        self.target("two", cases={"c1": "no"})
        self.target("steady")
        self.target("flaky", cases={"c1": fail(2), "c2": "no"})
        single = self.run_target("single")
        two = self.run_target("two", trials=2)
        # three trials, finished only two of them
        everyone = {f"c{i}" for i in range(1, 7)}
        slow = self.fake_group("slow", planned=3, finished=2, passes=[everyone, everyone, everyone])
        steady = self.run_target("steady", trials=3)
        flaky = self.run_target("flaky", trials=3)

        result = report.rank(self.store, "evaldemo", "main")
        self.assertEqual([r["target"] for r in result["rows"]], ["steady", "flaky"])
        steady_row, flaky_row = result["rows"]
        self.assertEqual((steady_row["run"], steady_row["trials"], steady_row["n"], steady_row["n_passed"]),
                         (steady, 3, 6, 6))
        # flaky: c1 fails one trial, c2 fails all: 4 of 6 cases passed every trial
        self.assertEqual((flaky_row["run"], flaky_row["trials"], flaky_row["n"], flaky_row["n_passed"]),
                         (flaky, 3, 6, 4))
        self.assertEqual(flaky_row["pass_rate"]["point"], 4 / 6)
        self.assertEqual(len([r for r in result["rows"] if r["target"] == "steady"]), 1)  # not one row per trial
        skipped = {s["target"]: s for s in result["skipped"]}
        self.assertEqual(set(skipped), {"single", "two", "slow"})
        self.assertEqual(skipped["single"]["reason"], "ran a different number of trials per case")
        self.assertEqual(skipped["single"]["run"], single)
        self.assertEqual(skipped["two"]["reason"], "ran a different number of trials per case")
        self.assertEqual(skipped["two"]["run"], two)
        self.assertEqual(skipped["slow"]["reason"], "its repeated trials did not all finish")
        self.assertNotIn("slow", [r["target"] for r in result["rows"]])
        self.assertNotIn(slow[0], [r["run"] for r in result["rows"]])
        text = report.format_rank(result)
        self.assertIn("passed all 3 trials", text)
        self.assertIn("per passing case result", text)
        self.assertNotIn("per passing trial", text)
        self.assertIn("ran a different number of trials per case", text)
        self.assertIn("its repeated trials did not all finish", text)

    def test_a_lone_unfinished_group_is_never_ranked(self):
        everyone = {f"c{i}" for i in range(1, 7)}
        self.fake_group("slow", planned=3, finished=1, passes=[everyone])
        result = report.rank(self.store, "evaldemo", "main")
        self.assertEqual(result["rows"], [])
        self.assertEqual([s["reason"] for s in result["skipped"]], ["its repeated trials did not all finish"])

    def test_single_runs_still_rank_as_before(self):
        self.target("good")
        self.target("worse", cases={"c1": "no", "c2": "no"})
        good, worse = self.run_target("good"), self.run_target("worse")
        result = report.rank(self.store, "evaldemo", "main")
        self.assertEqual([(r["target"], r["run"], r["trials"], r["n_passed"]) for r in result["rows"]],
                         [("good", good, 1, 6), ("worse", worse, 1, 4)])
        self.assertEqual(result["skipped"], [])
        self.assertNotIn("trials", report.format_rank(result))


    def test_a_group_with_no_finished_trial_is_listed_as_skipped(self):
        everyone = {f"c{i}" for i in range(1, 7)}
        self.fake_group("steady", planned=3, finished=3, passes=[everyone] * 3)
        never = self.fake_group("never", planned=3, finished=0, passes=[])
        same = self.fake_group("steady", planned=3, finished=0, passes=[])  # same target as a ranked row
        result = report.rank(self.store, "evaldemo", "main")
        self.assertEqual([r["target"] for r in result["rows"]], ["steady"])
        self.assertEqual([(s["run"], s["target"], s["reason"]) for s in result["skipped"]],
                         [(never[0], "never", "its repeated trials did not all finish")])
        self.assertNotIn(same[0], [s["run"] for s in result["skipped"]])
        self.assertIn(f"skipped run {never[0]} (never): its repeated trials did not all finish",
                      report.format_rank(result))

    def test_a_lone_group_with_no_finished_trial_is_still_listed(self):
        never = self.fake_group("never", planned=2, finished=0, passes=[])
        result = report.rank(self.store, "evaldemo", "main")
        self.assertEqual(result["rows"], [])
        self.assertEqual([s["run"] for s in result["skipped"]], [never[0]])
        text = report.format_rank(result)
        self.assertIn("no finished runs", text)
        self.assertIn("its repeated trials did not all finish", text)


# --- text output ----------------------------------------------------------------


class TextOutputTests(TrialBase):
    def test_group_report_labels_every_figure(self):
        self.target("t", cases={"c1": fail(2), "c2": "no"})
        text = report.format_report(report.build_report(self.store, self.run_target("t", trials=5)))
        for label in ("Passed all 5 trials", "Passed at least once", "Per-trial pass rate", "Unstable cases"):
            self.assertIn(label, text)
        self.assertNotIn("1. Pass rate", text)
        self.assertIn("c1 (4 of 5)", text)
        self.assertIn("repeated trials: runs", text)
        self.assertIn("This trial alone", text)

    def test_single_report_is_unchanged(self):
        self.target("t")
        text = report.format_report(report.build_report(self.store, self.run_target("t")))
        self.assertIn("1. Pass rate", text)
        for label in ("Passed all", "Passed at least once", "Per-trial", "Unstable", "provisional",
                      "repeated trials", "This trial alone"):
            self.assertNotIn(label, text)

    def test_group_compare_prints_the_per_case_block(self):
        self.write_cases(3)
        self.target("before", cases={"c1": "no"})
        self.target("after")
        text = report.format_compare(report.compare_runs(
            self.store, self.run_target("before", trials=5), self.run_target("after", trials=5)))
        self.assertIn("Each case on its own", text)
        self.assertIn("c1", text)
        self.assertIn("0 of 5 -> 5 of 5", text)
        self.assertIn("p 0.008", text)
        self.assertIn("Testing many cases one by one", text)
        self.assertIn("of cases passed every trial", text)

    def test_single_compare_prints_neither(self):
        self.write_cases(3)
        self.target("before", cases={"c1": "no"})
        self.target("after")
        text = report.format_compare(report.compare_runs(
            self.store, self.run_target("before"), self.run_target("after")))
        self.assertNotIn("Each case on its own", text)
        self.assertNotIn("one by one", text)
        self.assertNotIn("Fisher", text)
        self.assertIn("pass rate", text)


# --- CLI -------------------------------------------------------------------------


class CliTests(TrialBase):
    def cli(self, *argv, expect=0):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([*argv, "--store", str(self.tmp / "clistore"), "--config", str(self.root)])
        self.assertEqual(code, expect, err.getvalue())
        return out.getvalue()

    def test_run_json_is_strict_and_has_the_trials_section(self):
        self.target("t", cases={"c1": fail(2), "c2": "no"})
        out = self.cli("run", "t", "main", "--trials", "3", "--json")
        data = strict_loads(out)  # NaN or Infinity would raise
        tr = data["trials"]
        self.assertEqual((tr["planned"], tr["finished"], tr["complete"], tr["n"]), (3, 3, True, 6))
        self.assertEqual(tr["passed_all"]["n_passed"], 4)
        self.assertEqual(tr["unstable_cases"], ["c1"])
        self.assertEqual(len(tr["runs"]), 3)
        self.assertEqual(data["run"]["id"], tr["group"])

    def test_runs_marks_each_trial_and_report_shows_the_group(self):
        self.target("t", cases={"c1": fail(2)})
        self.cli("run", "t", "main", "--trials", "3", "--json")
        listing = self.cli("runs")
        self.assertEqual(sum("of group 1" in line for line in listing.splitlines()), 3)
        for number in (1, 2, 3):
            self.assertIn(f"trial {number}/3 of group 1", listing)
        text = self.cli("report", "2")
        self.assertIn("repeated trials: runs 1, 2, 3", text)
        self.assertIn("Passed all 3 trials", text)
        self.assertIn("This trial alone (run 2)", text)
        data = strict_loads(self.cli("report", "3", "--json"))
        self.assertEqual(data["trials"]["group"], 1)
        # a single run listed through the CLI carries no trial mark
        self.cli("run", "t", "main", "--json")
        last = self.cli("runs").splitlines()[0]
        self.assertNotIn("of group", last)

    def test_baseline_gate_and_compare_through_the_cli(self):
        self.target("good")
        self.target("flaky", cases={"c1": fail(3)})
        self.cli("run", "good", "main", "--trials", "3", "--set-baseline", "--json")
        self.cli("run", "flaky", "main", "--trials", "3", "--json")
        gate = self.cli("gate", "5", "--max-drop", "0.03", expect=1)
        self.assertIn("passing every trial", gate)
        compare = self.cli("compare", "1", "4")
        self.assertIn("Each case on its own", compare)
        self.assertIn("3 of 3 -> 2 of 3", compare)


class ReviewFixTests(TrialBase):
    """Defects found by an independent check of the repeated-trials feature."""

    EVERYONE = {f"c{i}" for i in range(1, 7)}

    def cli(self, *argv, expect=0):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([*argv, "--store", str(self.tmp / "store"), "--config", str(self.root)])
        self.assertEqual(code, expect, err.getvalue())
        return out.getvalue()

    # 1. the gate never passes an unfinished group
    def test_gate_fails_an_unfinished_candidate_group(self):
        self.target("good")
        self.store.set_baseline(self.run_target("good"))
        slow = self.fake_group("slow", planned=5, finished=1, passes=[self.EVERYONE])
        result = report.gate(self.store, slow[0])
        self.assertFalse(result["ok"])
        self.assertIn("the candidate's repeated trials did not all finish (1 of 5)", result["reason"])
        self.assertIn("so the gate cannot pass it", result["reason"])
        self.assertTrue(any("only 1 of the candidate's 5 planned trials" in w
                            for w in result["comparison"]["warnings"]))
        text = self.cli("gate", str(slow[0]), expect=1)
        self.assertTrue(text.startswith("FAIL: "), text)
        self.assertIn("provisional", text)

    def test_gate_fails_an_unfinished_baseline_group(self):
        self.target("good")
        slow = self.fake_group("slow", planned=5, finished=2, passes=[self.EVERYONE] * 2)
        self.store.set_baseline(slow[0])
        candidate = self.run_target("good", trials=2)
        result = report.gate(self.store, candidate)
        self.assertFalse(result["ok"])
        self.assertIn("the baseline's repeated trials did not all finish (2 of 5)", result["reason"])
        self.assertIsNotNone(result["comparison"])
        self.cli("gate", str(candidate), expect=1)

    def test_gate_still_passes_a_complete_group_and_an_ordinary_run(self):
        self.target("good")
        self.store.set_baseline(self.run_target("good", trials=3))
        self.assertTrue(report.gate(self.store, self.run_target("good", trials=3))["ok"])
        single_base = self.run_target("good")
        self.assertTrue(report.gate(self.store, self.run_target("good"), single_base)["ok"])

    # 2. the report prints the comparison's warnings
    def test_report_prints_the_baseline_warnings(self):
        self.target("good")
        self.store.set_baseline(self.run_target("good", trials=3))
        single = report.format_report(report.build_report(self.store, self.run_target("good")))
        self.assertIn("   warning: the baseline ran 3 trial(s) per case and the candidate 1", single)
        slow = self.fake_group("slow", planned=4, finished=1, passes=[self.EVERYONE])
        grouped = report.format_report(report.build_report(self.store, slow[0]))
        self.assertIn("   warning: only 1 of the candidate's 4 planned trials finished", grouped)
        self.store.set_baseline(self.run_target("good"))
        clean = report.format_report(report.build_report(self.store, self.run_target("good")))
        self.assertNotIn("warning:", clean)

    # 3 and 4. labels use the finished count; a case result is not a trial run or a case
    def test_unfinished_group_is_labelled_by_the_finished_trials(self):
        slow = self.fake_group("slow", planned=5, finished=1, passes=[self.EVERYONE])
        text = report.format_report(report.build_report(self.store, slow[0]))
        self.assertIn("Passed the 1 finished trial (5 planned)", text)
        self.assertIn("Usage (the 1 finished trial (5 planned) together", text)
        self.assertNotIn("all 1 trials", text)
        self.assertNotIn("all 5 trials", text)
        more = self.fake_group("more", planned=5, finished=3, passes=[self.EVERYONE] * 3)
        self.assertIn("Passed all 3 finished trials (5 planned)",
                      report.format_report(report.build_report(self.store, more[0])))

    def test_group_text_says_case_results(self):
        self.write_cases(3)
        self.target("t", cost=0.01, cases={"c1": ["crash"] + ["yes"] * 4, "c2": ["crash"] + ["yes"] * 4,
                                          "c3": ["crash"] + ["yes"] * 4})
        text = report.format_report(report.build_report(self.store, self.run_target("t", trials=5)))
        self.assertIn("(12 of 15 case results)", text)
        self.assertIn("3 case results on 3 cases", text)
        self.assertIn("per passing case result", text)
        self.assertNotIn("trial runs", text)
        self.assertNotIn("passing trial", text)
        self.assertNotIn("per passed case", text)

    def test_usage_line_nouns(self):
        use = {"mixed_units": False, "cost": 0.12, "cost_unit": "credits", "cases_with_cost": 12,
               "cost_per_pass": 0.01, "prompt_tokens": None, "completion_tokens": None, "median_duration_s": 1.0}
        ordinary = report._usage_line(use, 15)
        self.assertIn("reported for 12 of 15 cases)", ordinary)
        self.assertIn("per passed case", ordinary)
        self.assertIn("per case", ordinary)
        grouped = report._usage_line(use, 15, group=True)
        self.assertIn("reported for 12 of 15 case results)", grouped)
        self.assertIn("per passing case result", grouped)
        self.assertNotIn("per passed case", grouped)

    def test_ordinary_report_usage_still_says_cases(self):
        self.target("t", cost=0.01)
        text = report.format_report(report.build_report(self.store, self.run_target("t")))
        self.assertIn("per passed case", text)
        self.assertNotIn("case result", text)

    # 5. which figures the top level describes
    def test_figures_describe(self):
        self.target("t")
        single = report.build_report(self.store, self.run_target("t"))
        self.assertEqual(single["figures_describe"], "this run")
        ids = self.group_ids(self.run_target("t", trials=3))
        rep = report.build_report(self.store, ids[1])
        self.assertEqual(rep["figures_describe"],
                         f"trial 2 of 3 (run {ids[1]}) alone;"
                         " the figures for the whole group of trials are under 'trials'")
        self.assertIn("figures_describe", strict_loads(json.dumps(report.json_safe(rep))))

    # 6. baseline reports the stored run
    def test_baseline_on_a_middle_trial_names_the_stored_run(self):
        self.target("t")
        ids = self.group_ids(self.run_target("t", trials=3))
        text = self.cli("baseline", str(ids[1]))
        self.assertIn(f"run {ids[0]}, the whole group of 3 trials, is now the baseline for evaldemo / main", text)
        self.assertIn(f"(run {ids[1]} is one trial of it)", text)
        self.assertNotIn(f"run {ids[1]} is now", text)
        data = strict_loads(self.cli("baseline", str(ids[1]), "--json"))
        self.assertEqual((data["baseline_run"], data["trials"]), (ids[0], 3))
        self.assertEqual(self.store.get_baseline("evaldemo", "main"), ids[0])
        single = self.run_target("t")
        self.assertIn(f"run {single} is now the baseline for evaldemo / main", self.cli("baseline", str(single)))


# --- MCP --------------------------------------------------------------------------


class McpBoundTests(unittest.TestCase):
    def test_the_agent_bound_is_ten(self):
        self.assertEqual(mcp_server.MAX_TOOL_TRIALS, 10)
        self.assertLess(mcp_server.MAX_TOOL_TRIALS, MAX_TRIALS)


@unittest.skipUnless(Client is not None, "the 'mcp' package is not installed")
class McpTrialTests(TrialBase):
    def setUp(self):
        super().setUp()
        self.target("t", cases={"c1": fail(2)})
        self.home = self.tmp / "home"  # never the user's real store
        self.store_dir = self.tmp / "mcpstore"

    def params(self):
        existing = os.environ.get("PYTHONPATH")
        return StdioServerParameters(
            command=PY, args=["-m", "llmtuner", "mcp", "--store", str(self.store_dir)],
            env={"PYTHONPATH": SRC + (os.pathsep + existing if existing else ""),
                 "LLMTUNER_HOME": str(self.home)})

    def test_trials_are_bounded_and_reported(self):
        asyncio.run(asyncio.wait_for(self._calls(), timeout=120))

    async def _calls(self):
        async with Client(self.params()) as client:
            args = {"project_dir": str(self.root), "target": "t", "dataset": "main"}
            too_many = await client.call_tool("run_eval", {**args, "trials": 11})
            self.assertTrue(too_many.is_error)
            self.assertIn("trials must be a whole number from 1 to 10", too_many.content[0].text)
            zero = await client.call_tool("run_eval", {**args, "trials": 0})
            self.assertTrue(zero.is_error)
            ran = await client.call_tool("run_eval", {**args, "trials": 2})
            self.assertFalse(ran.is_error, ran.content)
            tr = ran.structured_content["trials"]
            self.assertEqual((tr["planned"], tr["finished"], tr["n"]), (2, 2, self.N))
            self.assertEqual(tr["unstable_cases"], ["c1"])
        with Store(self.store_dir) as store:
            self.assertEqual(len(store.list_runs()), 2)  # the rejected calls ran nothing
        self.assertFalse((self.home / "store.db").exists(), "the run leaked into $LLMTUNER_HOME")


class UnfinishedGroupWarningTests(unittest.TestCase):
    """compare and gate must say when a group of trials did not all finish."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(home=self.tmp.name)
        self.addCleanup(self.store.close)

    def make_run(self, passes, finish=True, **group):
        from llmtuner.dataset import Case
        from llmtuner.targets import CommandResult
        run_id = self.store.create_run(project="p", dataset="d", dataset_hash="h", target="t", command=["x"],
                                       model=None, tool_version=None, split="test", graders=[], note=None, **group)
        for index, ok in enumerate(passes):
            self.store.add_case(run_id, Case(f"c{index}", "in"), CommandResult("out", 0, False, 0.1, None), ok, [])
        if finish:
            self.store.finish_run(run_id)
        return run_id

    def test_compare_and_gate_warn_about_an_unfinished_group(self):
        single = self.make_run([True, True])
        first = self.make_run([True, True], trials_planned=3)
        self.make_run([True, True], finish=False, trial_group=first, trial=2, trials_planned=3)
        compared = report.compare_runs(self.store, single, first)
        self.assertTrue(any("only 1 of the candidate's 3 planned trials finished" in w for w in compared["warnings"]),
                        compared["warnings"])
        gated = report.gate(self.store, first, single)
        self.assertIn("provisional", report.format_gate(gated))
        self.assertIn("provisional", report.format_compare(compared))

    def test_figures_describe_needs_no_trial_number(self):
        first = self.make_run([True, True], trials_planned=3)  # created without `trial`
        self.assertTrue(report.build_report(self.store, first)["figures_describe"]
                        .startswith(f"trial 1 of 3 (run {first}) alone;"))

    def test_a_group_with_no_finished_trial_but_recorded_cases_prints_nothing_else(self):
        first = self.make_run([True, True], finish=False, trials_planned=3)
        text = report.format_report(report.build_report(self.store, first))
        self.assertIn("none of the 3 planned trials has finished yet", text)
        for bad in ("all 0", "0 of 0", "Usage", "This trial alone"):
            self.assertNotIn(bad, text)

    def test_a_complete_group_gets_no_such_warning(self):
        single = self.make_run([True, True])
        first = self.make_run([True, True], trials_planned=2)
        self.make_run([True, True], trial_group=first, trial=2, trials_planned=2)
        warnings = report.compare_runs(self.store, single, first)["warnings"]
        self.assertFalse(any("provisional" in w for w in warnings), warnings)

    def test_a_group_with_no_finished_trial_reports_that_plainly(self):
        first = self.make_run([], finish=False, trials_planned=3)
        text = report.format_report(report.build_report(self.store, first))
        self.assertIn("none of the 3 planned trials has finished yet", text)
        self.assertNotIn("0 of 0", text)
        self.assertNotIn("same result each time", text)


if __name__ == "__main__":
    unittest.main()
