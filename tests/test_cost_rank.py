"""Tests for the cost channel (LLMTUNER_META), the store migration, the {python} token and `rank`.

Offline: targets are stub scripts run with sys.executable; the only server is a stub on 127.0.0.1.
The fake key below is not a real credential.
"""

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_openrouter import FAKE, SRC, Stub, ok_body  # noqa: E402

from llmtuner import report  # noqa: E402
from llmtuner.cli import main  # noqa: E402
from llmtuner.config import load_config  # noqa: E402
from llmtuner.runner import run_eval  # noqa: E402
from llmtuner.store import _SCHEMA, Store  # noqa: E402

PY = sys.executable

META_STUB = """\
import json, os, sys, time
text = sys.stdin.read().strip()
mode = sys.argv[1]
path = os.environ["LLMTUNER_META"]
good = {"cost": 0.002, "cost_unit": "credits", "prompt_tokens": 10, "completion_tokens": 2, "model": "m"}
if mode == "good":
    open(path, "w").write(json.dumps(good))
elif mode == "malformed":
    open(path, "w").write("{not json")
elif mode == "list":
    open(path, "w").write("[1, 2, 3]")
elif mode == "big":
    open(path, "w").write(json.dumps({**good, "padding": "x" * 70000}))
elif mode == "odd":  # only odd-numbered cases report a cost
    if int(text[1:]) % 2:
        open(path, "w").write(json.dumps(good))
elif mode == "perinput":
    n = int(text[1:])
    time.sleep((9 - n) * 0.03)  # finish in reverse order so a mix-up would show
    open(path, "w").write(json.dumps({"cost": n / 1000, "cost_unit": "credits", "prompt_tokens": n,
                                      "completion_tokens": 1, "model": text}))
elif mode == "wrong":
    open(path, "w").write(json.dumps(good))
    text = "wrong"
print(text)
"""

# Rank stub: argv = fail-ids (comma separated), cost per case ("none" to report nothing).
RANK_STUB = """\
import json, os, sys
text = sys.stdin.read().strip()
fails = {f for f in sys.argv[1].split(",") if f != "-"}
if sys.argv[2] != "none":
    open(os.environ["LLMTUNER_META"], "w").write(json.dumps({"cost": float(sys.argv[2]), "cost_unit": "credits"}))
print("wrong" if text in fails else text)
"""


def strict_loads(text):
    def refuse(token):
        raise ValueError(f"not valid JSON: {token}")

    return json.loads(text, parse_constant=refuse)


class ProjectCase(unittest.TestCase):
    N = 8

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.root = self.tmp / "project"
        self.root.mkdir()
        self.store_dir = self.tmp / "store"
        self.store = Store(home=self.store_dir)
        self.addCleanup(self.store.close)
        (self.root / "meta_stub.py").write_text(META_STUB)
        (self.root / "rank_stub.py").write_text(RANK_STUB)
        self.targets = {}
        self.extra_grader = False
        self.write_cases()

    def write_cases(self, n=None):
        n = n or self.N
        rows = "\n".join(json.dumps({"id": f"c{i}", "input": f"c{i}", "expected": f"c{i}"})
                         for i in range(1, n + 1)) + "\n"
        (self.root / "cases.jsonl").write_text(rows)

    def add_target(self, name, command):
        self.targets[name] = command
        self.write_config()

    def add_meta_target(self, name, mode):
        self.add_target(name, [PY, "meta_stub.py", mode])

    def add_rank_target(self, name, fails="", cost="none"):
        self.add_target(name, [PY, "rank_stub.py", fails or "-", cost])

    def write_config(self):
        text = 'project = "costdemo"\n'
        for name, command in self.targets.items():
            text += f"[targets.{name}]\ncommand = {json.dumps(command)}\ntimeout = 20\n"
        text += '[graders.same]\ntype = "exact"\n'
        if self.extra_grader:
            text += '[graders.short]\ntype = "max_words"\nlimit = 100\nrequired = false\n'
        text += '[datasets.main]\npath = "cases.jsonl"\n'
        (self.root / "llmtuner.toml").write_text(text)

    def run_target(self, name, jobs=1, dataset="main", **kw):
        return run_eval(load_config(self.root), self.store, name, dataset, jobs=jobs, **kw)


class CostChannelTests(ProjectCase):
    N = 6

    def test_reported_meta_is_stored_and_summed(self):
        self.add_meta_target("t", "good")
        run_id = self.run_target("t")
        cases = self.store.get_cases(run_id)
        self.assertEqual(len(cases), self.N)
        for case in cases:
            self.assertEqual(case["meta"], {"cost": 0.002, "cost_unit": "credits", "prompt_tokens": 10,
                                            "completion_tokens": 2, "model": "m"})
        use = report.build_report(self.store, run_id)["usage"]
        self.assertAlmostEqual(use["cost"], 0.002 * self.N)
        self.assertEqual(use["cost_unit"], "credits")
        self.assertEqual(use["cases_with_cost"], self.N)
        self.assertEqual(use["prompt_tokens"], 10 * self.N)
        self.assertEqual(use["completion_tokens"], 2 * self.N)
        self.assertAlmostEqual(use["cost_per_pass"], 0.002 * self.N / self.N)
        self.assertIsInstance(use["median_duration_s"], float)

    def test_cost_per_pass_divides_by_passed_cases(self):
        self.add_rank_target("half", fails="c1,c2,c3", cost="0.01")
        run_id = self.run_target("half")
        rep = report.build_report(self.store, run_id)
        self.assertEqual(rep["n_passed"], 3)
        self.assertAlmostEqual(rep["usage"]["cost"], 0.06)
        self.assertAlmostEqual(rep["usage"]["cost_per_pass"], 0.02)

    def test_nothing_reported_is_none_not_zero(self):
        self.add_meta_target("t", "none")
        run_id = self.run_target("t")
        for case in self.store.get_cases(run_id):
            self.assertIsNone(case["meta"])
        use = report.build_report(self.store, run_id)["usage"]
        for key in ("cost", "cost_unit", "cost_per_pass", "prompt_tokens", "completion_tokens"):
            self.assertIsNone(use[key], key)
        self.assertEqual(use["cases_with_cost"], 0)
        self.assertIsInstance(use["median_duration_s"], float)
        self.assertIn("cost not reported", report.format_report(report.build_report(self.store, run_id)))

    def test_bad_meta_files_give_none_and_the_run_completes(self):
        for mode in ("malformed", "list", "big"):
            with self.subTest(mode=mode):
                self.targets = {}
                self.add_meta_target(mode, mode)
                run_id = self.run_target(mode)
                run = self.store.get_run(run_id)
                self.assertIsNotNone(run["finished_at"])
                cases = self.store.get_cases(run_id)
                self.assertEqual(len(cases), self.N)
                self.assertTrue(all(c["passed"] for c in cases))
                self.assertTrue(all(c["meta"] is None for c in cases))
                self.assertIsNone(report.build_report(self.store, run_id)["usage"]["cost"])

    def test_mixed_reporting(self):
        self.add_meta_target("t", "odd")
        run_id = self.run_target("t")
        rep = report.build_report(self.store, run_id)
        self.assertEqual(rep["usage"]["cases_with_cost"], 3)
        self.assertAlmostEqual(rep["usage"]["cost"], 0.006)
        self.assertIn("reported for 3 of 6 cases", report.format_report(rep))

    def test_all_cases_fail_but_cost_is_reported(self):
        self.add_meta_target("t", "wrong")
        run_id = self.run_target("t")
        rep = report.build_report(self.store, run_id)
        self.assertEqual(rep["n_passed"], 0)
        self.assertAlmostEqual(rep["usage"]["cost"], 0.002 * self.N)
        self.assertIsNone(rep["usage"]["cost_per_pass"])
        self.assertIn("no case passed", report.format_report(rep))

    def test_parallel_jobs_keep_each_cases_meta(self):
        self.N = 8
        self.write_cases(8)
        self.add_meta_target("t", "perinput")
        run_id = self.run_target("t", jobs=4)
        cases = self.store.get_cases(run_id)
        self.assertEqual(len(cases), 8)
        for case in cases:
            n = int(case["case_id"][1:])
            self.assertEqual(case["meta"]["model"], case["case_id"])
            self.assertAlmostEqual(case["meta"]["cost"], n / 1000)
            self.assertEqual(case["meta"]["prompt_tokens"], n)
        use = report.build_report(self.store, run_id)["usage"]
        self.assertAlmostEqual(use["cost"], sum(range(1, 9)) / 1000)
        self.assertEqual(use["cases_with_cost"], 8)

    def test_usage_function_directly(self):
        cases = [{"meta": {"cost": 1, "cost_unit": "credits", "prompt_tokens": 4}, "duration_s": 1.0},
                 {"meta": {"cost": True, "completion_tokens": "7"}, "duration_s": 3.0},  # junk values ignored
                 {"meta": None, "duration_s": 2.0}]
        use = report.usage(cases, 2)
        self.assertEqual(use["cost"], 1)
        self.assertEqual(use["cases_with_cost"], 1)
        self.assertEqual(use["cost_per_pass"], 0.5)
        self.assertEqual(use["prompt_tokens"], 4)
        self.assertIsNone(use["completion_tokens"])
        self.assertEqual(use["median_duration_s"], 2.0)
        empty = report.usage([], 0)
        self.assertIsNone(empty["cost"])
        self.assertIsNone(empty["median_duration_s"])


class PythonTokenTests(ProjectCase):
    N = 2

    def test_token_runs_under_sys_executable_and_is_stored_literally(self):
        self.add_target("t", ["{python}", "-c", "import sys; print(sys.executable)"])
        (self.root / "cases.jsonl").write_text(
            "\n".join(json.dumps({"id": f"c{i}", "input": "x", "expected": PY}) for i in (1, 2)) + "\n")
        run_id = self.run_target("t")
        for case in self.store.get_cases(run_id):
            self.assertEqual(case["output"].strip(), PY)
            self.assertTrue(case["passed"], case)
        self.assertEqual(self.store.get_run(run_id)["command"][0], "{python}")

    def test_token_in_a_judge_command(self):
        (self.root / "judge.py").write_text(
            "import sys\nsys.stdin.read()\n"
            "print('VERDICT: NO' if sys.executable == sys.argv[1] else 'VERDICT: YES')\nprint('EVIDENCE: x')\n")
        self.add_target("t", [PY, "meta_stub.py", "none"])
        config_path = self.root / "llmtuner.toml"
        config_path.write_text(config_path.read_text().replace(
            '[graders.same]', f'[graders.j]\ntype = "judge"\nquestion = "Is it bad?"\n'
            f'command = ["{{python}}", "judge.py", {json.dumps(PY)}]\n[graders.same]'))
        run_id = self.run_target("t")
        grades = [g for g in self.store.get_grades(run_id) if g["grader"] == "j"]
        self.assertEqual(len(grades), self.N)
        for g in grades:
            self.assertFalse(g["error"], g)
            self.assertTrue(g["passed"], g)  # fail_on "yes": NO means the output is fine
        self.assertTrue(all(c["passed"] for c in self.store.get_cases(run_id)))


class KeyNeverStoredTests(ProjectCase):
    N = 2

    def run_adapter_target(self, stub):
        self.add_target("or", ["{python}", "-m", "llmtuner.adapters.openrouter", "--model", "m",
                               "--base-url", stub.url])
        home = self.tmp / "home"
        home.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER_", "LLMTUNER_"))}
        env.update({"OPENROUTER_API_KEY": FAKE, "PYTHONPATH": SRC, "HOME": str(home)})
        with mock.patch.dict(os.environ, env, clear=True):
            run_id = self.run_target("or")
        self.store.close()
        self.assertTrue(self.store_dir.joinpath("store.db").is_file())
        return run_id

    def store_bytes(self):
        return b"".join(p.read_bytes() for p in sorted(self.store_dir.iterdir()) if p.is_file())

    def test_success_run(self):
        (self.root / "cases.jsonl").write_text(
            "\n".join(json.dumps({"id": f"c{i}", "input": f"q{i}", "expected": "hello"}) for i in (1, 2)) + "\n")
        with Stub((200, {}, ok_body("hello"))) as stub:
            run_id = self.run_adapter_target(stub)
        self.assertEqual(len(stub.requests), 2)
        self.assertEqual(stub.requests[0]["headers"]["Authorization"], f"Bearer {FAKE}")
        data = self.store_bytes()
        self.assertNotIn(FAKE.encode(), data)
        with Store(home=self.store_dir) as store:
            cases = store.get_cases(run_id)
            self.assertTrue(all(c["passed"] for c in cases))
            self.assertEqual(cases[0]["meta"]["cost"], 0.00042)

    def test_failing_run_with_echoed_key(self):
        def echo(request):
            return 401, {}, {"error": {"message": f"unauthorized: {request['headers']['Authorization']}"}}

        with Stub(echo) as stub:
            run_id = self.run_adapter_target(stub)
        self.assertNotIn(FAKE.encode(), self.store_bytes())
        with Store(home=self.store_dir) as store:
            cases = store.get_cases(run_id)
            self.assertFalse(any(c["passed"] for c in cases))
            self.assertIn("401", cases[0]["error"])
            self.assertIn("***", cases[0]["error"])


class MigrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        old_schema = _SCHEMA.replace("  meta TEXT,\n", "")
        self.assertNotEqual(old_schema, _SCHEMA)
        db = sqlite3.connect(self.home / "store.db")
        db.executescript(old_schema)
        db.execute("INSERT INTO runs (project, dataset, dataset_hash, target, command, model, tool_version,"
                   " split, graders, note, started_at, finished_at) VALUES"
                   " ('p','d','h','t','[\"x\"]',NULL,NULL,'test','[]',NULL,1.0,2.0)")
        db.execute("INSERT INTO cases VALUES (1,'old1','in',NULL,NULL,'out',0,0,0.5,NULL,1)")
        db.commit()
        db.close()

    def columns(self):
        db = sqlite3.connect(self.home / "store.db")
        try:
            return [r[1] for r in db.execute("PRAGMA table_info(cases)")]
        finally:
            db.close()

    def test_old_store_gains_the_column(self):
        self.assertNotIn("meta", self.columns())
        with Store(home=self.home) as store:
            pass
        self.assertEqual(self.columns().count("meta"), 1)

    def test_old_row_reads_back_and_new_runs_can_be_written(self):
        from llmtuner.dataset import Case
        from llmtuner.targets import CommandResult

        with Store(home=self.home) as store:
            old = store.get_cases(1)
            self.assertEqual(len(old), 1)
            self.assertEqual(old[0]["case_id"], "old1")
            self.assertIsNone(old[0]["meta"])
            run_id = store.create_run(project="p", dataset="d", dataset_hash="h2", target="t2", command=["x"],
                                      model=None, tool_version=None, split="test", graders=[], note=None)
            store.add_case(run_id, Case(id="new1", input="i", expected="e"), CommandResult("o", 0, False, 0.1, None),
                           True, [], {"cost": 1.5})
            store.finish_run(run_id)
            self.assertEqual(store.get_cases(run_id)[0]["meta"], {"cost": 1.5})
            self.assertIsNone(store.get_cases(1)[0]["meta"])

    def test_second_open_changes_nothing(self):
        with Store(home=self.home) as store:
            first = store.get_cases(1)
        columns = self.columns()
        with Store(home=self.home) as store:
            self.assertEqual(store.get_cases(1), first)
        self.assertEqual(self.columns(), columns)


class RankTests(ProjectCase):
    def rank(self, **kw):
        return report.rank(self.store, "costdemo", "main", **kw)

    def test_no_runs(self):
        result = self.rank()
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["skipped"], [])
        self.assertIn("no finished runs", report.format_rank(result))

    def test_ordered_by_pass_rate_then_cost(self):
        self.add_rank_target("cheap_bad", fails="c1,c2,c3", cost="0.0001")
        self.add_rank_target("dear_good", fails="", cost="0.5")
        self.run_target("cheap_bad")
        self.run_target("dear_good")
        rows = self.rank()["rows"]
        self.assertEqual([r["target"] for r in rows], ["dear_good", "cheap_bad"])
        self.assertEqual(rows[0]["n_passed"], 8)
        self.assertEqual(rows[1]["n_passed"], 5)

    def test_tie_goes_to_lower_cost_per_pass_and_unknown_last(self):
        self.add_rank_target("unknown", fails="c1", cost="none")
        self.add_rank_target("dear", fails="c1", cost="0.05")
        self.add_rank_target("cheap", fails="c1", cost="0.001")
        for name in ("unknown", "dear", "cheap"):
            self.run_target(name)
        rows = self.rank()["rows"]
        self.assertEqual([r["target"] for r in rows], ["cheap", "dear", "unknown"])
        self.assertIsNone(rows[2]["usage"]["cost_per_pass"])

    def test_different_dataset_version_is_skipped(self):
        self.add_rank_target("old", fails="", cost="0.01")
        self.add_rank_target("a", fails="", cost="0.01")
        self.add_rank_target("b", fails="c1", cost="0.01")
        self.run_target("old")
        self.write_cases(9)  # the dataset file changes between runs
        self.run_target("a")
        self.run_target("b")
        result = self.rank()
        self.assertEqual({r["target"] for r in result["rows"]}, {"a", "b"})
        self.assertEqual([(s["target"], s["reason"]) for s in result["skipped"]],
                         [("old", "ran on a different version of the cases")])
        self.assertIn("skipped run", report.format_rank(result))

    def test_different_graders_are_skipped(self):
        self.add_rank_target("strict", fails="", cost="0.01")
        self.add_rank_target("a", fails="", cost="0.01")
        self.add_rank_target("b", fails="c1", cost="0.01")
        self.run_target("strict")
        self.extra_grader = True
        self.write_config()
        self.run_target("a")
        self.run_target("b")
        result = self.rank()
        self.assertEqual({r["target"] for r in result["rows"]}, {"a", "b"})
        self.assertEqual([(s["target"], s["reason"]) for s in result["skipped"]],
                         [("strict", "ran with different graders")])

    def test_newest_run_per_target_is_used(self):
        self.add_rank_target("t", fails="c1,c2", cost="0.01")
        first = self.run_target("t")
        self.add_rank_target("t", fails="", cost="0.01")
        second = self.run_target("t")
        rows = self.rank()["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run"], second)
        self.assertNotEqual(rows[0]["run"], first)
        self.assertEqual(rows[0]["n_passed"], 8)

    def test_unfinished_run_is_ignored(self):
        self.add_rank_target("t", fails="c1", cost="0.01")
        finished = self.run_target("t")
        dataset_hash = self.store.get_run(finished)["dataset_hash"]
        self.store.create_run(project="costdemo", dataset="main", dataset_hash=dataset_hash, target="t",
                              command=["x"], model=None, tool_version=None, split="test", graders=[], note=None)
        self.store.create_run(project="costdemo", dataset="main", dataset_hash="zzz", target="ghost",
                              command=["x"], model=None, tool_version=None, split="test", graders=[], note=None)
        rows = self.rank()["rows"]
        self.assertEqual([(r["target"], r["run"]) for r in rows], [("t", finished)])
        self.assertEqual(self.rank()["skipped"], [])

    def test_other_dataset_or_project_has_no_rows(self):
        self.add_rank_target("t")
        self.run_target("t")
        self.assertEqual(report.rank(self.store, "costdemo", "nope")["rows"], [])
        self.assertEqual(report.rank(self.store, "other", "main")["rows"], [])
        self.assertEqual(self.rank(split="dev")["rows"], [])

    def test_verdict_sentences(self):
        self.add_rank_target("leader", fails="", cost="0.01")
        self.add_rank_target("close", fails="c1", cost="0.01")
        self.add_rank_target("far", fails="c1,c2,c3,c4,c5,c6", cost="0.01")
        for name in ("leader", "close", "far"):
            self.run_target(name)
        by_target = {r["target"]: r for r in self.rank()["rows"]}
        self.assertEqual(by_target["leader"]["vs_leader"], "leader")
        self.assertNotIn("p_value", by_target["leader"])
        self.assertEqual(by_target["close"]["vs_leader"], "not measurably different from the leader")
        self.assertEqual(by_target["far"]["vs_leader"], "measurably worse than the leader")
        self.assertLess(by_target["far"]["p_value"], 0.05)
        text = report.format_rank(self.rank())
        self.assertIn("leader", text)
        self.assertIn("measurably worse than the leader", text)

    def test_five_regressions_are_not_enough(self):
        self.add_rank_target("leader", fails="", cost="0.01")
        self.add_rank_target("five", fails="c1,c2,c3,c4,c5", cost="0.01")
        self.run_target("leader")
        self.run_target("five")
        rows = {r["target"]: r for r in self.rank()["rows"]}
        self.assertEqual(rows["five"]["vs_leader"], "not measurably different from the leader")

    def test_split_all(self):
        self.add_rank_target("t")
        self.run_target("t", split="all")
        self.assertEqual(len(self.rank(split="all")["rows"]), 1)
        self.assertEqual(self.rank(split="test")["rows"], [])


class RankCliTests(ProjectCase):
    def cli(self, *argv, json_out=False):
        out, err = io.StringIO(), io.StringIO()
        args = [argv[0], "--config", str(self.root), "--store", str(self.store_dir), *argv[1:]]
        if json_out:
            args.append("--json")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
        self.assertEqual(code, 0, f"{argv}: stdout={out.getvalue()!r} stderr={err.getvalue()!r}")
        return out.getvalue(), err.getvalue()

    def test_table_and_json(self):
        self.add_rank_target("good", fails="", cost="0.01")
        self.add_rank_target("bad", fails="c1,c2,c3,c4,c5,c6", cost="0.002")
        self.cli("run", "good", "main")
        self.cli("run", "bad", "main")
        text, _ = self.cli("rank", "main")
        self.assertIn("1. good", text)
        self.assertIn("2. bad", text)
        self.assertIn("leader", text)
        self.assertIn("measurably worse than the leader", text)
        self.assertIn("per passed case", text)
        out, _ = self.cli("rank", "main", json_out=True)
        parsed = strict_loads(out)
        self.assertEqual([r["target"] for r in parsed["rows"]], ["good", "bad"])
        self.assertEqual(parsed["project"], "costdemo")

    def test_json_when_no_target_reports_cost(self):
        self.add_rank_target("a", fails="")
        self.add_rank_target("b", fails="c1")
        self.cli("run", "a", "main")
        self.cli("run", "b", "main")
        text, _ = self.cli("rank", "main")
        self.assertIn("not reported", text)
        parsed = strict_loads(self.cli("rank", "main", json_out=True)[0])
        for row in parsed["rows"]:
            self.assertIsNone(row["usage"]["cost"])
            self.assertIsNone(row["usage"]["cost_per_pass"])

    def test_no_runs(self):
        self.add_rank_target("a")
        text, _ = self.cli("rank", "main")
        self.assertIn("no finished runs", text)
        parsed = strict_loads(self.cli("rank", "main", json_out=True)[0])
        self.assertEqual(parsed["rows"], [])

    def test_project_option(self):
        self.add_rank_target("a")
        self.cli("run", "a", "main")
        text, _ = self.cli("rank", "main", "--project", "someone-else")
        self.assertIn("no finished runs", text)


if __name__ == "__main__":
    unittest.main()
