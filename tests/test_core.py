"""Offline tests for config, dataset, targets, graders, runner and store.

Targets and judges are tiny stub scripts run with sys.executable in a temp dir.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from llmtuner.config import Config, ConfigError, Grader, load_config
from llmtuner.dataset import Case, DatasetError, load_cases, select_split
from llmtuner.graders import check_case, grade
from llmtuner.runner import TARGET_GRADER, run_eval
from llmtuner.store import Store, StoreError
from llmtuner.targets import run_command

PY = sys.executable

TARGET_STUB = """\
import sys, time
text = sys.stdin.read()
log = sys.argv[1] if len(sys.argv) > 1 else None
if log:
    with open(log, "a") as fh:
        fh.write("call\\n")
if "SLEEP" in text:
    time.sleep(30)
if "CRASH" in text:
    sys.stderr.write("boom")
    sys.exit(3)
if text.startswith("delay:"):
    time.sleep(float(text.split(":", 1)[1].split()[0]))
    print("ok")
else:
    print(text.strip())
"""

JUDGE_STUB = """\
import sys
sys.stdin.read()
mode = sys.argv[1]
if mode == "yes":
    print("VERDICT: YES\\nEVIDENCE: none")
elif mode == "no":
    print("VERDICT: NO\\nEVIDENCE: none")
elif mode == "silent":
    print("I am not sure what to say here.")
elif mode == "fail":
    sys.exit(5)
"""


def q(value) -> str:
    """TOML-safe quoted string (JSON strings are valid TOML basic strings)."""
    return json.dumps(str(value))


def mk_grader(name, kind, required=True, **options) -> Grader:
    return Grader(name=name, type=kind, required=required, options=options)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "target.py").write_text(TARGET_STUB)
        (self.root / "judge.py").write_text(JUDGE_STUB)

    def write_config(self, text: str) -> Path:
        path = self.root / "llmtuner.toml"
        path.write_text(text)
        return path

    def write_cases(self, rows, name="cases.jsonl") -> Path:
        path = self.root / name
        path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
        return path

    def store(self) -> Store:
        s = Store(home=self.root / "home")
        self.addCleanup(s.close)
        return s


# --- config -----------------------------------------------------------------


class ConfigTests(TmpCase):
    BASE = 'project = "demo"\n'

    def target_block(self, extra=""):
        return f"[targets.app]\ncommand = [{q(PY)}, \"target.py\"]\n{extra}\n"

    def test_valid_file_loads(self):
        self.write_config(
            self.BASE
            + self.target_block("timeout = 5\nmodel = \"m1\"\n")
            + '[graders.has_ok]\ntype = "contains_all"\nvalues = ["ok"]\n'
            + '[graders.short]\ntype = "max_words"\nlimit = 20\nrequired = false\n'
            + f'[graders.j]\ntype = "judge"\nquestion = "Is it fine?"\ncommand = [{q(PY)}, "judge.py", "no"]\n'
            + '[datasets.main]\npath = "cases.jsonl"\ngraders = ["has_ok"]\n'
            + '[datasets.everything]\npath = "cases.jsonl"\n'
        )
        cfg = load_config(self.root)
        self.assertEqual(cfg.project, "demo")
        self.assertEqual(cfg.root, self.root.resolve())
        self.assertEqual(cfg.target("app").command, (PY, "target.py"))
        self.assertEqual(cfg.target("app").timeout, 5.0)
        self.assertEqual(cfg.target("app").model, "m1")
        self.assertFalse(cfg.graders["short"].required)
        self.assertEqual(cfg.graders["j"].options["fail_on"], "yes")  # default
        self.assertEqual(cfg.dataset("main").path, (self.root / "cases.jsonl").resolve())
        self.assertEqual([g.name for g in cfg.graders_for(cfg.dataset("main"))], ["has_ok"])
        self.assertEqual(len(cfg.graders_for(cfg.dataset("everything"))), 3)
        info = cfg.describe()
        self.assertEqual(info["project"], "demo")
        self.assertIn("app", info["targets"])

    def test_load_by_file_path(self):
        path = self.write_config(self.BASE + self.target_block())
        self.assertEqual(load_config(path).project, "demo")

    def test_missing_config_file(self):
        with self.assertRaises(ConfigError):
            load_config(self.root / "nowhere")

    def test_unknown_target_and_dataset_lookup(self):
        self.write_config(self.BASE + self.target_block())
        cfg = load_config(self.root)
        with self.assertRaises(ConfigError):
            cfg.target("nope")
        with self.assertRaises(ConfigError):
            cfg.dataset("nope")

    def test_string_command_rejected(self):
        self.write_config(self.BASE + '[targets.app]\ncommand = "python3 target.py"\n')
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_empty_command_rejected(self):
        self.write_config(self.BASE + "[targets.app]\ncommand = []\n")
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_judge_string_command_rejected(self):
        self.write_config(
            self.BASE + '[graders.j]\ntype = "judge"\nquestion = "q?"\ncommand = "python3 judge.py"\n'
        )
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_unknown_grader_type_rejected(self):
        self.write_config(self.BASE + '[graders.x]\ntype = "vibes"\n')
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_unknown_grader_option_rejected(self):
        self.write_config(self.BASE + '[graders.x]\ntype = "nonempty"\nbogus = 1\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.root)
        self.assertIn("bogus", str(ctx.exception))

    def test_dataset_unknown_grader_rejected(self):
        self.write_config(
            self.BASE + '[graders.a]\ntype = "nonempty"\n[datasets.d]\npath = "c.jsonl"\ngraders = ["a", "ghost"]\n'
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.root)
        self.assertIn("ghost", str(ctx.exception))

    def test_dataset_without_path_rejected(self):
        self.write_config(self.BASE + "[datasets.d]\n")
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_missing_project_rejected(self):
        self.write_config(self.target_block())
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_blank_project_rejected(self):
        self.write_config('project = "  "\n')
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_contains_all_both_values_and_field_rejected(self):
        self.write_config(self.BASE + '[graders.x]\ntype = "contains_all"\nvalues = ["a"]\nfield = "k"\n')
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_contains_all_neither_values_nor_field_rejected(self):
        self.write_config(self.BASE + '[graders.x]\ntype = "contains_all"\n')
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_contains_all_field_only_ok(self):
        self.write_config(self.BASE + '[graders.x]\ntype = "contains_all"\nfield = "k"\n')
        self.assertEqual(load_config(self.root).graders["x"].options["field"], "k")

    def test_bad_regex_and_bad_fail_on_rejected(self):
        self.write_config(self.BASE + '[graders.x]\ntype = "regex"\npattern = "("\n')
        with self.assertRaises(ConfigError):
            load_config(self.root)
        self.write_config(
            self.BASE + f'[graders.j]\ntype = "judge"\nquestion = "q"\ncommand = [{q(PY)}]\nfail_on = "maybe"\n'
        )
        with self.assertRaises(ConfigError):
            load_config(self.root)

    def test_invalid_toml_rejected(self):
        self.write_config("project = \n")
        with self.assertRaises(ConfigError):
            load_config(self.root)


# --- dataset ----------------------------------------------------------------


class DatasetTests(TmpCase):
    def test_loads_cases_and_hash(self):
        path = self.write_cases(
            [
                {"id": "a", "input": "x", "expected": "y"},
                {"id": 2, "input": {"k": 1}, "split": "dev", "context": "ctx"},
            ]
        )
        cases, digest = load_cases(path)
        self.assertEqual([c.id for c in cases], ["a", "2"])
        self.assertEqual(cases[0].split, "test")
        self.assertEqual(cases[1].split, "dev")
        self.assertEqual(cases[1].input_text, '{"k": 1}')
        self.assertEqual(cases[1].context, "ctx")
        self.assertEqual(len(digest), 16)
        self.assertEqual(digest, load_cases(path)[1])

    def test_blank_lines_ignored(self):
        path = self.write_cases([{"id": "a", "input": "x"}, "", {"id": "b", "input": "y"}])
        self.assertEqual(len(load_cases(path)[0]), 2)

    def test_duplicate_ids(self):
        path = self.write_cases([{"id": "a", "input": "x"}, {"id": "a", "input": "y"}])
        with self.assertRaises(DatasetError):
            load_cases(path)

    def test_invalid_json(self):
        path = self.write_cases([{"id": "a", "input": "x"}, "{not json"])
        with self.assertRaises(DatasetError):
            load_cases(path)

    def test_bad_split(self):
        path = self.write_cases([{"id": "a", "input": "x", "split": "train"}])
        with self.assertRaises(DatasetError):
            load_cases(path)

    def test_missing_id_or_input(self):
        with self.assertRaises(DatasetError):
            load_cases(self.write_cases([{"input": "x"}]))
        with self.assertRaises(DatasetError):
            load_cases(self.write_cases([{"id": "a"}], name="b.jsonl"))

    def test_non_object_line(self):
        with self.assertRaises(DatasetError):
            load_cases(self.write_cases(["[1, 2]"]))

    def test_non_string_context(self):
        with self.assertRaises(DatasetError):
            load_cases(self.write_cases([{"id": "a", "input": "x", "context": 5}]))

    def test_empty_and_missing_file(self):
        with self.assertRaises(DatasetError):
            load_cases(self.write_cases([""], name="empty.jsonl"))
        with self.assertRaises(DatasetError):
            load_cases(self.root / "missing.jsonl")

    def test_select_split(self):
        cases = [
            Case("a", "x", split="dev"),
            Case("b", "x", split="test"),
            Case("c", "x", split="test"),
        ]
        self.assertEqual([c.id for c in select_split(cases, "dev")], ["a"])
        self.assertEqual([c.id for c in select_split(cases, "test")], ["b", "c"])
        self.assertEqual([c.id for c in select_split(cases, "all")], ["a", "b", "c"])

    def test_select_split_errors(self):
        with self.assertRaises(DatasetError):
            select_split([Case("a", "x", split="test")], "dev")  # none in dev
        with self.assertRaises(DatasetError):
            select_split([Case("a", "x")], "train")


# --- targets ----------------------------------------------------------------


class TargetTests(TmpCase):
    def run_stub(self, text, timeout=10.0):
        return run_command([PY, "target.py"], text, timeout, self.root)

    def test_normal_run(self):
        r = self.run_stub("hello there")
        self.assertTrue(r.ok)
        self.assertIsNone(r.error)
        self.assertEqual(r.exit_code, 0)
        self.assertFalse(r.timed_out)
        self.assertEqual(r.output.strip(), "hello there")

    def test_nonzero_exit(self):
        r = self.run_stub("CRASH")
        self.assertFalse(r.ok)
        self.assertEqual(r.exit_code, 3)
        self.assertFalse(r.timed_out)
        self.assertIn("exit code 3", r.error)
        self.assertIn("boom", r.error)

    def test_missing_executable(self):
        r = run_command(["definitely-not-a-real-binary-xyz"], "", 5.0, self.root)
        self.assertIsNotNone(r.error)
        self.assertIsNone(r.exit_code)
        self.assertFalse(r.ok)

    def test_timeout_is_returned_quickly(self):
        start = time.monotonic()
        r = self.run_stub("SLEEP", timeout=0.5)
        elapsed = time.monotonic() - start
        self.assertTrue(r.timed_out)
        self.assertIsNotNone(r.error)
        self.assertIn("timed out", r.error)
        self.assertLess(elapsed, 10.0)  # the stub sleeps 30s; it must have been killed


# --- code graders -----------------------------------------------------------


class CodeGraderTests(unittest.TestCase):
    cwd = Path(".")

    def g(self, kind, output, case=None, **options):
        case = case or Case("c1", "in")
        return grade(mk_grader("g", kind, **options), case, output, self.cwd)

    def test_nonempty(self):
        self.assertTrue(self.g("nonempty", "hi").passed)
        self.assertFalse(self.g("nonempty", "  \n ").passed)

    def test_exact(self):
        case = Case("c", "in", expected="Paris")
        self.assertTrue(self.g("exact", " Paris\n", case).passed)
        self.assertTrue(self.g("exact", "paris", case).passed)  # ignore_case defaults on
        self.assertFalse(self.g("exact", "paris", case, ignore_case=False).passed)
        self.assertTrue(self.g("exact", "Paris", case, ignore_case=False).passed)
        self.assertFalse(self.g("exact", "London", case).passed)

    def test_exact_non_string_expected_and_field(self):
        case = Case("c", "in", expected={"answer": "42", "n": [1, 2]})
        self.assertTrue(self.g("exact", "42", case, field="answer").passed)
        self.assertFalse(self.g("exact", "43", case, field="answer").passed)
        self.assertTrue(self.g("exact", "[1, 2]", case, field="n").passed)
        self.assertTrue(self.g("exact", '{"a": 1}', Case("c", "in", expected={"a": 1})).passed)

    def test_contains_all_values(self):
        self.assertTrue(self.g("contains_all", "Alpha and BETA", values=["alpha", "beta"]).passed)
        r = self.g("contains_all", "alpha only", values=["alpha", "beta"])
        self.assertFalse(r.passed)
        self.assertIn("beta", r.detail)

    def test_contains_all_field(self):
        case = Case("c", "in", expected={"must": ["red", "Blue"]})
        self.assertTrue(self.g("contains_all", "BLUE and red", case, field="must").passed)
        self.assertFalse(self.g("contains_all", "red", case, field="must").passed)

    def test_not_contains(self):
        self.assertTrue(self.g("not_contains", "all good", values=["bad", "evil"]).passed)
        r = self.g("not_contains", "This is EVIL", values=["bad", "evil"])
        self.assertFalse(r.passed)
        self.assertIn("evil", r.detail)

    def test_regex(self):
        self.assertTrue(self.g("regex", "order 123 ok", pattern=r"\d{3}").passed)
        self.assertFalse(self.g("regex", "order ok", pattern=r"\d{3}").passed)

    def test_max_words(self):
        self.assertTrue(self.g("max_words", "one two three", limit=3).passed)
        self.assertFalse(self.g("max_words", "one two three four", limit=3).passed)

    def test_json(self):
        self.assertTrue(self.g("json", '{"a": 1}').passed)
        self.assertFalse(self.g("json", "not json").passed)
        self.assertTrue(self.g("json", '{"a": 1, "b": 2}', required_keys=["a", "b"]).passed)
        r = self.g("json", '{"a": 1}', required_keys=["a", "b"])
        self.assertFalse(r.passed)
        self.assertIn("b", r.detail)
        self.assertFalse(self.g("json", "[1, 2]", required_keys=["a"]).passed)
        self.assertTrue(self.g("json", "[1, 2]").passed)

    def test_code_graders_never_flag_error(self):
        self.assertFalse(self.g("nonempty", "").error)

    def test_check_case(self):
        with self.assertRaises(DatasetError):
            check_case(mk_grader("g", "exact"), Case("c", "in"))
        with self.assertRaises(DatasetError):
            check_case(mk_grader("g", "exact", field="a"), Case("c", "in", expected="s"))
        with self.assertRaises(DatasetError):
            check_case(mk_grader("g", "contains_all", field="a"), Case("c", "in", expected={"a": "str"}))
        check_case(mk_grader("g", "contains_all", field="a"), Case("c", "in", expected={"a": ["x"]}))
        check_case(mk_grader("g", "nonempty"), Case("c", "in"))


# --- judge grader -----------------------------------------------------------


class JudgeGraderTests(TmpCase):
    def judge(self, mode, fail_on="yes", **extra):
        g = mk_grader(
            "j", "judge", command=(PY, str(self.root / "judge.py"), mode), question="Is it bad?",
            fail_on=fail_on, **extra,
        )
        return grade(g, Case("c1", "the input", expected="ref", context="some context"), "the output", self.root)

    def test_yes_with_fail_on_yes_fails(self):
        r = self.judge("yes")
        self.assertFalse(r.passed)
        self.assertFalse(r.error)

    def test_no_with_fail_on_yes_passes(self):
        r = self.judge("no")
        self.assertTrue(r.passed)
        self.assertFalse(r.error)

    def test_fail_on_no_inverts(self):
        self.assertTrue(self.judge("yes", fail_on="no").passed)
        self.assertFalse(self.judge("no", fail_on="no").passed)

    def test_no_verdict_is_error(self):
        r = self.judge("silent")
        self.assertTrue(r.error)
        self.assertFalse(r.passed)

    def test_nonzero_exit_is_error(self):
        r = self.judge("fail")
        self.assertTrue(r.error)
        self.assertFalse(r.passed)

    def test_missing_judge_command_is_error(self):
        g = mk_grader("j", "judge", command=("no-such-judge-binary-xyz",), question="q", fail_on="yes")
        r = grade(g, Case("c", "in"), "out", self.root)
        self.assertTrue(r.error)
        self.assertFalse(r.passed)

    def test_prompt_carries_case_data(self):
        echo = self.root / "echo_judge.py"
        echo.write_text(
            "import sys\n"
            "p = sys.stdin.read()\n"
            "need = ['Is it bad?', 'the input', 'some context', '\"ref\"', 'the output']\n"
            "print('VERDICT: NO' if all(n in p for n in need) else 'VERDICT: YES')\n"
        )
        g = mk_grader("j", "judge", command=(PY, str(echo)), question="Is it bad?", fail_on="yes",
                      include_expected=True)
        r = grade(g, Case("c1", "the input", expected="ref", context="some context"), "the output", self.root)
        self.assertTrue(r.passed, "judge prompt is missing question, input, context, reference or output")


# --- runner -----------------------------------------------------------------


class RunnerTests(TmpCase):
    def config(self, graders_toml=None, timeout=1.0, log=None, dataset="cases.jsonl"):
        cmd = [PY, "target.py"] + ([str(log)] if log else [])
        body = (
            'project = "demo"\n'
            f"[targets.app]\ncommand = {json.dumps(cmd)}\ntimeout = {timeout}\n"
            + (
                graders_toml
                if graders_toml is not None
                else '[graders.same]\ntype = "exact"\n'
            )
            + f'[datasets.main]\npath = "{dataset}"\n'
        )
        self.write_config(body)
        return load_config(self.root)

    def passed_by_case(self, store, run_id):
        return {c["case_id"]: bool(c["passed"]) for c in store.get_cases(run_id)}

    def test_end_to_end_pass_fail(self):
        self.write_cases(
            [
                {"id": "a", "input": "hello", "expected": "hello"},
                {"id": "b", "input": "hello", "expected": "goodbye"},
                {"id": "c", "input": "x y", "expected": "X Y"},
            ]
        )
        store = self.store()
        seen = []
        run_id = run_eval(self.config(), store, "app", "main", on_case=lambda c, p: seen.append((c.id, p)))
        self.assertEqual(self.passed_by_case(store, run_id), {"a": True, "b": False, "c": True})
        self.assertEqual(seen, [("a", True), ("b", False), ("c", True)])
        run = store.get_run(run_id)
        self.assertIsNotNone(run["finished_at"])
        self.assertEqual(run["project"], "demo")
        self.assertEqual(run["dataset"], "main")
        self.assertEqual(run["split"], "test")
        grades = store.get_grades(run_id)
        self.assertEqual([(g["case_id"], g["grader"], g["passed"]) for g in grades],
                         [("a", "same", 1), ("b", "same", 0), ("c", "same", 1)])

    def test_target_crash_and_timeout_recorded_as_failed(self):
        self.write_cases(
            [
                {"id": "ok", "input": "fine", "expected": "fine"},
                {"id": "crash", "input": "CRASH", "expected": "CRASH"},
                {"id": "slow", "input": "SLEEP", "expected": "SLEEP"},
                {"id": "ok2", "input": "fine2", "expected": "fine2"},
            ]
        )
        store = self.store()
        start = time.monotonic()
        run_id = run_eval(self.config(timeout=0.7), store, "app", "main")
        self.assertLess(time.monotonic() - start, 15.0)
        self.assertEqual(
            self.passed_by_case(store, run_id), {"ok": True, "crash": False, "slow": False, "ok2": True}
        )
        self.assertIsNotNone(store.get_run(run_id)["finished_at"])
        cases = {c["case_id"]: c for c in store.get_cases(run_id)}
        self.assertEqual(cases["crash"]["exit_code"], 3)
        self.assertTrue(cases["slow"]["timed_out"])
        grades = {(g["case_id"], g["grader"]): g for g in store.get_grades(run_id)}
        for cid in ("crash", "slow"):
            g = grades[(cid, TARGET_GRADER)]
            self.assertEqual((g["passed"], g["error"]), (0, 1))
            self.assertNotIn((cid, "same"), grades)
        self.assertNotIn(("ok", TARGET_GRADER), grades)

    def test_optional_grader_failure_does_not_fail_case(self):
        self.write_cases([{"id": "a", "input": "hello", "expected": "hello"}])
        graders = (
            '[graders.same]\ntype = "exact"\n'
            '[graders.nice]\ntype = "regex"\npattern = "zzz"\nrequired = false\n'
        )
        store = self.store()
        run_id = run_eval(self.config(graders), store, "app", "main")
        self.assertEqual(self.passed_by_case(store, run_id), {"a": True})
        by_name = {g["grader"]: g["passed"] for g in store.get_grades(run_id)}
        self.assertEqual(by_name, {"same": 1, "nice": 0})

    def test_required_grader_failure_fails_case(self):
        self.write_cases([{"id": "a", "input": "hello", "expected": "hello"}])
        graders = '[graders.same]\ntype = "exact"\n[graders.must]\ntype = "regex"\npattern = "zzz"\n'
        store = self.store()
        run_id = run_eval(self.config(graders), store, "app", "main")
        self.assertEqual(self.passed_by_case(store, run_id), {"a": False})

    def test_no_required_grader_is_config_error(self):
        self.write_cases([{"id": "a", "input": "hello"}])
        graders = '[graders.nice]\ntype = "nonempty"\nrequired = false\n'
        with self.assertRaises(ConfigError):
            run_eval(self.config(graders), self.store(), "app", "main")

    def test_judge_error_in_required_grader_fails_case(self):
        self.write_cases([{"id": "a", "input": "hello"}])
        graders = (
            f'[graders.j]\ntype = "judge"\nquestion = "ok?"\ncommand = [{q(PY)}, "judge.py", "silent"]\n'
        )
        store = self.store()
        run_id = run_eval(self.config(graders), store, "app", "main")
        self.assertEqual(self.passed_by_case(store, run_id), {"a": False})
        self.assertEqual(store.get_grades(run_id)[0]["error"], 1)

    def test_check_case_errors_raise_before_any_target_call(self):
        self.write_cases(
            [
                {"id": "a", "input": "hello", "expected": "hello"},
                {"id": "b", "input": "hello"},  # no expected
            ]
        )
        log = self.root / "calls.log"
        store = self.store()
        with self.assertRaises(DatasetError):
            run_eval(self.config(log=log), store, "app", "main")
        self.assertFalse(log.exists(), "target must not be called when check_case fails")
        self.assertEqual(store.list_runs(), [])

    def test_default_split_is_test_only(self):
        self.write_cases(
            [
                {"id": "d1", "input": "x", "expected": "x", "split": "dev"},
                {"id": "t1", "input": "x", "expected": "x", "split": "test"},
                {"id": "t2", "input": "y", "expected": "y"},
            ]
        )
        cfg, store = self.config(), self.store()
        run_id = run_eval(cfg, store, "app", "main")
        self.assertEqual(list(self.passed_by_case(store, run_id)), ["t1", "t2"])
        dev_id = run_eval(cfg, store, "app", "main", split="dev")
        self.assertEqual(list(self.passed_by_case(store, dev_id)), ["d1"])
        all_id = run_eval(cfg, store, "app", "main", split="all")
        self.assertEqual(list(self.passed_by_case(store, all_id)), ["d1", "t1", "t2"])

    def test_limit(self):
        self.write_cases([{"id": f"c{i}", "input": "x", "expected": "x"} for i in range(5)])
        store = self.store()
        run_id = run_eval(self.config(), store, "app", "main", limit=2)
        self.assertEqual(list(self.passed_by_case(store, run_id)), ["c0", "c1"])

    def test_jobs_4_matches_jobs_1_in_order(self):
        n = 8
        rows = []
        for i in range(n):
            # earlier cases are slower, so out-of-order completion is likely with jobs=4
            delay = (n - i) * 0.05
            rows.append({"id": f"c{i}", "input": f"delay:{delay}", "expected": "ok" if i % 3 else "nope"})
        self.write_cases(rows)
        cfg, store = self.config(), self.store()
        r1 = run_eval(cfg, store, "app", "main", jobs=1)
        r4 = run_eval(cfg, store, "app", "main", jobs=4)

        def summary(run_id):
            return [(c["case_id"], c["passed"], c["output"]) for c in store.get_cases(run_id)]

        self.assertEqual(summary(r1), summary(r4))
        self.assertEqual([c[0] for c in summary(r4)], [f"c{i}" for i in range(n)])
        self.assertEqual([c[1] for c in summary(r4)], [0 if i % 3 == 0 else 1 for i in range(n)])

    def test_unknown_target_or_dataset(self):
        self.write_cases([{"id": "a", "input": "x", "expected": "x"}])
        cfg, store = self.config(), self.store()
        with self.assertRaises(ConfigError):
            run_eval(cfg, store, "ghost", "main")
        with self.assertRaises(ConfigError):
            run_eval(cfg, store, "app", "ghost")


# --- store ------------------------------------------------------------------


class StoreTests(TmpCase):
    def make_run(self, store, project="p", dataset="d", ok=(True,)):
        run_id = store.create_run(
            project=project, dataset=dataset, dataset_hash="h", target="t", command=["x"], model=None,
            tool_version=None, split="test", graders=[], note=None,
        )
        from llmtuner.targets import CommandResult

        for i, passed in enumerate(ok):
            store.add_case(run_id, Case(f"c{i}", "in"), CommandResult("out", 0, False, 0.1, None), passed, [])
        store.finish_run(run_id)
        return run_id

    def test_uses_given_home(self):
        store = self.store()
        self.assertEqual(store.home, self.root / "home")
        self.assertTrue((self.root / "home" / "store.db").exists())

    def test_set_label_unknown_case(self):
        store = self.store()
        run_id = self.make_run(store)
        with self.assertRaises(StoreError):
            store.set_label(run_id, "nope", True)
        with self.assertRaises(StoreError):
            store.set_label(999, "c0", True)

    def test_set_label_round_trip_and_replace(self):
        store = self.store()
        run_id = self.make_run(store)
        store.set_label(run_id, "c0", True, note="fine")
        store.set_label(run_id, "c0", False)
        labels = store.get_labels(run_id)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["passed"], 0)

    def test_baseline_round_trip(self):
        store = self.store()
        self.assertIsNone(store.get_baseline("p", "d"))
        r1 = self.make_run(store)
        r2 = self.make_run(store)
        run = store.set_baseline(r1)
        self.assertEqual(run["id"], r1)
        self.assertEqual(store.get_baseline("p", "d"), r1)
        store.set_baseline(r2)
        self.assertEqual(store.get_baseline("p", "d"), r2)
        self.assertIsNone(store.get_baseline("p", "other"))
        self.assertIsNone(store.get_baseline("other", "d"))
        with self.assertRaises(StoreError):
            store.set_baseline(12345)

    def test_latest_run_id(self):
        store = self.store()
        with self.assertRaises(StoreError):
            store.latest_run_id()
        a = self.make_run(store, project="a")
        b = self.make_run(store, project="b")
        self.assertEqual(store.latest_run_id(), b)
        self.assertEqual(store.latest_run_id("a"), a)
        with self.assertRaises(StoreError):
            store.latest_run_id("zzz")

    def test_list_runs_counts(self):
        store = self.store()
        r1 = self.make_run(store, project="a", ok=(True, False, True))
        r2 = self.make_run(store, project="b", ok=(False,))
        r3 = self.make_run(store, project="a", ok=())
        rows = store.list_runs()
        self.assertEqual([r["id"] for r in rows], [r3, r2, r1])  # newest first
        by_id = {r["id"]: r for r in rows}
        self.assertEqual((by_id[r1]["n"], by_id[r1]["n_passed"]), (3, 2))
        self.assertEqual((by_id[r2]["n"], by_id[r2]["n_passed"]), (1, 0))
        self.assertEqual((by_id[r3]["n"], by_id[r3]["n_passed"]), (0, 0))
        self.assertEqual([r["id"] for r in store.list_runs(project="a")], [r3, r1])
        self.assertEqual(len(store.list_runs(limit=1)), 1)

    def test_get_run_unknown(self):
        with self.assertRaises(StoreError):
            self.store().get_run(1)


# --- fixes: config validation, dataset parsing, bounded targets, odd content, judge verdicts ---------


class ConfigValidationTests(TmpCase):
    BASE = 'project = "demo"\n'

    def rejects(self, body, needle=None):
        self.write_config(self.BASE + body)
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.root)
        if needle:
            self.assertIn(needle, str(ctx.exception))

    def target(self, extra):
        return f"[targets.app]\ncommand = [{q(PY)}, \"target.py\"]\n{extra}\n"

    def test_bad_target_timeouts(self):
        for value in ("0", "-5", '"10"', "true", "false", "[1]"):
            with self.subTest(value=value):
                self.rejects(self.target(f"timeout = {value}"), "timeout")

    def test_good_target_timeouts(self):
        for value in ("5", "0.5"):
            self.write_config(self.BASE + self.target(f"timeout = {value}"))
            self.assertEqual(load_config(self.root).target("app").timeout, float(value))

    def test_bad_judge_timeout(self):
        for value in ("0", "-1", '"x"', "true"):
            with self.subTest(value=value):
                self.rejects(f'[graders.j]\ntype = "judge"\nquestion = "q"\ncommand = [{q(PY)}]\ntimeout = {value}\n',
                             "timeout")

    def test_required_must_be_boolean(self):
        for value in ('"false"', "0", "1"):
            with self.subTest(value=value):
                self.rejects(f'[graders.a]\ntype = "nonempty"\nrequired = {value}\n', "required")

    def test_dataset_graders_must_be_list_of_strings(self):
        head = '[graders.a]\ntype = "nonempty"\n'
        self.rejects(head + '[datasets.d]\npath = "c.jsonl"\ngraders = "a"\n', "graders")
        self.rejects(head + '[datasets.d]\npath = "c.jsonl"\ngraders = [1]\n', "graders")
        self.rejects(head + '[datasets.d]\npath = "c.jsonl"\ngraders = [["a"]]\n', "graders")

    def test_grader_named_target_is_reserved(self):
        self.rejects('[graders.target]\ntype = "nonempty"\n', "reserved")

    def test_reserved_name_matches_runner_constant(self):
        from llmtuner import config

        self.assertEqual(config.TARGET_GRADER, TARGET_GRADER)


class DatasetParsingTests(TmpCase):
    def test_unicode_line_separators_stay_inside_strings(self):
        rows = [{"id": "a", "input": "one\u2028two\u0085three"}, {"id": "b", "input": "x"}]
        cases, _ = load_cases(self.write_cases(rows))
        self.assertEqual([c.id for c in cases], ["a", "b"])
        self.assertEqual(cases[0].input, "one\u2028two\u0085three")

    def test_crlf_line_endings(self):
        path = self.root / "crlf.jsonl"
        path.write_bytes(b'{"id": "a", "input": "x"}\r\n{"id": "b", "input": "y"}\r\n')
        self.assertEqual([c.id for c in load_cases(path)[0]], ["a", "b"])

    def test_non_utf8_is_dataset_error(self):
        path = self.root / "bad.jsonl"
        path.write_bytes(b'{"id": "a", "input": "caf\xe9"}\n')
        with self.assertRaises(DatasetError):
            load_cases(path)


class BoundedTargetTests(TmpCase):
    def test_detached_child_holding_stdout_does_not_block(self):
        (self.root / "detached.py").write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'], start_new_session=True)\n"
            "time.sleep(60)\n"
        )
        start = time.monotonic()
        r = run_command([PY, "detached.py"], "", 1.0, self.root)
        self.assertLess(time.monotonic() - start, 6.0)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.ok)


class OddContentTests(TmpCase):
    def run_one(self, graders, rows, target_script=None):
        if target_script:
            (self.root / "target.py").write_text(target_script)
        self.write_cases(rows)
        self.write_config(
            'project = "demo"\n'
            f"[targets.app]\ncommand = [{q(PY)}, \"target.py\"]\ntimeout = 20\n"
            + graders + '[datasets.main]\npath = "cases.jsonl"\n'
        )
        store = self.store()
        run_id = run_eval(load_config(self.root), store, "app", "main")
        self.assertIsNotNone(store.get_run(run_id)["finished_at"])
        return store, run_id

    def test_deeply_nested_json_output_is_a_failed_case(self):
        store, run_id = self.run_one('[graders.j]\ntype = "json"\n', [{"id": "a", "input": "x"}],
                                     'import sys\nsys.stdin.read()\nprint("[" * 100000)\n')
        self.assertEqual([c["passed"] for c in store.get_cases(run_id)], [0])
        self.assertEqual(store.get_grades(run_id)[0]["passed"], 0)

    def test_json_grader_catches_recursion(self):
        r = grade(mk_grader("j", "json"), Case("c", "x"), "[" * 100000, Path("."))
        self.assertFalse(r.passed)

    def test_lone_surrogate_in_input_is_stored(self):
        rows = ['{"id": "a", "input": "bad \\ud800 text", "expected": "x\\udc00"}']
        store, run_id = self.run_one('[graders.n]\ntype = "nonempty"\n', rows)
        case = store.get_cases(run_id)[0]
        self.assertEqual(case["passed"], 1)
        self.assertNotIn("\ud800", case["input"])
        store.set_label(run_id, "a", True, note="note \ud800")

    def test_crashing_grader_becomes_error_grade(self):
        from unittest import mock

        self.write_cases([{"id": "a", "input": "hello"}])
        self.write_config(
            'project = "demo"\n'
            f"[targets.app]\ncommand = [{q(PY)}, \"target.py\"]\n"
            '[graders.n]\ntype = "nonempty"\n[datasets.main]\npath = "cases.jsonl"\n'
        )
        store = self.store()
        with mock.patch("llmtuner.runner.grade", side_effect=RuntimeError("kaboom")):
            run_id = run_eval(load_config(self.root), store, "app", "main")
        grade_row = store.get_grades(run_id)[0]
        self.assertEqual((grade_row["passed"], grade_row["error"]), (0, 1))
        self.assertIn("grader crashed: RuntimeError: kaboom", grade_row["detail"])
        self.assertIsNotNone(store.get_run(run_id)["finished_at"])

    def test_unexpected_harness_error_is_a_failed_case(self):
        from unittest import mock

        self.write_cases([{"id": "a", "input": "hello"}, {"id": "b", "input": "world"}])
        self.write_config(
            'project = "demo"\n'
            f"[targets.app]\ncommand = [{q(PY)}, \"target.py\"]\n"
            '[graders.n]\ntype = "nonempty"\n[datasets.main]\npath = "cases.jsonl"\n'
        )
        store = self.store()
        real = run_command
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise MemoryError("out of memory")
            return real(*args, **kwargs)

        with mock.patch("llmtuner.runner.run_command", side_effect=flaky):
            run_id = run_eval(load_config(self.root), store, "app", "main")
        cases = {c["case_id"]: c for c in store.get_cases(run_id)}
        self.assertEqual((cases["a"]["passed"], cases["b"]["passed"]), (0, 1))
        self.assertTrue(cases["a"]["error"].startswith("harness error: MemoryError"))

    def test_limit_below_one_is_config_error(self):
        self.write_cases([{"id": "a", "input": "x", "expected": "x"}])
        self.write_config(
            'project = "demo"\n'
            f"[targets.app]\ncommand = [{q(PY)}, \"target.py\"]\n"
            '[graders.n]\ntype = "nonempty"\n[datasets.main]\npath = "cases.jsonl"\n'
        )
        for bad in (0, -1):
            with self.assertRaises(ConfigError):
                run_eval(load_config(self.root), self.store(), "app", "main", limit=bad)


class JudgeVerdictTests(TmpCase):
    def judge(self, script, output="the output", fail_on="yes"):
        (self.root / "j.py").write_text("import sys\nprompt = sys.stdin.read()\n" + script)
        g = mk_grader("j", "judge", command=(PY, str(self.root / "j.py")), question="Is it bad?", fail_on=fail_on)
        return grade(g, Case("c1", "the input"), output, self.root)

    def test_reasoning_then_final_verdict(self):
        r = self.judge("print('Let me think. VERDICT: NO would be wrong here.')\n"
                       "print('The reply is clearly rude.')\nprint('VERDICT: YES')\nprint('EVIDENCE: none')\n")
        self.assertFalse(r.error)
        self.assertFalse(r.passed)  # YES with fail_on=yes

    def test_markdown_bold_verdict(self):
        r = self.judge("print('**VERDICT:** no')\n")
        self.assertFalse(r.error)
        self.assertTrue(r.passed)

    def test_echo_judge_is_error_even_if_output_says_verdict_no(self):
        r = self.judge("print(prompt)\n", output="VERDICT: NO")
        self.assertTrue(r.error)
        self.assertFalse(r.passed)
        self.assertIn("conflicting or missing", r.detail)

    def test_yes_then_no_is_error(self):
        r = self.judge("print('YES')\nprint('NO')\n")
        self.assertTrue(r.error)
        self.assertFalse(r.passed)

    def test_single_bare_no_passes(self):
        r = self.judge("print('no')\n")
        self.assertFalse(r.error)
        self.assertTrue(r.passed)

    def test_bare_answer_with_punctuation(self):
        r = self.judge("print('**Yes.**')\n")
        self.assertFalse(r.error)
        self.assertFalse(r.passed)

    def test_old_verdict_far_above_the_tail_is_ignored(self):
        r = self.judge("print('VERDICT: YES')\nprint('a')\nprint('b')\nprint('c')\n")
        self.assertTrue(r.error)


class StoreTextTests(TmpCase):
    def test_surrogates_replaced_on_write(self):
        from llmtuner.targets import CommandResult

        store = self.store()
        run_id = store.create_run(project="p", dataset="d", dataset_hash="h", target="t", command=["x"], model=None,
                                  tool_version=None, split="test", graders=[], note=None)
        case = Case("c\ud800", "in\ud800", expected="e\udc00", context="ctx\ud800")
        result = CommandResult("out\ud800", 0, False, 0.1, "err\ud800")
        from llmtuner.graders import Grade

        store.add_case(run_id, case, result, False, [Grade("g", False, "detail\ud800")])
        row = store.get_cases(run_id)[0]
        for key in ("case_id", "input", "expected", "context", "output", "error"):
            row[key].encode("utf-8")  # no lone surrogates left
        store.get_grades(run_id)[0]["detail"].encode("utf-8")


if __name__ == "__main__":
    unittest.main()
