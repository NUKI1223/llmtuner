"""CLI tests: call llmtuner.cli.main in-process against a temp store and a temp project of stub scripts.

Offline, and never touches the user's real store: every call passes --store.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from llmtuner.cli import main
from llmtuner.store import Store

PY = sys.executable

STUB = """\
import sys
text = sys.stdin.read().strip()
mode = sys.argv[1]
if mode == "bad1" and text == "c1":
    text = "wrong"
if mode == "bad6" and text in ("c1", "c2", "c3", "c4", "c5", "c6"):
    text = "wrong"
print(text)
"""


def strict_loads(text: str):
    def refuse(token):
        raise ValueError(f"not valid JSON: {token}")

    return json.loads(text, parse_constant=refuse)


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.project = root / "project"
        self.project.mkdir()
        self.store_dir = root / "store"
        (self.project / "stub.py").write_text(STUB)
        rows = lambda prefix: "\n".join(
            json.dumps({"id": f"{prefix}{i}", "input": f"{prefix}{i}", "expected": f"{prefix}{i}"}) for i in range(1, 9)
        ) + "\n"
        (self.project / "cases.jsonl").write_text(rows("c"))
        (self.project / "other.jsonl").write_text(rows("o"))
        (self.project / "bad.jsonl").write_bytes(b'{"id": "a", "input": "caf\xe9"}\n')
        targets = "".join(
            f'[targets.{name}]\ncommand = [{json.dumps(PY)}, "stub.py", "{name}"]\ntimeout = 10\n'
            for name in ("good", "bad1", "bad6")
        )
        (self.project / "llmtuner.toml").write_text(
            'project = "clidemo"\n' + targets
            + '[graders.same]\ntype = "exact"\n'
            + '[datasets.main]\npath = "cases.jsonl"\n'
            + '[datasets.other]\npath = "other.jsonl"\n'
            + '[datasets.bad]\npath = "bad.jsonl"\n'
        )

    def cli(self, *argv, expect=0, json_out=False):
        """Run the CLI; return (stdout, stderr). Checks the exit code and, for --json, strict JSON."""
        out, err = io.StringIO(), io.StringIO()
        args = [argv[0], "--config", str(self.project), "--store", str(self.store_dir), *argv[1:]]
        if json_out:
            args.append("--json")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
        self.assertEqual(code, expect, f"{argv}: stdout={out.getvalue()!r} stderr={err.getvalue()!r}")
        if json_out:
            self.parsed = strict_loads(out.getvalue())
        return out.getvalue(), err.getvalue()

    def run_target(self, target, dataset="main", *extra):
        self.cli("run", target, dataset, *extra, json_out=True)
        return self.parsed["run"]["id"]

    # --- paths ---------------------------------------------------------------

    def test_run_report_compare_runs_show(self):
        text, _ = self.cli("run", "good", "main", "--set-baseline")
        self.assertIn("Pass rate", text)
        self.assertIn("this run is the baseline", text)
        self.cli("run", "bad6", "main")
        text, _ = self.cli("report", "2")
        self.assertIn("2 of 8", text)
        text, _ = self.cli("compare", "1", "2")
        self.assertIn("bootstrap interval", text)
        self.assertIn("candidate is worse", text)
        text, _ = self.cli("runs")
        self.assertIn("good", text)
        self.assertIn("bad6", text)
        text, _ = self.cli("show", "2", "--failed")
        self.assertIn("=== c1  FAIL", text)
        text, _ = self.cli("info")
        self.assertIn("clidemo", text)

    def test_one_case_regression_is_no_measurable_difference(self):
        self.run_target("good")
        self.run_target("bad1")
        text, _ = self.cli("compare", "1", "2")
        self.assertIn("no measurable difference (exact McNemar p = 1.000, needs p < 0.05)", text)
        self.assertNotIn("candidate is worse", text)

    def test_label_and_report_show_agreement_input(self):
        self.run_target("good")
        text, _ = self.cli("label", "1", "--case", "c1", "--pass")
        self.assertIn("labelled c1 as pass", text)
        self.assertEqual(len(Store(self.store_dir).get_labels(1)), 1)

    def test_label_case_json(self):
        self.run_target("good")
        self.cli("label", "1", "--case", "c1", "--fail", "--note", "meh", json_out=True)
        self.assertEqual(self.parsed["case"], "c1")
        self.assertIs(self.parsed["passed"], False)

    # --- json is always strict -----------------------------------------------

    def test_json_output_is_strict_everywhere(self):
        self.run_target("good")
        self.run_target("bad6")
        self.run_target("good", "other")
        for argv in (["info"], ["report", "1"], ["report", "2"], ["compare", "1", "2"],
                     ["compare", "1", "3"],  # different datasets: zero shared cases
                     ["gate", "2", "--baseline", "1"], ["runs"], ["show", "2"], ["show", "1", "--case", "c1"]):
            expect = 1 if argv[0] == "gate" else 0
            self.cli(*argv, expect=expect, json_out=True)

    def test_compare_with_zero_shared_cases(self):
        self.run_target("good")
        self.run_target("good", "other")
        self.cli("compare", "1", "2", json_out=True)
        self.assertEqual(self.parsed["n"], 0)
        self.assertIsNone(self.parsed["baseline_rate"])
        self.assertEqual(self.parsed["verdict"], "no shared cases")
        text, _ = self.cli("compare", "1", "2")
        self.assertIn("no shared cases", text)

    def test_report_of_run_without_cases(self):
        with Store(self.store_dir) as store:
            run_id = store.create_run(project="clidemo", dataset="main", dataset_hash="h", target="good",
                                      command=["x"], model=None, tool_version=None, split="test", graders=[],
                                      note=None)
        self.cli("report", str(run_id), json_out=True)
        self.assertEqual(self.parsed["n"], 0)
        self.assertIsNone(self.parsed["pass_rate"]["point"])
        text, _ = self.cli("report", str(run_id))
        self.assertIn("recorded no cases", text)
        self.assertNotIn("Detection floor", text)

    # --- gate -----------------------------------------------------------------

    def test_gate_exit_codes(self):
        self.run_target("good")  # run 1
        text, _ = self.cli("gate", expect=1)  # no baseline yet
        self.assertIn("no baseline set", text)
        self.cli("baseline", "1")
        self.run_target("good")  # run 2, same result
        text, _ = self.cli("gate", "2")
        self.assertIn("PASS", text)
        self.run_target("bad6")  # run 3, drops 75 points
        text, _ = self.cli("gate", "3", expect=1)
        self.assertIn("FAIL", text)
        self.assertIn("got worse", text)
        self.cli("gate", "3", "--max-drop", "0.9")

    def test_gate_prints_comparison_warnings(self):
        self.run_target("good")
        self.cli("baseline", "1")
        self.run_target("bad6", "main", "--limit", "3")
        text, _ = self.cli("gate", "2", expect=1)
        self.assertIn("only 3 cases are in both runs", text)
        self.assertIn("note:", text)

    # --- errors ---------------------------------------------------------------

    def test_unknown_run_id_exits_2(self):
        for argv in (["show", "999"], ["report", "999"], ["gate", "999"],
                     ["label", "999", "--case", "c1", "--pass"], ["baseline", "999"]):
            with self.subTest(argv=argv):
                _, err = self.cli(*argv, expect=2)
                self.assertIn("no run with id 999", err)

    def test_zero_and_negative_counts_rejected(self):
        for argv in (["run", "good", "main", "--limit", "0"], ["run", "good", "main", "--jobs", "0"],
                     ["run", "good", "main", "--limit", "-2"], ["show", "1", "--limit", "0"],
                     ["runs", "--limit", "0"], ["label", "1", "--sample", "0"]):
            with self.subTest(argv=argv):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
                    main([argv[0], "--config", str(self.project), "--store", str(self.store_dir), *argv[1:]])
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("must be 1 or more", err.getvalue())

    def test_non_utf8_dataset_exits_2_without_traceback(self):
        _, err = self.cli("run", "good", "bad", expect=2)
        self.assertIn("error:", err)
        self.assertIn("UTF-8", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(Store(self.store_dir).list_runs(), [])


if __name__ == "__main__":
    unittest.main()
