"""Tests for the anti-overfitting work: holdout split, change log, changed-case detection, the
holdout rules for agents, and `llmtuner split`.

Offline and without `mcp`: everything the MCP tools do for these rules lives in library functions,
and those are tested directly. The test process itself may run under Claude Code, so CLAUDECODE can
be set in the real environment; every test that depends on it sets or clears it explicitly.
"""

import contextlib
import io
import json
import os
import sqlite3
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from llmtuner import changes, holdout, report
from llmtuner.changes import UNSET, ChangeError, edit_case, who_is_editing
from llmtuner.cli import main
from llmtuner.config import ConfigError, load_config
from llmtuner.dataset import DatasetError, load_cases, select_split, Case
from llmtuner.runner import run_eval
from llmtuner.store import Store

PY = sys.executable
STUB = "import sys\nprint(sys.stdin.read().strip())\n"  # echoes its input: a case passes when expected == input

ROWS = (
    [{"id": f"d{i}", "input": f"in-d{i}", "expected": f"in-d{i}", "split": "dev"} for i in (1, 2)]
    + [{"id": f"t{i}", "input": f"in-t{i}", "expected": f"in-t{i}", "split": "test"} for i in range(1, 9)]
    + [{"id": f"hold-{c}", "input": f"SECRET-input-{c}", "expected": f"SECRET-input-{c}", "split": "holdout"}
       for c in ("alpha", "beta", "gamma")]
)


def human_env():
    """An environment without any agent marker, whatever the test process runs under."""
    env = mock.patch.dict(os.environ)
    env.start()
    for name in changes.AGENT_ENV:
        os.environ.pop(name, None)
    return env


class Project(unittest.TestCase):
    """A temp project with an echo target, a dataset and a store."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = human_env()
        self.addCleanup(env.stop)
        (self.root / "stub.py").write_text(STUB)
        self.write_cases(ROWS)
        self.write_config()
        self.store = Store(self.root / "store")
        self.addCleanup(self.store.close)

    def write_config(self, extra=""):
        targets = "".join(f'[targets.{n}]\ncommand = [{json.dumps(PY)}, "stub.py"]\n' for n in ("a", "b"))
        (self.root / "llmtuner.toml").write_text(
            'project = "overfit"\n' + targets + '[graders.same]\ntype = "exact"\n'
            + '[datasets.main]\npath = "cases.jsonl"\n' + extra)

    def write_cases(self, rows, newline="\n"):
        (self.root / "cases.jsonl").write_bytes(
            "".join(json.dumps(r, ensure_ascii=False) + newline for r in rows).encode("utf-8"))

    @property
    def config(self):
        return load_config(self.root)

    @property
    def data_path(self) -> Path:
        return self.root / "cases.jsonl"

    @property
    def log(self) -> list[dict]:
        return changes.read_log(self.config.dataset("main").changes)

    def go(self, target="a", split="test", **kw):
        return run_eval(self.config, self.store, target, "main", split=split, **kw)

    def edit(self, case_id, by="human", via="cli", reason="the owner corrected it", **fields):
        if by == "agent" and "basis" not in fields:
            fields["basis"] = {"kind": "rule", "ref": "rules/labels.md"}
        return edit_case(self.config, "main", case_id, reason=reason, by=by, via=via,
                         signal="CLAUDECODE" if by == "agent" else None, **fields)

    def hand_edit(self, case_id, **fields):
        """Change a case in the file without the tool: no log entry."""
        rows = [json.loads(l) for l in self.data_path.read_text().splitlines() if l.strip()]
        for row in rows:
            if row["id"] == case_id:
                row.update(fields)
        self.write_cases(rows)

    def changed(self, base, cand, log=True):
        return report.case_changes(self.store, base, cand, self.log if log else None)

    def by_id(self, changes_):
        return {c["case_id"]: c for c in changes_}


class SplitTests(unittest.TestCase):
    CASES = [Case("a", "x", split="dev"), Case("b", "x", split="test"), Case("c", "x", split="holdout")]

    def test_all_is_dev_and_test_only(self):
        self.assertEqual([c.id for c in select_split(self.CASES, "all")], ["a", "b"])

    def test_holdout_must_be_named(self):
        self.assertEqual([c.id for c in select_split(self.CASES, "holdout")], ["c"])

    def test_all_with_only_holdout_cases_is_an_error(self):
        with self.assertRaises(DatasetError):
            select_split([Case("c", "x", split="holdout")], "all")

    def test_bad_and_empty_splits(self):
        with self.assertRaises(DatasetError):
            select_split(self.CASES, "train")
        with self.assertRaises(DatasetError):
            select_split(self.CASES[:2], "holdout")

    def test_loader_accepts_holdout_and_names_all_splits_in_its_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text('{"id": "a", "input": "x", "split": "holdout"}\n')
            self.assertEqual(load_cases(path)[0][0].split, "holdout")
            path.write_text('{"id": "a", "input": "x", "split": "train"}\n')
            with self.assertRaises(DatasetError) as caught:
                load_cases(path)
            self.assertIn("holdout", str(caught.exception))

    def test_run_all_leaves_holdout_out_and_holdout_runs_when_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "stub.py").write_text(STUB)
            (project / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
            (project / "llmtuner.toml").write_text(
                f'project = "p"\n[targets.a]\ncommand = [{json.dumps(PY)}, "stub.py"]\n'
                '[graders.same]\ntype = "exact"\n[datasets.main]\npath = "cases.jsonl"\n')
            cfg = load_config(project)
            with Store(project / "store") as store:
                every = run_eval(cfg, store, "a", "main", split="all")
                self.assertEqual({c["case_id"] for c in store.get_cases(every)}, {r["id"] for r in ROWS[:10]})
                held = run_eval(cfg, store, "a", "main", split="holdout")
                self.assertEqual(len(store.get_cases(held)), 3)
                self.assertEqual({c["split"] for c in store.get_cases(held)}, {"holdout"})


class WhoIsEditingTests(unittest.TestCase):
    def test_claudecode_is_an_agent_and_human_flag_cannot_override(self):
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}):
            os.environ.pop("AI_AGENT", None)
            self.assertEqual(who_is_editing(None, "cli"), ("agent", "CLAUDECODE"))
            self.assertEqual(who_is_editing("human", "cli"), ("agent", "CLAUDECODE"))

    def test_ai_agent_variable_and_order(self):
        with mock.patch.dict(os.environ, {"AI_AGENT": "codex"}):
            os.environ.pop("CLAUDECODE", None)
            self.assertEqual(who_is_editing("human", "cli"), ("agent", "AI_AGENT"))
            os.environ["CLAUDECODE"] = "1"
            self.assertEqual(who_is_editing(None, "cli"), ("agent", "CLAUDECODE"))

    def test_empty_variable_does_not_count(self):
        with mock.patch.dict(os.environ, {"CLAUDECODE": "", "AI_AGENT": ""}):
            self.assertEqual(who_is_editing(None, "cli"), ("human", None))

    def test_flag_mcp_and_plain_human(self):
        env = human_env()
        self.addCleanup(env.stop)
        self.assertEqual(who_is_editing("agent", "cli"), ("agent", "flag"))
        self.assertEqual(who_is_editing(None, "mcp"), ("agent", "mcp"))
        self.assertEqual(who_is_editing("human", "mcp"), ("agent", "mcp"))
        self.assertEqual(who_is_editing(None, "cli"), ("human", None))
        self.assertEqual(who_is_editing("human", "cli"), ("human", None))


class ConfigTests(Project):
    def test_default_and_overridden_log_path(self):
        self.assertEqual(self.config.dataset("main").changes, self.root / "cases.changes.jsonl")
        self.assertEqual(self.config.describe()["datasets"]["main"]["changes"], str(self.root / "cases.changes.jsonl"))
        self.write_config('changes = "artifacts/log.jsonl"\n')
        self.assertEqual(self.config.dataset("main").changes, (self.root / "artifacts/log.jsonl").resolve())

    def test_bad_changes_values(self):
        for line in ('changes = 5\n', 'changes = ""\n', 'changes = "cases.jsonl"\n'):
            self.write_config(line)
            with self.assertRaises(ConfigError, msg=line):
                load_config(self.root)


class EditCaseTests(Project):
    def test_rewrites_only_that_line_and_keeps_crlf(self):
        lines = [json.dumps(r, ensure_ascii=False) for r in ROWS]
        lines[3] = '{"id":"t2",  "input": "caf\u00e9", "expected": "caf\u00e9", "split": "test"}'
        raw = ("\r\n".join(lines) + "\r\n").encode("utf-8")
        self.data_path.write_bytes(raw)
        self.edit("t2", expected="changed")
        new = self.data_path.read_bytes()
        old_lines, new_lines = raw.split(b"\n"), new.split(b"\n")
        self.assertEqual(len(old_lines), len(new_lines))
        for number, (a, b) in enumerate(zip(old_lines, new_lines)):
            if number == 3:
                self.assertTrue(b.endswith(b"\r"))
                self.assertEqual(json.loads(b), {"id": "t2", "input": "caf\u00e9", "expected": "changed", "split": "test"})
                self.assertIn("caf\u00e9".encode(), b)  # not escaped
                self.assertEqual(list(json.loads(b)), ["id", "input", "expected", "split"])
            else:
                self.assertEqual(a, b)

    def test_file_without_final_newline_and_mode_kept(self):
        self.data_path.write_bytes(self.data_path.read_bytes().rstrip(b"\n"))
        os.chmod(self.data_path, 0o640)
        self.edit("hold-gamma", expected="x")
        self.assertFalse(self.data_path.read_bytes().endswith(b"\n"))
        self.assertEqual(stat.S_IMODE(self.data_path.stat().st_mode), 0o640)
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith(".")), [])  # no temp files left

    def test_log_entry_fields(self):
        entry = self.edit("t1", expected={"b": 1, "a": 2}, context="ctx", now=None, reason="  because  ",
                          by="agent", via="cli", basis={"kind": "owner", "quote": " refunds are billing "})
        self.assertEqual(self.log, [entry])
        self.assertEqual(set(entry), {"ts", "at", "case_id", "action", "by", "via", "agent_signal", "reason",
                                      "basis", "before", "after"})
        self.assertEqual((entry["case_id"], entry["action"], entry["by"], entry["via"], entry["agent_signal"]),
                         ("t1", "edit", "agent", "cli", "CLAUDECODE"))
        self.assertEqual(entry["reason"], "because")
        self.assertEqual(entry["basis"], {"kind": "owner", "quote": "refunds are billing"})
        self.assertEqual(entry["before"], {"expected": "in-t1", "context": None})
        self.assertEqual(entry["after"], {"expected": {"b": 1, "a": 2}, "context": "ctx"})
        self.assertRegex(entry["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertLess(abs(entry["ts"] - time.time()), 30)

    def test_split_only_edit_is_a_split_action_and_missing_split_reads_as_test(self):
        self.write_cases([{"id": "x", "input": "i", "expected": "i"}])  # no split key: it is "test"
        entry = self.edit("x", split="dev")
        self.assertEqual((entry["action"], entry["before"], entry["after"]), ("split", {"split": "test"}, {"split": "dev"}))
        with self.assertRaises(ChangeError):
            self.edit("x", split="dev")

    def test_no_op_rejected_even_when_only_key_order_differs(self):
        self.write_cases([{"id": "x", "input": {"p": 1, "q": 2}, "expected": {"a": 1, "b": [1, 2]}, "split": "dev"}])
        before = self.data_path.read_bytes()
        for fields in ({"expected": {"b": [1, 2], "a": 1}}, {"input": {"q": 2, "p": 1}}, {"split": "dev"}):
            with self.assertRaisesRegex(ChangeError, "nothing to change"):
                self.edit("x", **fields)
        self.assertEqual(self.data_path.read_bytes(), before)
        self.assertEqual(self.log, [])

    def test_true_and_one_are_different_values(self):
        self.write_cases([{"id": "x", "input": "i", "expected": True, "split": "dev"}])
        self.assertEqual(self.edit("x", expected=1)["before"], {"expected": True})

    def test_setting_a_missing_expected_to_none_is_a_no_op(self):
        self.write_cases([{"id": "x", "input": "i", "split": "dev"}])
        with self.assertRaises(ChangeError):
            self.edit("x", expected=None)

    def test_unknown_case_bad_split_bad_values(self):
        with self.assertRaisesRegex(ChangeError, "no case"):
            self.edit("nope", expected="x")
        with self.assertRaises(ChangeError):
            self.edit("d1", split="train")
        with self.assertRaises(ChangeError):
            self.edit("d1", context=5)
        with self.assertRaises(ChangeError):
            self.edit("d1", input=None)

    def test_reason_required_for_test_and_holdout_optional_for_dev(self):
        for case_id in ("t1", "hold-alpha"):
            for reason in (None, "", "   "):
                with self.assertRaisesRegex(ChangeError, "reason"):
                    self.edit(case_id, reason=reason, expected="x")
        self.assertEqual(self.edit("d1", reason=None, expected="x")["reason"], None)
        # Moving a dev case into test is an edit of a test case after the edit.
        with self.assertRaisesRegex(ChangeError, "reason"):
            self.edit("d2", reason=None, split="test")
        # ... and so is moving a test case out of it.
        with self.assertRaisesRegex(ChangeError, "reason"):
            self.edit("t2", reason=None, split="dev")
        self.assertEqual(self.edit("t3", reason="moved it", split="dev")["after"], {"split": "dev"})

    def test_agent_needs_a_basis_for_test_and_holdout(self):
        bad = [None, {}, {"kind": "rule"}, {"kind": "rule", "ref": "  "}, {"kind": "owner", "quote": ""},
               {"kind": "model", "ref": "its answer"}, "rule"]
        for basis in bad:
            with self.assertRaises(ChangeError, msg=repr(basis)):
                edit_case(self.config, "main", "t1", expected="x", reason="why", basis=basis, by="agent",
                          via="cli", signal="flag")
        self.assertEqual(self.log, [])
        for basis in ({"kind": "rule", "ref": "rules/a.md"}, {"kind": "owner", "quote": "he said so"}):
            edit_case(self.config, "main", "t1", expected=str(basis), reason="why", basis=basis, by="agent",
                      via="cli", signal="flag")
        # Dev-only: no reason, no basis.
        edit_case(self.config, "main", "d1", expected="y", by="agent", via="cli", signal="flag")
        # A person needs no basis, but one that is given is kept.
        self.assertEqual(self.edit("t2", expected="z", basis={"kind": "rule", "ref": "r"})["basis"],
                         {"kind": "rule", "ref": "r"})

    def test_mcp_cannot_touch_holdout_before_or_after(self):
        before = self.data_path.read_bytes()
        with self.assertRaisesRegex(ChangeError, "holdout"):
            self.edit("hold-alpha", via="mcp", by="agent", expected="in-hold")
        with self.assertRaisesRegex(ChangeError, "holdout"):  # the true value: no "nothing to change" oracle
            self.edit("hold-alpha", via="mcp", by="agent", expected="SECRET-input-alpha")
        with self.assertRaises(ChangeError) as missing:  # same answer for an id that does not exist
            self.edit("nope", via="mcp", by="agent", expected="x")
        self.assertIn("holdout", str(missing.exception))
        with self.assertRaisesRegex(ChangeError, "holdout"):
            self.edit("hold-alpha", via="mcp", by="agent", expected="x")
        with self.assertRaisesRegex(ChangeError, "holdout"):
            self.edit("t1", via="mcp", by="agent", split="holdout")
        with self.assertRaisesRegex(ChangeError, "holdout"):
            changes.agent_edit(self.config, "main", "hold-beta", reason="r", basis_kind="rule", basis_text="x",
                               expected_json='"y"')
        self.assertEqual(self.data_path.read_bytes(), before)
        self.assertEqual(self.log, [])

    def test_agent_edit_through_the_mcp_wrapper(self):
        entry = changes.agent_edit(self.config, "main", "t1", reason="rule says so", basis_kind="rule",
                                   basis_text="rules/x.md", expected_json='{"k": [1, 2]}', context="c")
        self.assertEqual((entry["by"], entry["via"], entry["agent_signal"]), ("agent", "mcp", "mcp"))
        self.assertEqual(json.loads(self.data_path.read_text().splitlines()[2])["expected"], {"k": [1, 2]})
        for kwargs in ({"expected_json": "{not json"}, {"input_json": ""}):
            with self.assertRaisesRegex(ChangeError, "JSON"):
                changes.agent_edit(self.config, "main", "t2", reason="r", basis_kind="rule", basis_text="x", **kwargs)
        with self.assertRaisesRegex(ChangeError, "basis_kind"):
            changes.agent_edit(self.config, "main", "t2", reason="r", basis_kind="model", basis_text="x", context="c")

    def test_failed_log_write_puts_the_dataset_back(self):
        before = self.data_path.read_bytes()
        with mock.patch.object(changes, "append_entry", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(ChangeError, "left as it was"):
                self.edit("t1", expected="x")
        self.assertEqual(self.data_path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith(".")), [])

    def test_an_unreadable_dataset_is_not_touched(self):
        self.data_path.write_text('{"id": "a", "input": "x"}\nnot json\n')
        before = self.data_path.read_bytes()
        with self.assertRaises(DatasetError):
            self.edit("a", expected="x")
        self.assertEqual(self.data_path.read_bytes(), before)

    def test_log_reading_skips_malformed_lines_and_counts_them(self):
        path = self.config.dataset("main").changes
        self.assertEqual(changes.read_log(path), [])
        self.edit("d1", expected="x")
        with open(path, "a") as handle:
            handle.write("garbage\n[1]\n{\"case_id\": \"x\"}\n")  # not JSON, not an object, no ts
        entries, skipped = changes.read_log_with_skipped(path)
        self.assertEqual((len(entries), skipped), (1, 3))
        path.write_bytes(path.read_bytes().rstrip(b"\n"))  # a hand-edited log without its final newline
        self.edit("d2", expected="y")
        self.assertEqual(len(changes.read_log(path)), 2)


class StoreTests(Project):
    def test_run_records_origin_limit_and_case_splits(self):
        run_id = self.go(split="all", limit=3, origin="mcp")
        run = self.store.get_run(run_id)
        self.assertEqual((run["origin"], run["case_limit"]), ("mcp", 3))
        self.assertEqual([c["split"] for c in self.store.get_cases(run_id)], ["dev", "dev", "test"])
        full = self.store.get_run(self.go())
        self.assertEqual((full["origin"], full["case_limit"]), ("cli", None))

    def test_old_store_is_migrated_and_old_rows_have_null(self):
        home = self.root / "old"
        home.mkdir()
        db = sqlite3.connect(home / "store.db")
        db.executescript("""
            CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL, dataset TEXT NOT NULL,
              dataset_hash TEXT NOT NULL, target TEXT NOT NULL, command TEXT NOT NULL, model TEXT, tool_version TEXT,
              split TEXT NOT NULL, graders TEXT NOT NULL, note TEXT, started_at REAL NOT NULL, finished_at REAL);
            CREATE TABLE cases (run_id INTEGER NOT NULL, case_id TEXT NOT NULL, input TEXT NOT NULL, expected TEXT,
              context TEXT, output TEXT NOT NULL, exit_code INTEGER, timed_out INTEGER NOT NULL,
              duration_s REAL NOT NULL, error TEXT, passed INTEGER NOT NULL, PRIMARY KEY (run_id, case_id));
            INSERT INTO runs VALUES (1,'old','main','h','t','["x"]',NULL,NULL,'test','[]',NULL,1.0,2.0);
            INSERT INTO cases VALUES (1,'a','in',NULL,NULL,'out',0,0,0.1,NULL,1);
        """)
        db.commit()
        db.close()
        with Store(home) as store:
            run = store.get_run(1)
            self.assertEqual((run["origin"], run["case_limit"]), (None, None))
            self.assertIsNone(store.get_cases(1)[0]["split"])
            new = store.create_run(project="old", dataset="main", dataset_hash="h", target="t", command=["x"],
                                   model=None, tool_version=None, split="test", graders=[], note=None,
                                   origin="cli", case_limit=4)
            self.assertEqual(store.get_run(new)["case_limit"], 4)
        with Store(home):  # opening it again changes nothing
            pass


class SharedFileTests(Project):
    def test_two_datasets_on_one_file_must_share_their_change_log(self):
        self.write_config('[datasets.other]\npath = "cases.jsonl"\nchanges = "other.changes.jsonl"\n')
        with self.assertRaisesRegex(ConfigError, "different change log"):
            load_config(self.root)
        self.write_config('[datasets.other]\npath = "cases.jsonl"\n')
        self.assertEqual(self.config.dataset("other").changes, self.config.dataset("main").changes)

    def test_a_change_log_cannot_be_another_datasets_file(self):
        (self.root / "other.jsonl").write_text('{"id": "o1", "input": "i", "expected": "i"}\n')
        self.write_config('[datasets.other]\npath = "other.jsonl"\nchanges = "cases.jsonl"\n')
        with self.assertRaisesRegex(ConfigError, "a change log must be a file of its own"):
            load_config(self.root)


class SharedFileCliTests(Project):
    def test_runs_under_another_name_for_the_same_file_count_as_seen(self):
        self.write_config('[datasets.other]\npath = "cases.jsonl"\n')
        store = ["--config", str(self.root), "--store", str(self.root / "cli-store")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "a", "other", "--split", "test", *store]), 0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["split", "main", "--seed", "1", "--json", *store]), 0)
        proposal = json.loads(out.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(main(["case", "edit", "main", "t1", "--split", "holdout", "--reason", "x", *store]), 0)
        tests = {r["id"] for r in ROWS if r.get("split", "test") == "test"}
        self.assertFalse(tests & {c["case_id"] for c in proposal["cases"] if c["proposed"] == "holdout"})
        self.assertIn("already in a run outside holdout", err.getvalue())


class ChangeDetectionTests(Project):
    def test_each_kind_of_field_change_is_found(self):
        base = self.go()
        self.edit("t1", expected="other")
        self.edit("t2", input="other input")
        self.edit("t3", context="new context")
        self.edit("t4", split="dev")
        cand = self.go()
        found = self.by_id(self.changed(base, cand))
        self.assertEqual(found["t1"]["fields"], ["expected"])
        self.assertEqual(found["t2"]["fields"], ["input"])
        self.assertEqual(found["t3"]["fields"], ["context"])
        self.assertEqual(found["t4"]["kind"], "missing")  # moved out of the split the candidate ran
        self.assertEqual((found["t4"]["split_before"], found["t4"]["split_after"]), ("test", "dev"))
        self.assertEqual({c["status"] for c in found.values()}, {"human"})
        self.assertEqual(set(found), {"t1", "t2", "t3", "t4"})

    def test_split_change_inside_a_run_that_holds_both_is_a_changed_case(self):
        base = self.go(split="all")
        self.edit("d1", split="test")
        cand = self.go(split="all")
        found = self.by_id(self.changed(base, cand))
        self.assertEqual((found["d1"]["kind"], found["d1"]["fields"]), ("changed", ["split"]))

    def test_json_key_order_is_not_a_change_but_a_value_is(self):
        self.write_cases([{"id": "x", "input": {"p": 1, "q": 2}, "expected": {"a": 1, "b": 2}},
                          {"id": "y", "input": "i", "expected": "i"}])
        base = self.go()
        self.write_cases([{"id": "x", "input": {"q": 2, "p": 1}, "expected": {"b": 2, "a": 1}},
                          {"id": "y", "input": "i", "expected": "j"}])
        cand = self.go()
        self.assertEqual(list(self.by_id(self.changed(base, cand))), ["y"])

    def test_missing_only_counts_when_the_candidate_ran_without_a_limit(self):
        base = self.go()
        limited = self.go(limit=3)
        self.assertEqual(self.changed(base, limited), [])
        self.write_cases([r for r in ROWS if r["id"] != "t5"])
        full = self.go()
        found = self.changed(base, full)
        self.assertEqual([(c["case_id"], c["kind"]) for c in found], [("t5", "missing")])
        self.assertEqual(self.changed(base, self.go(limit=2)), [])
        new_case = ROWS + [{"id": "t99", "input": "n", "expected": "n"}]
        self.write_cases(new_case)
        self.assertEqual(self.changed(base, self.go()), [])  # a case only in the candidate is new

    def test_dev_cases_absent_from_a_test_run_are_not_missing(self):
        base = self.go(split="all")
        self.assertEqual(self.changed(base, self.go(split="test")), [])

    def test_status_human_agent_unrecorded(self):
        base = self.go()
        self.edit("t1", expected="a")
        self.edit("t2", expected="b", by="agent")
        self.edit("t3", expected="c")
        self.edit("t3", expected="d", by="agent")  # one agent entry among human ones
        self.hand_edit("t4", expected="e")
        cand = self.go()
        found = self.by_id(self.changed(base, cand))
        self.assertEqual({k: v["status"] for k, v in found.items()},
                         {"t1": "human", "t2": "agent", "t3": "agent", "t4": "unrecorded"})
        self.assertEqual(len(found["t3"]["entries"]), 2)
        self.assertEqual(found["t4"]["entries"], [])

    def test_log_entries_outside_the_window_do_not_count(self):
        self.edit("t1", expected="x")  # before the baseline run
        base = self.go()
        self.hand_edit("t1", expected="y")
        cand = self.go()
        self.assertEqual(self.by_id(self.changed(base, cand))["t1"]["status"], "unrecorded")

    def test_old_baseline_without_case_splits_is_before_log(self):
        base = self.go()
        self.store._db.execute("UPDATE cases SET split = NULL WHERE run_id = ?", (base,))
        self.store._db.commit()
        self.hand_edit("t1", expected="x")
        self.hand_edit("t2", split="dev")
        cand = self.go()
        found = self.by_id(self.changed(base, cand))
        self.assertEqual({k: v["status"] for k, v in found.items()}, {"t1": "before_log", "t2": "before_log"})
        self.assertEqual(found["t2"]["kind"], "missing")  # a split change cannot be seen without a stored split
        self.assertEqual(found["t2"]["fields"], [])

    def test_no_log_makes_every_change_unrecorded_and_says_so(self):
        base = self.go()
        self.edit("t1", expected="x")
        cand = self.go()
        self.assertEqual(self.changed(base, cand, log=False)[0]["status"], "unrecorded")
        cmp = report.compare_runs(self.store, base, cand)
        self.assertFalse(cmp["changes_log_available"])
        self.assertTrue(any("log was not available" in w for w in cmp["warnings"]))
        self.assertTrue(report.compare_runs(self.store, base, cand, self.log)["changes_log_available"])

    def test_changed_cases_are_left_out_of_the_statistics(self):
        base = self.go()
        self.edit("t1", expected="no longer matches")  # would flip to failing
        cand = self.go()
        cmp = report.compare_runs(self.store, base, cand, self.log)
        self.assertEqual(cmp["n"], 7)
        self.assertEqual(cmp["regressions"], [])
        self.assertEqual([c["case_id"] for c in cmp["changed_cases"]], ["t1"])
        self.assertIn("1 cases changed since the baseline run and were left out of the comparison", cmp["warnings"])
        self.assertFalse(any("dataset file changed" in w for w in cmp["warnings"]))
        text = report.format_compare(cmp)
        self.assertIn("t1", text)
        self.assertIn("changed expected", text)

    def test_an_edit_elsewhere_in_the_file_is_not_a_warning(self):
        base = self.go()
        self.edit("hold-alpha", expected="x")
        cand = self.go()
        cmp = report.compare_runs(self.store, base, cand, self.log)
        self.assertEqual((cmp["changed_cases"], cmp["warnings"]), ([], []))

    def test_trial_groups_use_their_first_run(self):
        base = self.go(trials=2)
        self.edit("t1", expected="x")
        cand = self.go(trials=2)
        second_trial = self.store.group_runs(cand)[1]["id"]
        for left, right in ((base, cand), (base, second_trial)):
            found = self.changed(left, right)
            self.assertEqual([c["case_id"] for c in found], ["t1"])
        cmp = report.compare_runs(self.store, base, cand, self.log)
        self.assertEqual(cmp["n"], 7)
        self.assertNotIn("t1", [c["case_id"] for c in cmp["per_case"]])

    def test_different_datasets_have_no_changes(self):
        base = self.go()
        other = self.store.create_run(project="overfit", dataset="other", dataset_hash="h", target="a", command=["x"],
                                      model=None, tool_version=None, split="test", graders=[], note=None)
        self.assertEqual(self.changed(base, other), [])


class GateTests(Project):
    def gate(self, base, cand, log=True):
        return report.gate(self.store, cand, base, 0.03, self.log if log else None)

    def setUp(self):
        super().setUp()
        self.base = self.go()
        self.store.set_baseline(self.base)

    def test_unchanged_data_passes(self):
        self.assertTrue(self.gate(self.base, self.go())["ok"])

    def test_agent_edit_of_a_test_case_fails_with_advice(self):
        self.edit("t1", expected="x", by="agent")
        cand = self.go()
        result = self.gate(self.base, cand)
        self.assertFalse(result["ok"])
        self.assertIn("t1", result["reason"])
        self.assertIn(f"llmtuner baseline {cand}", result["reason"])
        self.assertEqual(result["blocking_changes"], ["t1"])
        self.assertIn("changed by an agent", report.format_gate(result))
        self.assertIn("FAIL", report.format_gate(result))

    def test_unrecorded_edit_of_a_test_case_fails(self):
        self.hand_edit("t2", expected="x")
        self.assertFalse(self.gate(self.base, self.go())["ok"])

    def test_missing_unrecorded_test_case_fails(self):
        self.write_cases([r for r in ROWS if r["id"] != "t2"])
        result = self.gate(self.base, self.go())
        self.assertFalse(result["ok"])
        self.assertEqual(result["blocking_changes"], ["t2"])

    def test_a_logged_edit_does_not_cover_a_later_hand_deletion_or_move(self):
        self.edit("t2", expected="WRONG2")
        for rows in ([r for r in ROWS if r["id"] != "t2"],
                     [dict(r, split="dev") if r["id"] == "t2" else r for r in ROWS]):
            with self.subTest(rows=[r.get("split") for r in rows]):
                self.write_cases(rows)
                result = self.gate(self.base, self.go())
                self.assertFalse(result["ok"])
                self.assertEqual(result["blocking_changes"], ["t2"])

    def test_a_logged_move_out_of_the_split_by_a_person_passes(self):
        self.edit("t2", split="dev")
        result = self.gate(self.base, self.go())
        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(self.by_id(result["comparison"]["changed_cases"])["t2"]["status"], "human")

    def test_human_edit_with_a_reason_passes_and_is_listed(self):
        self.edit("t1", expected="x")
        result = self.gate(self.base, self.go())
        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual([c["case_id"] for c in result["comparison"]["changed_cases"]], ["t1"])
        self.assertIn("t1", report.format_gate(result))

    def test_dev_changes_never_fail(self):
        base = self.go(split="all")
        self.hand_edit("d1", expected="x")
        self.edit("d2", expected="y", by="agent")
        result = report.gate(self.store, self.go(split="all"), base, 0.03, self.log)
        self.assertTrue(result["ok"], result["reason"])

    def test_changes_to_a_baseline_before_the_log_do_not_fail(self):
        self.store._db.execute("UPDATE cases SET split = NULL WHERE run_id = ?", (self.base,))
        self.store._db.commit()
        self.hand_edit("t1", expected="x")
        self.assertTrue(self.gate(self.base, self.go())["ok"])

    def test_a_new_baseline_clears_it(self):
        self.edit("t1", expected="x", by="agent")
        cand = self.go()
        self.assertFalse(self.gate(self.base, cand)["ok"])
        self.store.set_baseline(cand)
        self.assertTrue(self.gate(cand, self.go())["ok"])

    def test_gate_through_the_stored_baseline_and_a_trial_group(self):
        self.hand_edit("t3", expected="x")
        cand = self.go(trials=2)
        self.assertFalse(report.gate(self.store, cand, None, 0.03, self.log)["ok"])

    def test_without_a_log_a_changed_test_case_fails(self):
        self.edit("t1", expected="x")
        self.assertFalse(self.gate(self.base, self.go(), log=False)["ok"])


class ReportTests(Project):
    def test_report_shows_the_cases_changed_since_the_previous_run(self):
        first = self.go()
        self.assertIsNone(report.build_report(self.store, first, self.log)["changed_since_previous"])
        self.edit("t1", expected="x", by="agent")
        second = self.go()
        rep = report.build_report(self.store, second, self.log)
        self.assertEqual(rep["previous_run"], first)
        self.assertEqual([c["case_id"] for c in rep["changed_since_previous"]], ["t1"])
        text = report.format_report(rep)
        self.assertIn(f"CHANGED SINCE RUN {first}", text)
        self.assertIn("t1", text.split("1. Pass rate")[0])  # above the numbers
        self.assertIn("by an agent", text)

    def test_previous_run_is_of_the_same_split_and_outside_the_own_group(self):
        first = self.go()
        self.go(split="dev")
        self.edit("t1", expected="x")
        group = self.go(trials=2)
        for member in self.store.group_runs(group):
            self.assertEqual(report.build_report(self.store, member["id"], self.log)["previous_run"], first)

    def test_unchanged_report_has_an_empty_list_and_no_block(self):
        self.go()
        rep = report.build_report(self.store, self.go(), self.log)
        self.assertEqual(rep["changed_since_previous"], [])
        self.assertNotIn("CHANGED", report.format_report(rep))


class RankTests(Project):
    def test_an_edit_to_a_holdout_case_does_not_split_the_ranking(self):
        a = self.go("a")
        self.edit("hold-alpha", expected="x")
        b = self.go("b")
        result = report.rank(self.store, "overfit", "main")
        self.assertEqual(sorted(r["run"] for r in result["rows"]), sorted([a, b]))
        self.assertEqual(result["skipped"], [])

    def test_an_edit_to_a_test_case_still_does(self):
        a = self.go("a")
        self.edit("t1", expected="x")
        b = self.go("b")
        result = report.rank(self.store, "overfit", "main")
        self.assertEqual([r["run"] for r in result["rows"]], [b])
        self.assertEqual(result["skipped"], [{"run": a, "target": "a", "reason": "ran on a different version of the cases"}])

    def test_limit_runs_are_still_a_different_set_of_cases(self):
        self.go("a")
        limited = self.go("b", limit=3)
        result = report.rank(self.store, "overfit", "main")
        self.assertEqual(result["skipped"][0]["run"], limited)
        self.assertIn("--limit", result["skipped"][0]["reason"])


class HoldoutRulesTests(Project):
    def fake_run(self, started, origin="mcp", split="holdout", dataset="main", finished=False, **kw):
        run_id = self.store.create_run(project="overfit", dataset=dataset, dataset_hash="h", target="a",
                                       command=["x"], model=None, tool_version=None, split=split, graders=[],
                                       note=None, origin=origin, **kw)
        self.store._db.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started, run_id))
        if finished:
            self.store.finish_run(run_id)
        self.store._db.commit()
        return run_id

    def check(self, now, split="holdout", limit=None, dataset="main"):
        holdout.check_agent_run(self.store, "overfit", dataset, split, limit, now=now)

    def test_limit_is_rejected_for_holdout_only(self):
        with self.assertRaisesRegex(holdout.HoldoutError, "limit"):
            self.check(1000.0, limit=5)
        self.check(1000.0, split="test", limit=5)

    def test_rate_limit_is_a_rolling_24_hours_and_counts_unfinished_runs(self):
        self.check(10_000.0)
        self.fake_run(started=10_000.0)  # never finished: still used the attempt
        with self.assertRaisesRegex(holdout.HoldoutError, "next attempt is allowed at 1970-01-02 02:46 UTC"):
            self.check(10_000.0 + 3600)
        self.check(10_000.0 + 86400 + 1)  # rolled out of the window
        self.check(10_000.0 + 3600, split="test")  # other splits are not limited

    def test_a_group_of_trials_counts_once_by_its_first_run(self):
        first = self.fake_run(10_000.0, trials_planned=3, trial=1)
        self.store._db.execute("UPDATE runs SET trial_group = id WHERE id = ?", (first,))
        for n in (2, 3):
            self.fake_run(10_000.0 + n * 40_000, trial_group=first, trial=n, trials_planned=3)
        self.assertEqual(self.store.holdout_attempts("overfit", "main", 0), [10_000.0])
        self.check(10_000.0 + 86400 + 1)  # later trials do not push the window out

    def test_other_datasets_origins_and_projects_are_not_limited(self):
        self.fake_run(10_000.0, dataset="other")
        self.fake_run(10_000.0, origin="cli")
        self.fake_run(10_000.0, origin=None)
        self.fake_run(10_000.0, split="test")
        self.check(10_000.0 + 60)
        self.assertEqual(self.store.holdout_attempts("another-project", "main", 0), [])
        self.fake_run(10_000.0)
        with self.assertRaises(holdout.HoldoutError):
            self.check(10_000.0 + 60)
        self.check(10_000.0 + 60, dataset="other-dataset")


class RedactionTests(Project):
    def setUp(self):
        super().setUp()
        self.hold = self.go(split="holdout", origin="mcp")
        # A failing holdout case, so there is a failure to hide.
        self.hand_edit("hold-beta", expected="not what the target says")
        self.hold2 = self.go(split="holdout", origin="mcp", trials=2)

    def assert_no_leak(self, data):
        text = json.dumps(report.json_safe(data), default=str)
        for needle in ("hold-alpha", "hold-beta", "hold-gamma", "SECRET", "not what the target"):
            self.assertNotIn(needle, text)

    def test_holdout_report_has_totals_only(self):
        rep = holdout.agent_report(self.store, self.hold)
        self.assert_no_leak(rep)
        self.assertEqual((rep["n"], rep["n_passed"]), (3, 3))
        self.assertIn("point", rep["pass_rate"])
        for gone in ("failed_cases", "graders", "grader_checks", "unchecked_judges"):
            self.assertNotIn(gone, rep)
        grouped = holdout.agent_report(self.store, self.hold2, changes.read_log(self.config.dataset("main").changes))
        self.assert_no_leak(grouped)
        self.assertEqual(grouped["trials"]["passed_all"]["n_passed"], 2)
        self.assertNotIn("cases", grouped["trials"])
        self.assertNotIn("unstable_cases", grouped["trials"])
        self.assertEqual(grouped["n_changed_since_previous"], 1)  # a count, not an id

    def test_the_owner_still_sees_everything(self):
        rep = report.build_report(self.store, self.hold2, self.log_entries())
        self.assertEqual([c["case_id"] for c in rep["changed_since_previous"]], ["hold-beta"])
        self.assertEqual(rep["failed_cases"], ["hold-beta"])

    def log_entries(self):
        return changes.read_log(self.config.dataset("main").changes)

    def test_holdout_compare_and_gate_keep_the_numbers_without_ids(self):
        self.store.set_baseline(self.hold)
        cmp = holdout.agent_compare(self.store, self.hold, self.hold2, self.log_entries())
        self.assert_no_leak(cmp)
        self.assertEqual(cmp["n_changed_cases"], 1)
        for gone in ("regressions", "improvements", "per_case", "changed_cases"):
            self.assertNotIn(gone, cmp)
        for kept in ("verdict", "diff", "p_value", "n", "baseline_rate", "n_regressions"):
            self.assertIn(kept, cmp)
        gate = holdout.agent_gate(self.store, self.hold2, self.hold, 0.03, self.log_entries())
        self.assert_no_leak(gate)
        self.assertFalse(gate["ok"])  # the edit was not logged by the tool
        self.assertEqual(gate["n_blocking_changes"], 1)
        self.assertIn("1 test or holdout case(s) changed", gate["reason"])

    def test_gate_without_a_baseline_passes_through(self):
        other = self.go()
        gate = holdout.agent_gate(self.store, other, None, 0.03, None)
        self.assertFalse(gate["ok"])
        self.assertIn("no baseline", gate["reason"])

    def test_a_test_comparison_hides_holdout_cases_that_changed(self):
        base = self.go()
        self.edit("t1", split="holdout")  # a person moves a test case into the holdout
        cand = self.go()
        cmp = holdout.agent_compare(self.store, base, cand, self.log_entries())
        self.assert_no_leak(cmp)
        self.assertEqual(cmp["changed_cases"], [{"kind": "missing", "hidden": True}])
        self.assertNotIn("t1", json.dumps(cmp["changed_cases"]))
        self.store.set_baseline(base)
        self.hand_edit("t2", expected="x")  # blocking, and not in the holdout
        gate = holdout.agent_gate(self.store, self.go(), base, 0.03, self.log_entries())
        self.assertFalse(gate["ok"])
        self.assertIn("t2", gate["reason"])
        self.assertEqual(gate["comparison"]["changed_cases"][0], {"kind": "missing", "hidden": True})

    def test_test_report_for_an_agent_hides_holdout_moves(self):
        base = self.go()
        self.edit("t1", split="holdout")
        cand = self.go()
        rep = holdout.agent_report(self.store, cand, self.log_entries())
        self.assertEqual(rep["changed_since_previous"], [{"kind": "missing", "hidden": True}])
        self.assertEqual(rep["run"]["id"], cand)
        self.assertIn("failed_cases", rep)

    def test_show_cases_refuses_a_holdout_run_only(self):
        with self.assertRaisesRegex(holdout.HoldoutError, "holdout run"):
            holdout.check_show_cases(self.store, self.hold)
        holdout.check_show_cases(self.store, self.go())


class AgentRunAndGapsTests(Project):
    def test_holdout_run_for_agent_records_mcp_origin_and_enforces_limit(self):
        run_id = holdout.run_for_agent(self.store, self.config, "a", "main", "holdout", None, None, 1)
        self.assertEqual(self.store.get_run(run_id)["origin"], "mcp")
        with self.assertRaises(holdout.HoldoutError):
            holdout.run_for_agent(self.store, self.config, "a", "main", "holdout", None, None, 1)
        with self.assertRaises(holdout.HoldoutError):
            holdout.run_for_agent(self.store, self.config, "a", "main", "holdout", None, 2, 1)

    def test_a_failing_holdout_check_names_no_case(self):
        self.write_config()
        (self.root / "llmtuner.toml").write_text((self.root / "llmtuner.toml").read_text().replace(
            'type = "exact"', 'type = "json_match"'))
        with self.assertRaises(holdout.HoldoutError) as caught:
            holdout.run_for_agent(self.store, self.config, "a", "main", "holdout", None, None, 1)
        self.assertNotIn("hold-", str(caught.exception))

    def test_log_entry_must_cover_the_changed_field(self):
        base = self.go()
        self.hand_edit("t6", expected="sneaky")
        self.edit("t6", context="logged")  # an entry exists, but not for expected
        found = self.by_id(self.changed(base, self.go()))
        self.assertEqual(found["t6"]["status"], "unrecorded")

    def test_a_case_moved_into_the_split_is_listed_and_blocks_for_an_agent(self):
        base = self.go()
        self.store.set_baseline(base)
        self.edit("d2", split="test", by="agent")
        cand = self.go()
        found = self.by_id(self.changed(base, cand))
        self.assertEqual((found["d2"]["fields"], found["d2"]["status"], found["d2"]["split_before"]),
                         (["split"], "agent", "dev"))
        self.assertFalse(report.gate(self.store, cand, base, 0.03, self.log)["ok"])


def _reserve_worker(home, start, queue):
    """One of several processes sharing a store, all trying to take the holdout attempt at once."""
    store = Store(home)
    while time.time() < start:
        pass
    token, _ = store.reserve_holdout("overfit", "main", time.time(), holdout.WINDOW_SECONDS)
    queue.put(token is not None)
    store.close()


class ReviewFixTests(Project):
    def test_a_later_hand_edit_is_not_covered_by_an_earlier_logged_one(self):
        base = self.go()
        self.edit("t1", expected="WRONG")
        self.hand_edit("t1", expected="OTHER")
        self.store.set_baseline(base)
        cand = self.go()
        self.assertEqual(self.by_id(self.changed(base, cand))["t1"]["status"], "unrecorded")
        self.assertFalse(report.gate(self.store, cand, base, 0.03, self.log)["ok"])

    def test_the_last_logged_value_must_be_the_stored_one(self):
        base = self.go()
        self.edit("t1", expected="first")
        self.edit("t1", expected="second")
        found = self.by_id(self.changed(base, self.go()))
        self.assertEqual(found["t1"]["status"], "human")
        self.assertEqual(len(found["t1"]["entries"]), 2)

    def test_object_values_and_inputs_match_their_log_entries(self):
        base = self.go()
        self.edit("t1", expected={"b": 1, "a": [1, 2]})
        self.edit("t2", input={"k": "v"})
        found = self.by_id(self.changed(base, self.go()))
        self.assertEqual({k: v["status"] for k, v in found.items()}, {"t1": "human", "t2": "human"})

    def test_a_persons_entry_without_a_reason_does_not_count_for_a_test_case(self):
        base = self.go()
        self.hand_edit("t1", expected="x")
        changes.append_entry(self.config.dataset("main").changes, {
            "ts": time.time(), "at": "?", "case_id": "t1", "action": "edit", "by": "human", "via": "cli",
            "agent_signal": None, "reason": None, "basis": None,
            "before": {"expected": "in-t1"}, "after": {"expected": "x"}})
        self.assertEqual(self.by_id(self.changed(base, self.go()))["t1"]["status"], "unrecorded")

    def test_mcp_gate_takes_no_baseline_argument(self):
        source = Path(changes.__file__).with_name("mcp_server.py").read_text()
        signature = source[source.index("def check_gate("):].split(")")[0]
        self.assertNotIn("baseline", signature)

    def test_only_one_process_gets_the_holdout_attempt(self):
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        queue, start = ctx.Queue(), time.time() + 1.0
        procs = [ctx.Process(target=_reserve_worker, args=(str(self.root / "store"), start, queue))
                 for _ in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        self.assertEqual(sorted(queue.get() for _ in procs), [False, False, False, True])

    def test_a_reservation_is_released_when_the_run_never_started(self):
        (self.root / "llmtuner.toml").write_text((self.root / "llmtuner.toml").read_text().replace(
            'type = "exact"', 'type = "json_match"'))
        with self.assertRaises(holdout.HoldoutError):
            holdout.run_for_agent(self.store, self.config, "a", "main", "holdout", None, None, 1)
        self.assertEqual(self.store.holdout_attempts("overfit", "main", 0), [])  # no attempt used

    def test_a_started_run_keeps_the_attempt_after_the_reservation_goes(self):
        holdout.run_for_agent(self.store, self.config, "a", "main", "holdout", None, None, 1)
        self.assertEqual(len(self.store.holdout_attempts("overfit", "main", 0)), 1)
        self.assertEqual(self.store._db.execute("SELECT COUNT(*) FROM holdout_reservations").fetchone()[0], 0)

    def test_concurrent_edits_all_land_and_are_all_logged(self):
        import threading
        ids = [f"t{i}" for i in range(1, 9)]
        errors = []

        def work(case_id):
            try:
                self.edit(case_id, expected=f"new-{case_id}")
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        cases = {c.id: c.expected for c in load_cases(self.data_path)[0]}
        self.assertEqual({i: cases[i] for i in ids}, {i: f"new-{i}" for i in ids})
        self.assertEqual(sorted(e["case_id"] for e in self.log), ids)

    def test_a_leftover_lock_file_blocks_nothing_and_does_not_disturb_the_loader(self):
        (self.root / "cases.changes.jsonl.lock").write_text("stale")
        self.edit("t1", expected="x")
        self.edit("t2", expected="y")
        self.assertEqual(len(load_cases(self.data_path)[0]), len(ROWS))

    def test_locking_nests(self):
        with changes.dataset_lock(self.config.dataset("main")):
            self.edit("t1", expected="x")

    def test_a_limit_that_cut_nothing_is_a_full_run(self):
        base = self.go()
        self.store.set_baseline(base)
        self.write_cases([r for r in ROWS if r["id"] != "t5"])
        cand = self.go(limit=99)
        self.assertEqual([(c["case_id"], c["kind"]) for c in self.changed(base, cand)], [("t5", "missing")])
        self.write_cases(ROWS)
        self.assertTrue(report.gate(self.store, self.go(limit=8), base, 0.03, self.log)["ok"])

    def test_the_gate_fails_for_a_limit_that_cut_cases(self):
        base = self.go()
        cand = self.go(limit=3)
        result = report.gate(self.store, cand, base, 0.03, self.log)
        self.assertFalse(result["ok"])
        self.assertIn("ran on only 3 of 8 cases (limit); the gate needs a full run", result["reason"])
        cmp = report.compare_runs(self.store, base, cand, self.log)
        self.assertTrue(any("only 3 of 8" in w for w in cmp["warnings"]))
        self.assertEqual(cmp["n"], 3)

    def test_an_old_test_run_cannot_be_paired_with_a_holdout_run_for_an_agent(self):
        old = self.go()
        self.store._db.execute("UPDATE cases SET split = NULL WHERE run_id = ?", (old,))
        self.store._db.commit()
        self.edit("t3", split="holdout")
        held = self.go(split="holdout")
        cmp = holdout.agent_compare(self.store, old, held, self.log)
        self.assertEqual((cmp["n"], cmp["n_regressions"], cmp["n_improvements"]), (0, 0, 0))
        # The owner's own view still pairs them.
        self.assertEqual(report.compare_runs(self.store, old, held)["n"], 1)

    def test_a_test_run_stored_with_splits_is_not_paired_with_the_holdout_either(self):
        test_run = self.go()
        self.edit("t3", split="holdout")
        held = self.go(split="holdout")
        self.assertEqual(holdout.agent_compare(self.store, test_run, held, self.log)["n"], 0)
        self.store.set_baseline(test_run)
        gate = holdout.agent_gate(self.store, held, None, 0.03, self.log)
        self.assertEqual(gate["comparison"]["n"], 0)
        rep = holdout.agent_report(self.store, held, self.log)
        self.assertEqual(rep["baseline"]["n"], 0)

    def test_holdout_against_holdout_still_pairs(self):
        first = self.go(split="holdout")
        second = self.go(split="holdout")
        self.assertEqual(holdout.agent_compare(self.store, first, second, self.log)["n"], 3)

    def test_an_agent_may_edit_dev_without_reason_or_basis(self):
        entry = changes.agent_edit(self.config, "main", "d1", reason=None, expected_json='"x"')
        self.assertEqual((entry["by"], entry["basis"], entry["reason"]), ("agent", None, None))
        entry = changes.agent_edit(self.config, "main", "d2", reason="r", basis_kind="", basis_text="",
                                   context="c")
        self.assertIsNone(entry["basis"])
        with self.assertRaisesRegex(ChangeError, "basis"):
            changes.agent_edit(self.config, "main", "t1", reason="r", expected_json='"x"')

    def test_an_unknown_split_prints_a_question_mark(self):
        text = "\n".join(report.format_changes([{
            "case_id": "x", "kind": "changed", "fields": ["expected"], "split_before": None,
            "split_after": None, "status": "before_log", "entries": []}]))
        self.assertNotIn("None", text)
        self.assertIn("(?)", text)

    def test_the_page_does_not_report_a_missing_log_as_a_fault(self):
        from llmtuner import ui
        base = self.go()
        self.edit("t1", expected="x")
        cand = self.go()
        for result in (ui.api_compare(self.store, {"baseline": [str(base)], "candidate": [str(cand)]}),
                       ui.api_report(self.store, cand, {})["baseline"] or {"warnings": []}):
            self.assertNotIn(report.LOG_UNAVAILABLE, result["warnings"])
        self.assertIn(ui.NOT_CHECKED, ui.api_compare(self.store, {"baseline": [str(base)],
                                                                   "candidate": [str(cand)]})["warnings"])


class CliTests(Project):
    def cli(self, *argv, expect=0):
        out, err = io.StringIO(), io.StringIO()
        args = [*argv, "--config", str(self.root), "--store", str(self.root / "cli-store")]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(args)
            except SystemExit as exc:  # argparse rejects bad arguments this way
                code = exc.code
        self.assertEqual(code, expect, err.getvalue() + out.getvalue())
        return out.getvalue(), err.getvalue()

    def test_case_edit_prints_what_changed_and_who(self):
        out, _ = self.cli("case", "edit", "main", "t1", "--expected", "fixed", "--reason", "owner said so")
        self.assertIn("recorded as human", out)
        self.assertIn('"in-t1" -> "fixed"', out)
        self.assertEqual(self.log[-1]["reason"], "owner said so")

    def test_case_edit_as_an_agent_needs_a_basis(self):
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}):
            _, err = self.cli("case", "edit", "main", "t1", "--expected", "x", "--reason", "r", "--by", "human",
                              expect=2)
            self.assertIn("basis", err)
            out, _ = self.cli("case", "edit", "main", "t1", "--expected-json", '{"a": 1}', "--reason", "r",
                              "--basis-rule", "rules/a.md")
        self.assertIn("recorded as agent (CLAUDECODE)", out)
        self.assertEqual((self.log[-1]["by"], self.log[-1]["agent_signal"]), ("agent", "CLAUDECODE"))

    def test_case_edit_errors(self):
        self.cli("case", "edit", "main", "t1", "--reason", "r", expect=2)  # nothing to change
        self.cli("case", "edit", "main", "t1", "--expected-json", "{bad", "--reason", "r", expect=2)
        self.cli("case", "edit", "main", "t1", "--expected", "x", expect=2)  # no reason
        self.assertEqual(self.log, [])

    def test_run_accepts_holdout_and_records_cli_origin(self):
        self.cli("run", "a", "main", "--split", "holdout", "--json")
        self.assertEqual(self.store_run_origin(), "cli")

    def store_run_origin(self):
        with Store(self.root / "cli-store") as store:
            return store.get_run(1)["origin"]

    def test_gate_and_report_read_the_log_through_the_config(self):
        self.cli("run", "a", "main", "--set-baseline", "--json")
        self.cli("case", "edit", "main", "t1", "--expected", "x", "--reason", "fix")
        self.cli("run", "a", "main", "--json")
        out, _ = self.cli("gate")
        self.assertIn("PASS", out)
        self.assertIn("t1", out)
        out, _ = self.cli("report")
        self.assertIn("CHANGED SINCE RUN 1", out)
        self.hand_edit("t2", expected="x")
        self.cli("run", "a", "main", "--json")
        out, _ = self.cli("gate", expect=1)
        self.assertIn("t2", out)

    def split_args(self, *extra):
        return ("split", "main", "--seed", "7", *extra)

    def test_split_keeps_cases_already_run_out_of_holdout_and_says_so(self):
        self.cli("run", "a", "main", "--split", "all")
        seen = {r["id"] for r in ROWS if r.get("split", "test") in ("dev", "test")}
        out, _ = self.cli("split", "main", "--seed", "7", "--json")
        proposed = {c["case_id"]: c["proposed"] for c in json.loads(out)["cases"]}
        self.assertFalse(seen & {k for k, v in proposed.items() if v == "holdout"})
        text, _ = self.cli("split", "main", "--seed", "7", "--dev", "0.25", "--test", "0.25", "--holdout", "0.5")
        self.assertIn("holdout gets 3 of the 7 cases asked for", text)  # only the 3 hold-* were never run

    def test_moving_a_case_that_was_run_into_holdout_warns(self):
        self.cli("run", "a", "main", "--split", "test")
        _, err = self.cli("case", "edit", "main", "t1", "--split", "holdout", "--reason", "hide it")
        self.assertIn("already in a run outside holdout", err)

    def test_split_proposal_writes_nothing(self):
        before = self.data_path.read_bytes()
        out, _ = self.cli(*self.split_args())
        self.assertEqual(self.data_path.read_bytes(), before)
        self.assertEqual(self.log, [])
        self.assertIn("seed 7", out)
        self.assertIn("Nothing was written", out)
        self.assertIn("dev 6, test 4, holdout 3", out)  # 13 cases: 6.5, 3.9, 2.6 -> 6, 4, 3
        self.assertIn("->", out)
        out2, _ = self.cli("split", "main")  # no seed: a random one, always printed
        self.assertRegex(out2, r"seed \d+")

    def test_split_apply_is_refused_in_an_agent_shell(self):
        before = self.data_path.read_bytes()
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1"}):
            _, err = self.cli(*self.split_args("--apply"), expect=2)
        self.assertIn("agent", err)
        self.assertEqual(self.data_path.read_bytes(), before)
        self.assertEqual(self.log, [])

    def test_split_apply_writes_each_change_with_a_log_entry(self):
        self.cli(*self.split_args("--apply"))
        cases, _ = load_cases(self.data_path)
        counts = {s: sum(1 for c in cases if c.split == s) for s in ("dev", "test", "holdout")}
        self.assertEqual(counts, {"dev": 6, "test": 4, "holdout": 3})
        self.assertGreater(len(self.log), 0)
        for entry in self.log:
            self.assertEqual((entry["action"], entry["by"], entry["via"], entry["reason"]),
                             ("split", "human", "cli", "llmtuner split --seed 7"))
        # The same proposal again changes nothing.
        before = self.data_path.read_bytes()
        out, _ = self.cli(*self.split_args("--apply"))
        self.assertIn("0 of 13", out)
        self.assertEqual(self.data_path.read_bytes(), before)

    def test_split_fractions_are_validated(self):
        self.cli("split", "main", "--dev", "0.5", "--test", "0.5", "--holdout", "0.5", expect=2)
        self.cli("split", "main", "--dev", "-1", "--test", "1", "--holdout", "1", expect=2)


class ProposeSplitTests(unittest.TestCase):
    FRACTIONS = {"dev": 0.5, "test": 0.3, "holdout": 0.2}

    @staticmethod
    def counts(result):
        return tuple(sum(1 for v in result.values() if v == s) for s in ("dev", "test", "holdout"))

    def test_seen_cases_never_go_to_holdout_and_the_shortfall_goes_to_dev_and_test(self):
        ids = [f"c{i}" for i in range(10)]
        plain = changes.propose_split(ids, self.FRACTIONS, 3)
        self.assertEqual(plain, changes.propose_split(ids, self.FRACTIONS, 3, frozenset()))
        seen = set(ids[:9])
        result = changes.propose_split(ids, self.FRACTIONS, 3, seen)
        self.assertEqual([k for k, v in result.items() if v == "holdout"], ["c9"])
        self.assertEqual(self.counts(result), (6, 3, 1))  # the missing holdout case: 0.5/0.8 to dev
        self.assertEqual(self.counts(changes.propose_split(ids, self.FRACTIONS, 3, set(ids))), (6, 4, 0))

    def test_counts_use_largest_remainder_and_add_up(self):
        ids = [f"c{i}" for i in range(10)]
        self.assertEqual(self.counts(changes.propose_split(ids, self.FRACTIONS, 1)), (5, 3, 2))
        self.assertEqual(self.counts(changes.propose_split(ids[:7], self.FRACTIONS, 1)), (4, 2, 1))  # 3.5 2.1 1.4
        self.assertEqual(self.counts(changes.propose_split(ids[:3], {"dev": 1 / 3, "test": 1 / 3, "holdout": 1 / 3}, 1)),
                         (1, 1, 1))
        self.assertEqual(self.counts(changes.propose_split(ids, {"dev": 1, "test": 0, "holdout": 0}, 1)), (10, 0, 0))
        for n in range(0, 40):
            many = [f"c{i}" for i in range(n)]
            self.assertEqual(sum(self.counts(changes.propose_split(many, self.FRACTIONS, n))), n)

    def test_the_seed_decides_and_the_input_is_not_mutated(self):
        ids = [f"c{i}" for i in range(30)]
        copy = list(ids)
        first = changes.propose_split(ids, self.FRACTIONS, 42)
        self.assertEqual(first, changes.propose_split(ids, self.FRACTIONS, 42))
        self.assertNotEqual(first, changes.propose_split(ids, self.FRACTIONS, 43))
        self.assertEqual(ids, copy)
        self.assertEqual(set(first), set(ids))

    def test_bad_fractions(self):
        for fractions in ({"dev": 0.5, "test": 0.3, "holdout": 0.3}, {"dev": 0.5, "test": 0.3, "holdout": 0.1},
                          {"dev": -0.2, "test": 0.7, "holdout": 0.5}, {"dev": float("nan"), "test": 1, "holdout": 0}):
            with self.assertRaises(ChangeError, msg=str(fractions)):
                changes.propose_split(["a", "b"], fractions, 1)
        changes.propose_split(["a", "b"], {"dev": 0.7, "test": 0.2, "holdout": 0.1}, 1)  # float sum within 1e-9


if __name__ == "__main__":
    unittest.main()
