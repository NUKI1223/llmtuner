"""MCP server test: run the server as a subprocess over stdio and talk to it with the SDK client.

No network and no model calls: the target is a stub script. Skipped when `mcp` is missing.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from llmtuner.store import Store

try:
    from mcp import Client, StdioServerParameters
except ImportError:  # the core package has no dependencies; mcp is optional
    Client = StdioServerParameters = None

SRC = str(Path(__file__).resolve().parent.parent / "src")

STUB = """\
import sys
text = sys.stdin.read().strip()
print("yes" if text == "ping" else "no")
"""

DATASET = "\n".join(json.dumps(row) for row in [
    {"id": "a", "input": "ping", "expected": "yes"},
    {"id": "b", "input": "pong", "expected": "yes"},
]) + "\n"


def make_config() -> str:
    return f"""\
project = "stubproj"

[targets.stub]
command = [{json.dumps(sys.executable)}, "stub.py"]

[datasets.main]
path = "cases.jsonl"

[graders.answer]
type = "exact"
"""


@unittest.skipUnless(Client is not None, "the 'mcp' package is not installed")
class McpServerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.project = root / "project"
        self.project.mkdir()
        (self.project / "stub.py").write_text(STUB, encoding="utf-8")
        (self.project / "cases.jsonl").write_text(DATASET, encoding="utf-8")
        (self.project / "llmtuner.toml").write_text(make_config(), encoding="utf-8")
        self.home = root / "home"  # never the user's real store
        self.store_dir = root / "chosen-store"

    def params(self, store_dir=None):
        existing = os.environ.get("PYTHONPATH")
        args = ["-m", "llmtuner.mcp_server"]
        if store_dir is not None:
            args = ["-m", "llmtuner", "mcp", "--store", str(store_dir)]
        return StdioServerParameters(
            command=sys.executable,
            args=args,
            env={
                "PYTHONPATH": SRC + (os.pathsep + existing if existing else ""),
                "LLMTUNER_HOME": str(self.home),
            },
        )

    def test_store_option_is_honoured(self):
        asyncio.run(asyncio.wait_for(self._store_option(), timeout=90))

    async def _store_option(self):
        async with Client(self.params(self.store_dir)) as client:
            ran = await client.call_tool("run_eval", {
                "project_dir": str(self.project), "target": "stub", "dataset": "main",
            })
            self.assertFalse(ran.is_error, ran.content)
            run_id = ran.structured_content["run"]["id"]
            # Every tool reads the same --store, not the default one.
            report = await client.call_tool("get_report", {"run_id": run_id})
            self.assertFalse(report.is_error, report.content)
            listed = await client.call_tool("list_runs", {})
            self.assertEqual([r["id"] for r in listed.structured_content["result"]], [run_id])
        with Store(self.store_dir) as store:
            self.assertEqual([r["id"] for r in store.list_runs()], [run_id])
        self.assertFalse((self.home / "store.db").exists(), "the run leaked into $LLMTUNER_HOME")

    def test_round_trip(self):
        asyncio.run(asyncio.wait_for(self._round_trip(), timeout=90))

    async def _round_trip(self):
        async with Client(self.params()) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            expected = {"describe_project", "run_eval", "get_report", "compare_runs",
                        "check_gate", "list_runs", "show_cases"}
            self.assertTrue(expected <= names, names)
            # Nothing that executes commands or writes ground truth is exposed.
            self.assertEqual(names, expected)

            described = await client.call_tool("describe_project", {"project_dir": str(self.project)})
            self.assertFalse(described.is_error)
            self.assertEqual(described.structured_content["project"], "stubproj")
            self.assertIn("stub", described.structured_content["targets"])

            ran = await client.call_tool("run_eval", {
                "project_dir": str(self.project), "target": "stub", "dataset": "main",
            })
            self.assertFalse(ran.is_error, ran.content)
            run = ran.structured_content
            run_id = run["run"]["id"]
            self.assertEqual((run["n"], run["n_passed"]), (2, 1))
            self.assertEqual(run["failed_cases"], ["b"])

            report = await client.call_tool("get_report", {"run_id": run_id})
            self.assertFalse(report.is_error)
            self.assertEqual(report.structured_content["n_passed"], 1)

            failed = await client.call_tool("show_cases", {"run_id": run_id})
            self.assertFalse(failed.is_error)
            cases = failed.structured_content["result"]
            self.assertEqual([c["case_id"] for c in cases], ["b"])
            self.assertEqual(cases[0]["input"], "pong")
            self.assertEqual(cases[0]["output"].strip(), "no")
            self.assertEqual(cases[0]["grades"][0]["grader"], "answer")
            self.assertFalse(cases[0]["grades"][0]["passed"])

            everything = await client.call_tool("show_cases", {"run_id": run_id, "failed_only": False})
            self.assertEqual(len(everything.structured_content["result"]), 2)

            listed = await client.call_tool("list_runs", {})
            self.assertEqual(listed.structured_content["result"][0]["id"], run_id)
            self.assertTrue((self.home / "store.db").exists())  # the default store, here $LLMTUNER_HOME

            bad_target = await client.call_tool("run_eval", {
                "project_dir": str(self.project), "target": "nope", "dataset": "main",
            })
            self.assertTrue(bad_target.is_error)
            self.assertIn("unknown target 'nope'", bad_target.content[0].text)

            bad_run = await client.call_tool("get_report", {"run_id": 9999})
            self.assertTrue(bad_run.is_error)
            self.assertIn("no run with id 9999", bad_run.content[0].text)


if __name__ == "__main__":
    unittest.main()
