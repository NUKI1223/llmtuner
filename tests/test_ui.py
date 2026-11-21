"""UI tests: start the server on port 0 in a thread, against a temp store seeded by stub targets.

Offline, and never touches the user's real store: every call passes an explicit store folder.
"""

import contextlib
import io
import json
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request
from pathlib import Path

from llmtuner import report, runner, ui
from llmtuner.cli import main as cli_main
from llmtuner.config import load_config
from llmtuner.store import Store

PY = sys.executable
XSS = "<script>alert(1)</script>"

STUB = """\
import sys
text = sys.stdin.read().strip()
mode = sys.argv[1]
if mode == "bad" and text in ("c1", "c2", "c3", "c4", "c5", "c6"):
    text = "wrong"
print(text)
"""


def strict_loads(raw: bytes):
    def refuse(token):
        raise ValueError(f"not valid JSON: {token}")

    return json.loads(raw.decode("utf-8"), parse_constant=refuse)


class UiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        project = root / "project"
        project.mkdir()
        self.store_dir = root / "store"
        (project / "stub.py").write_text(STUB)
        rows = [{"id": f"c{i}", "input": f"c{i}", "expected": f"c{i}"} for i in range(1, 9)]
        rows.append({"id": "x1", "input": XSS, "expected": XSS})
        (project / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        targets = "".join(
            f'[targets.{name}]\ncommand = [{json.dumps(PY)}, "stub.py", "{name}"]\ntimeout = 10\n'
            for name in ("good", "bad"))
        (project / "llmtuner.toml").write_text(
            'project = "uidemo"\n' + targets + '[graders.same]\ntype = "exact"\n'
            '[datasets.main]\npath = "cases.jsonl"\n')
        for target in ("good", "bad"):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = cli_main(["run", target, "main", "--config", str(project), "--store", str(self.store_dir)])
            self.assertEqual(code, 0)
        self.running = ui.make_server(self.store_dir, 0)
        self.thread = threading.Thread(target=self.running.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        self.port = self.running.httpd.server_address[1]

    def _stop(self):
        self.running.httpd.shutdown()
        self.running.httpd.server_close()
        self.thread.join(5)

    # --- helpers --------------------------------------------------------------

    def call(self, path, body=None, headers=None, method=None):
        """Return (status, headers, raw bytes). A body means POST."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.running.url.rstrip("/") + path, data=data, method=method)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return res.status, res.headers, res.read()
        except urllib.error.HTTPError as err:
            try:
                return err.code, err.headers, err.read()
            finally:
                err.close()

    def get_json(self, path, status=200):
        code, headers, raw = self.call(path)
        self.assertEqual(code, status, raw)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        return strict_loads(raw)

    def post(self, path, body, token=True, headers=None):
        hdrs = {"Content-Type": "application/json"}
        if token is True:
            hdrs["X-LLMTuner-Token"] = self.running.token
        elif token:
            hdrs["X-LLMTuner-Token"] = token
        hdrs.update(headers or {})
        return self.call(path, body, hdrs)

    def stored_labels(self):
        with Store(self.store_dir) as store:
            return store.get_labels(1)

    # --- GET endpoints --------------------------------------------------------

    def test_projects(self):
        data = self.get_json("/api/projects")
        self.assertEqual(data["projects"][0]["project"], "uidemo")
        self.assertEqual(data["projects"][0]["datasets"][0]["dataset"], "main")

    def test_runs(self):
        data = self.get_json("/api/runs?project=uidemo&limit=5")
        self.assertEqual([r["id"] for r in data["runs"]], [2, 1])
        for key in ("pass_rate", "cost_per_pass", "n", "n_passed", "target", "is_baseline", "cost_units", "mixed_units"):
            self.assertIn(key, data["runs"][0])
        self.assertEqual(len(self.get_json("/api/runs?limit=1")["runs"]), 1)
        self.get_json("/api/runs?limit=abc", status=400)

    def test_runs_rows_carry_mixed_unit_fields(self):
        mixed = {"cost": None, "cost_unit": None, "cost_per_pass": None, "cost_units": ["credits", "usd"], "mixed_units": True}
        with mock.patch.object(ui.report, "usage", return_value=mixed):
            rows = self.get_json("/api/runs?project=uidemo")["runs"]
        for row in rows:
            self.assertTrue(row["mixed_units"])
            self.assertEqual(row["cost_units"], ["credits", "usd"])
            self.assertIsNone(row["cost_per_pass"])

    def test_report(self):
        rep = self.get_json("/api/runs/2")
        for key in ("run", "n", "pass_rate", "detection_floor", "graders", "grader_checks", "baseline", "usage"):
            self.assertIn(key, rep)
        self.assertEqual(rep["n"], 9)

    def test_cases_and_failed_only(self):
        data = self.get_json("/api/runs/2/cases")
        self.assertEqual(data["total"], 9)
        case = data["cases"][0]
        for key in ("case_id", "input", "expected", "output", "grades", "duration_s", "meta", "passed", "truncated"):
            self.assertIn(key, case)
        failed = self.get_json("/api/runs/2/cases?failed=1")
        self.assertEqual(failed["total"], 6)
        self.assertTrue(all(not c["passed"] for c in failed["cases"]))
        one = self.get_json("/api/runs/2/cases?case=c1")
        self.assertEqual(one["shown"], 1)
        self.get_json("/api/runs/2/cases?case=nope", status=404)

    def test_long_text_is_cut_and_says_so(self):
        import sqlite3
        long_text = "z" * 30_000
        db = sqlite3.connect(self.store_dir / "store.db")
        db.execute("UPDATE cases SET output = ? WHERE run_id = 2 AND case_id = 'c1'", (long_text,))
        db.commit()
        db.close()
        case = self.get_json("/api/runs/2/cases?case=c1")["cases"][0]
        self.assertEqual(len(case["output"]), 20_000)
        self.assertEqual(case["truncated"], {"output": 30_000})

    def test_compare_rank(self):
        cmp = self.get_json("/api/compare?baseline=1&candidate=2")
        for key in ("verdict", "warnings", "regressions", "improvements", "diff", "p_value", "baseline", "candidate"):
            self.assertIn(key, cmp)
        self.assertEqual(cmp["verdict"], "candidate is worse")
        rank = self.get_json("/api/rank?project=uidemo&dataset=main&split=test")
        self.assertEqual(len(rank["rows"]), 2)
        self.assertEqual(rank["rows"][0]["vs_leader"], "leader")
        self.get_json("/api/rank?project=uidemo&dataset=main&split=bogus", status=400)
        self.get_json("/api/rank?project=uidemo", status=400)
        self.get_json("/api/compare?baseline=1&candidate=99", status=404)

    def test_unknown_run_and_paths_are_json_404(self):
        for path in ("/api/runs/999", "/api/runs/999/cases", "/api/runs/999/queue", "/api/nothing", "/api/runs/abc"):
            body = self.get_json(path, status=404)
            self.assertIn("error", body)
            self.assertNotIn("Traceback", json.dumps(body))

    # --- host, token, origin --------------------------------------------------

    def test_wrong_host_is_refused(self):
        for host in ("evil.example", f"evil.example:{self.port}", "127.0.0.1:1", "localhost"):
            code, _, raw = self.call("/api/projects", headers={"Host": host})
            self.assertEqual(code, 403, host)
            code, _, _ = self.call("/", headers={"Host": host})
            self.assertEqual(code, 403, host)
        code, _, _ = self.call("/api/projects", headers={"Host": f"localhost:{self.port}"})
        self.assertEqual(code, 200)

    def test_post_needs_token_and_own_origin(self):
        body = {"run": 1, "case_id": "c1", "passed": True}
        attempts = [
            self.post("/api/label", body, token=False),
            self.post("/api/label", body, token="wrong-token"),
            self.post("/api/label", body, headers={"Origin": "http://evil.example"}),
            self.post("/api/label", body, headers={"Origin": "null"}),
            self.post("/api/baseline", {"run": 1}, token=False),
            self.post("/api/baseline", {"run": 1}, headers={"Origin": "http://evil.example"}),
            self.post("/api/label", body, headers={"Host": "evil.example"}),
        ]
        for code, _, raw in attempts:
            self.assertEqual(code, 403, raw)
            self.assertIn("error", strict_loads(raw))
        self.assertEqual(self.stored_labels(), [])
        with Store(self.store_dir) as store:
            self.assertIsNone(store.get_baseline("uidemo", "main"))
        origin = f"http://127.0.0.1:{self.port}"
        code, _, _ = self.post("/api/label", body, headers={"Origin": origin})
        self.assertEqual(code, 200)

    def test_post_bad_input_and_size_cap(self):
        for body in ({"run": 1, "case_id": "c1"}, {"run": "x", "case_id": "c1", "passed": True},
                     {"run": 1, "case_id": "c1", "passed": "yes"}, {"run": 1, "case_id": "c1", "passed": True, "grader": "nope"}):
            code, _, _ = self.post("/api/label", body)
            self.assertIn(code, (400, 404), body)
        code, _, _ = self.post("/api/label", {"run": 1, "case_id": "zzz", "passed": True})
        self.assertEqual(code, 404)
        code, _, _ = self.post("/api/label", {"run": 1, "case_id": "c1", "passed": True, "note": "n" * 70_000})
        self.assertEqual(code, 413)
        code, _, _ = self.call("/api/label", None, {"X-LLMTuner-Token": self.running.token}, method="POST")
        self.assertEqual(code, 400)
        self.assertEqual(self.stored_labels(), [])

    # --- labelling ------------------------------------------------------------

    def test_label_writes_and_report_shows_it(self):
        code, _, raw = self.post("/api/label", {"run": 1, "case_id": "c1", "passed": True, "note": "ok"})
        self.assertEqual(code, 200, raw)
        labels = self.stored_labels()
        self.assertEqual([(l["case_id"], l["passed"], l["note"]) for l in labels], [("c1", 1, "ok")])
        rep = self.get_json("/api/runs/1")
        self.assertEqual(rep["grader_checks"][0]["grader"], "overall verdict")
        self.assertEqual(rep["grader_checks"][0]["n"], 1)
        code, _, _ = self.post("/api/label", {"run": 1, "case_id": "c2", "passed": False, "grader": "same"})
        self.assertEqual(code, 200)
        self.assertEqual({c["grader"] for c in self.get_json("/api/runs/1")["grader_checks"]}, {"overall verdict", "same"})

    def test_queue_has_only_allowed_keys_and_shrinks(self):
        q = self.get_json("/api/runs/2/queue")
        self.assertEqual(q["remaining"], 9)
        self.assertEqual(q["done"], 0)
        self.assertIn("acceptable overall", q["question"])
        self.assertEqual(len(q["items"]), 9)
        for item in q["items"]:
            self.assertEqual(set(item), {"case_id", "input", "context", "expected", "output"})
        text = json.dumps(q)
        self.assertNotIn("passed", text)
        self.assertNotIn('"grades"', text)
        again = self.get_json("/api/runs/2/queue")
        self.assertEqual([i["case_id"] for i in q["items"]], [i["case_id"] for i in again["items"]])  # stable order
        first = q["items"][0]["case_id"]
        self.assertEqual(self.post("/api/label", {"run": 2, "case_id": first, "passed": True})[0], 200)
        after = self.get_json("/api/runs/2/queue")
        self.assertEqual((after["remaining"], after["done"]), (8, 1))
        self.assertNotIn(first, [i["case_id"] for i in after["items"]])
        graded = self.get_json("/api/runs/2/queue?grader=same")
        self.assertEqual(graded["remaining"], 9)  # labels are kept per grader
        self.assertIn("'same'", graded["question"])
        self.get_json("/api/runs/2/queue?grader=nope", status=404)

    # --- baseline -------------------------------------------------------------

    def test_make_baseline(self):
        code, _, raw = self.post("/api/baseline", {"run": 1})
        self.assertEqual(code, 200, raw)
        with Store(self.store_dir) as store:
            self.assertEqual(store.get_baseline("uidemo", "main"), 1)
        rep = self.get_json("/api/runs/2")
        self.assertEqual(rep["baseline"]["baseline"]["id"], 1)
        self.assertTrue(self.get_json("/api/runs/1")["is_baseline"])
        self.assertEqual(self.post("/api/baseline", {"run": 999})[0], 404)

    # --- untrusted text and the page ------------------------------------------

    def test_script_text_is_a_plain_json_string(self):
        case = self.get_json("/api/runs/1/cases?case=x1")["cases"][0]
        self.assertEqual(case["input"], XSS)
        self.assertEqual(case["output"].strip(), XSS)
        code, headers, raw = self.call("/api/runs/1/cases?case=x1")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertTrue(headers["Content-Type"].startswith("application/json"))

    def test_page_headers_and_nonce(self):
        code, headers, raw = self.call("/")
        self.assertEqual(code, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        csp = headers["Content-Security-Policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        html = raw.decode()
        nonce = re.search(r"script-src 'nonce-([^']+)'", csp).group(1)
        self.assertIn(f'<script nonce="{nonce}">', html)
        self.assertIn(f'<style nonce="{nonce}">', html)
        self.assertIn(f"style-src 'nonce-{nonce}'", csp)
        self.assertIn(self.running.token, html)
        self.assertNotIn("__NONCE__", html)
        self.assertNotIn("__TOKEN__", html)
        _, headers2, _ = self.call("/")
        self.assertNotEqual(headers2["Content-Security-Policy"], csp)  # fresh nonce per response

    def test_page_source_is_safe_and_self_contained(self):
        source = (Path(ui.__file__).with_name("ui_page.html")).read_text(encoding="utf-8")
        for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            self.assertNotIn(banned, source)
        self.assertIsNone(re.search(r" on[a-z]+=", source), "inline event handler attribute")
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)
        self.assertNotIn("//cdn", source)

    # --- strict validation, ids, replies, timeouts ------------------------------

    def test_bad_label_bodies_are_400_and_write_nothing(self):
        good = {"run": 1, "case_id": "c1", "passed": True}
        bad = [
            {**good, "note": "\ud800"}, {**good, "case_id": "\ud800"}, {**good, "grader": "\ud800"},
            {**good, "run": True}, {**good, "run": 1.9}, {**good, "run": "1"}, {**good, "run": " 1_0 "},
            {**good, "run": 0}, {**good, "run": -1}, {**good, "run": 10**30}, {**good, "run": None},
            {**good, "passed": 1}, {**good, "passed": "true"}, {**good, "passed": None},
            {**good, "case_id": ""}, {**good, "case_id": 5}, {**good, "case_id": "x" * 1001},
            {**good, "grader": "g" * 1001}, {**good, "grader": 3},
            {**good, "note": "n" * 10_001}, {**good, "note": 7},
        ]
        for body in bad:
            code, _, raw = self.post("/api/label", body)
            self.assertEqual(code, 400, (body, raw))
            self.assertIn("error", strict_loads(raw))
        for body in ({"run": True}, {"run": 1.5}, {"run": "1"}, {"run": 0}, {"run": 10**30}):
            code, _, raw = self.post("/api/baseline", body)
            self.assertEqual(code, 400, (body, raw))
        self.assertEqual(self.stored_labels(), [])
        with Store(self.store_dir) as store:
            self.assertIsNone(store.get_baseline("uidemo", "main"))
        # limits are inclusive
        ok = {**good, "case_id": "c1", "note": "n" * 10_000}
        self.assertEqual(self.post("/api/label", ok)[0], 200)

    def test_ids_in_paths_and_queries_are_strict(self):
        for path in ("/api/runs/" + "9" * 5000, "/api/runs/-1", "/api/runs/1e3", "/api/runs/%EF%BC%91",
                     "/api/runs/", "/api/runs/1_0", "/api/runs/" + "1" * 19, "/api/runs/%EF%BC%91/cases"):
            self.get_json(path, status=404)
        self.get_json("/api/runs/" + "0" * 17 + "1")  # 18 digits is the cap, and leading zeros are fine
        for path in ("/api/runs?limit=1e3", "/api/runs?limit=%EF%BC%91", "/api/runs?limit=-1", "/api/runs?limit=+1",
                     "/api/runs?limit=" + "9" * 5000, "/api/runs?limit=1%0A", "/api/runs?limit=0",
                     "/api/compare?baseline=%EF%BC%91&candidate=2", "/api/compare?baseline=1&candidate=2.0",
                     "/api/compare?baseline=1&candidate=" + "1" * 19):
            self.get_json(path, status=400)

    def test_lone_surrogate_in_a_reply_never_drops_it(self):
        routes = [(re.compile(r"/api/projects"), lambda store, query: {"text": "\ud800 ok"})]
        with mock.patch.object(ui, "_ROUTES_GET", routes):
            body = self.get_json("/api/projects")
        self.assertEqual(body["text"], "\ud800 ok")

    def test_unexpected_error_is_500_with_traceback_on_stderr(self):
        err = io.StringIO()
        with mock.patch.object(Store, "list_datasets", side_effect=RuntimeError("boom-marker")):
            with contextlib.redirect_stderr(err):
                code, _, raw = self.call("/api/projects")
        self.assertEqual(code, 500)
        body = strict_loads(raw)
        self.assertIn("terminal", body["error"])
        self.assertNotIn("boom-marker", json.dumps(body))
        self.assertIn("Traceback", err.getvalue())
        self.assertIn("boom-marker", err.getvalue())

    def test_deeply_nested_json_is_400(self):
        raw = b"[" * 30_000
        req = urllib.request.Request(self.running.url + "api/label", data=raw, method="POST",
                                     headers={"X-LLMTuner-Token": self.running.token})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)
        ctx.exception.close()

    def _raw(self, payload: bytes, wait: float = 6.0) -> bytes:
        with socket.create_connection(("127.0.0.1", self.port), timeout=wait) as sock:
            sock.sendall(payload)
            chunks = []
            while True:
                data = sock.recv(4096)  # a socket.timeout here means the server never closed
                if not data:
                    return b"".join(chunks)
                chunks.append(data)

    def test_stalled_request_times_out_and_server_keeps_serving(self):
        head = (f"POST /api/label HTTP/1.0\r\nHost: 127.0.0.1:{self.port}\r\n"
                f"X-LLMTuner-Token: {self.running.token}\r\nContent-Length: 100\r\n\r\n").encode()
        with mock.patch.object(ui, "REQUEST_TIMEOUT", 1.0):
            started = time.monotonic()
            reply = self._raw(head + b"{")  # promises 100 bytes, sends 1
            self.assertLess(time.monotonic() - started, 5)
            self.assertTrue(reply.startswith(b"HTTP/1.0 408"), reply[:40])
            started = time.monotonic()
            self._raw(b"POST /api/label HTTP/1.0\r\n")  # headers never finish: closed without an answer
            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(len(self.get_json("/api/projects")["projects"]), 1)
        self.assertEqual(self.stored_labels(), [])

    def test_content_length_must_be_plain_digits_and_single(self):
        base = f"POST /api/baseline HTTP/1.0\r\nHost: 127.0.0.1:{self.port}\r\nX-LLMTuner-Token: {self.running.token}\r\n"
        for lines in ("Content-Length: 1_0\r\n", "Content-Length: +10\r\n", "Content-Length: 9\r\nContent-Length: 100\r\n"):
            reply = self._raw((base + lines + "\r\n{\"run\":2}").encode())
            self.assertTrue(reply.startswith(b"HTTP/1.0 400"), (lines, reply[:30]))
        with Store(self.store_dir) as store:
            self.assertIsNone(store.get_baseline("uidemo", "main"))

    def test_page_shows_tiny_costs_as_nonzero(self):
        self.assertIn("toExponential", self._page())

    def test_main_rejects_bad_ports_in_one_line(self):
        for port in (-1, 65536, 70000, "80", 1.5, True):
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = ui.main(self.store_dir, port, False)
            self.assertEqual(code, 1, port)
            self.assertEqual(len(err.getvalue().strip().splitlines()), 1, port)
            self.assertNotIn("Traceback", err.getvalue())

    # --- page source checks (the page cannot be run here) -----------------------

    def _page(self) -> str:
        return (Path(ui.__file__).with_name("ui_page.html")).read_text(encoding="utf-8")

    def test_page_routes_are_exact_and_cleanups_are_registered(self):
        source = self._page()
        self.assertNotIn('indexOf("#/label/"', source)
        self.assertNotIn("location.hash.indexOf", source)
        self.assertIn("++navId", source)
        self.assertIn("cleanups", source)
        self.assertIn("function parseRoute", source)
        self.assertIn("ctx.onLeave(", source)
        self.assertNotIn("cleanup = ", source)
        # every fetch continuation goes through ctx.live
        for view in ("viewRuns", "viewRun", "viewCompare", "viewRank", "viewLabel"):
            body = source.split("function " + view + "(")[1].split("\n  function ")[0]
            self.assertIn("ctx.live(", body, view)
            self.assertNotRegex(body, r"\.then\(function", view)
        key = source.split("function onKey(e)")[1].split("document.addEventListener")[0]
        self.assertIn("busy", key)
        self.assertIn("INPUT", key)

    def test_page_never_turns_unknown_usage_into_zero(self):
        source = self._page()
        self.assertNotRegex(source, r"\|\|\s*0\b")
        for needle in ("mixed_units", "cost_units", "cost_note", "different units", "not reported"):
            self.assertIn(needle, source)

    # --- main -----------------------------------------------------------------

    def test_main_reports_busy_port(self):
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        self.addCleanup(busy.close)
        port = busy.getsockname()[1]
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = ui.main(self.store_dir, port, False)
        self.assertEqual(code, 1)
        self.assertIn(f"127.0.0.1:{port}", err.getvalue())
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1)
        self.assertNotIn("Traceback", err.getvalue())

    def test_server_binds_loopback_only(self):
        self.assertEqual(self.running.httpd.server_address[0], "127.0.0.1")


TRIAL_STUB = """\
import os, sys
text = sys.stdin.read().strip()
mode = sys.argv[1]
trial = int(os.environ.get("LLMTUNER_TRIAL", "1"))
# "flaky" fails c1 in trial 2 only (unstable) and c2 in every trial (always fails).
if mode == "flaky" and ((text == "c1" and trial == 2) or text == "c2"):
    text = "wrong"
print(text)
"""


class TrialUiTests(unittest.TestCase):
    """Repeated trials through the web API. Stores: run 1 and 2 are ordinary (steady, flaky),
    runs 3-5 are a group of steady trials and 6-8 a group of flaky trials."""

    call = UiTests.call
    get_json = UiTests.get_json
    post = UiTests.post
    _stop = UiTests._stop

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        project = root / "project"
        project.mkdir()
        self.store_dir = root / "store"
        (project / "stub.py").write_text(TRIAL_STUB)
        rows = [{"id": f"c{i}", "input": f"c{i}", "expected": f"c{i}"} for i in range(1, 5)]
        (project / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        targets = "".join(
            f'[targets.{name}]\ncommand = [{json.dumps(PY)}, "stub.py", "{name}"]\ntimeout = 10\n'
            for name in ("steady", "flaky"))
        (project / "llmtuner.toml").write_text(
            'project = "trialdemo"\n' + targets + '[graders.same]\ntype = "exact"\n'
            '[datasets.main]\npath = "cases.jsonl"\n')
        config = load_config(project)
        with Store(self.store_dir) as store:
            for target, trials in (("steady", 1), ("flaky", 1), ("steady", 3), ("flaky", 3)):
                runner.run_eval(config, store, target, "main", trials=trials)
        self.running = ui.make_server(self.store_dir, 0)
        self.thread = threading.Thread(target=self.running.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        self.port = self.running.httpd.server_address[1]

    def test_runs_has_one_row_per_group_placed_by_newest_trial(self):
        rows = self.get_json("/api/runs?project=trialdemo")["runs"]
        self.assertEqual([r["id"] for r in rows], [6, 3, 2, 1])
        self.assertEqual([r["trials"] for r in rows], [3, 3, 1, 1])
        flaky = rows[0]
        self.assertEqual(flaky["trial_runs"], [6, 7, 8])
        self.assertEqual((flaky["trials_finished"], flaky["n"], flaky["n_passed"]), (3, 4, 2))
        self.assertEqual(flaky["target"], "flaky")
        self.assertNotIn("trial_runs", rows[2])

    def test_runs_limit_counts_rows_not_runs(self):
        for limit, ids in ((1, [6]), (2, [6, 3]), (3, [6, 3, 2]), (50, [6, 3, 2, 1])):
            rows = self.get_json(f"/api/runs?project=trialdemo&limit={limit}")["runs"]
            self.assertEqual([r["id"] for r in rows], ids, limit)

    def test_group_row_numbers_equal_the_report(self):
        rows = {r["id"]: r for r in self.get_json("/api/runs?project=trialdemo")["runs"]}
        for first in (3, 6):
            trials = self.get_json(f"/api/runs/{first}")["trials"]
            row = rows[first]
            self.assertEqual(row["trials"], trials["planned"])
            self.assertEqual(row["trials_finished"], trials["finished"])
            self.assertEqual(row["trial_runs"], trials["runs"])
            self.assertEqual(row["n"], trials["n"])
            self.assertEqual(row["n_passed"], trials["passed_all"]["n_passed"])
            self.assertEqual(row["pass_rate"], trials["passed_all"]["rate"])
            for key in ("cost_per_pass", "cost_unit", "cost_units", "mixed_units"):
                self.assertEqual(row[key], trials["usage"][key], key)
        self.assertEqual(rows[3]["n_passed"], 4)
        self.assertEqual(rows[6]["n_passed"], 2)

    def test_group_baseline_flag_follows_the_group(self):
        code, _, raw = self.post("/api/baseline", {"run": 4})  # a middle trial of the steady group
        self.assertEqual(code, 200, raw)
        self.assertEqual(strict_loads(raw)["baseline_run"], 3)
        rows = {r["id"]: r for r in self.get_json("/api/runs?project=trialdemo")["runs"]}
        self.assertTrue(rows[3]["is_baseline"])
        self.assertFalse(rows[6]["is_baseline"])
        self.assertFalse(rows[2]["is_baseline"])
        for trial in (3, 4, 5):
            self.assertTrue(self.get_json(f"/api/runs/{trial}")["is_baseline"], trial)

    def test_unfinished_group_is_one_row_with_strict_json(self):
        with Store(self.store_dir) as store:
            first = store.create_run(project="trialdemo", dataset="main", dataset_hash="h", target="steady",
                                     command=["x"], model=None, tool_version=None, split="test",
                                     graders=[], note=None, trials_planned=3, trial=1)
            single = store.create_run(project="trialdemo", dataset="main", dataset_hash="h", target="steady",
                                      command=["x"], model=None, tool_version=None, split="test",
                                      graders=[], note=None, trials_planned=1, trial=1)
        rows = self.get_json("/api/runs?project=trialdemo")["runs"]
        self.assertEqual([r["id"] for r in rows][:2], [single, first])
        self.assertEqual((rows[1]["trials"], rows[1]["trials_finished"], rows[1]["n"]), (3, 0, 0))
        self.assertIsNone(rows[1]["pass_rate"]["point"])
        self.assertEqual((rows[0]["trials"], rows[0]["trial_runs"]), (1, [single]))
        self.get_json(f"/api/runs/{first}")  # the report of a group with nothing finished is strict JSON too

    def test_run_endpoint_has_trials_for_any_trial_of_a_group(self):
        group = self.get_json("/api/runs/6")["trials"]
        for run_id in (6, 7, 8):
            rep = self.get_json(f"/api/runs/{run_id}")
            self.assertEqual(rep["trials"], group)
        for key in ("group", "runs", "planned", "finished", "complete", "n", "cases", "passed_all", "passed_any",
                    "per_trial", "unstable_cases", "error_cases", "target_error_trials", "usage"):
            self.assertIn(key, group)
        self.assertEqual(group["unstable_cases"], ["c1"])
        self.assertEqual(group["group"], 6)
        self.assertIsNone(self.get_json("/api/runs/1")["trials"])
        # the rest of a trial's report describes that trial alone
        self.assertEqual(self.get_json("/api/runs/7")["n_passed"], 2)  # c1 and c2 fail in trial 2
        self.assertEqual(self.get_json("/api/runs/8")["n_passed"], 3)
        self.assertEqual(self.get_json("/api/runs/7/cases")["total"], 4)

    def test_cases_labels_and_queue_stay_per_trial_run(self):
        queue = self.get_json("/api/runs/7/queue")
        self.assertEqual(queue["run"], 7)
        self.assertEqual(len(queue["items"]), 4)
        for item in queue["items"]:
            self.assertEqual(set(item), {"case_id", "input", "context", "expected", "output"})
        self.assertEqual(self.post("/api/label", {"run": 7, "case_id": "c1", "passed": True})[0], 200)
        self.assertEqual(self.get_json("/api/runs/7/queue")["remaining"], 3)
        self.assertEqual(self.get_json("/api/runs/6/queue")["remaining"], 4)
        with Store(self.store_dir) as store:
            self.assertEqual([(l["run_id"], l["case_id"]) for l in store.get_labels(7)], [(7, "c1")])
            self.assertEqual(store.get_labels(6), [])

    def test_compare_has_per_case_for_groups_only(self):
        cmp = self.get_json("/api/compare?baseline=3&candidate=6")
        self.assertEqual((cmp["baseline_trials"], cmp["candidate_trials"]), (3, 3))
        by_case = {row["case_id"]: row for row in cmp["per_case"]}
        self.assertEqual(set(by_case), {"c1", "c2", "c3", "c4"})
        self.assertEqual((by_case["c1"]["baseline_passes"], by_case["c1"]["candidate_passes"]), (3, 2))
        self.assertEqual((by_case["c2"]["baseline_trials"], by_case["c2"]["candidate_trials"]), (3, 3))
        self.assertEqual(sorted(cmp["regressions"]), ["c1", "c2"])  # failed a trial: no longer passed all
        self.assertEqual(cmp, strict_loads(json.dumps(report.json_safe(cmp)).encode()))
        plain = self.get_json("/api/compare?baseline=1&candidate=2")
        self.assertIsNone(plain["per_case"])
        self.assertEqual((plain["baseline_trials"], plain["candidate_trials"]), (1, 1))
        mixed = self.get_json("/api/compare?baseline=1&candidate=6")
        self.assertIsNotNone(mixed["per_case"])
        self.assertTrue(any("trial" in w for w in mixed["warnings"]))

    def test_rank_rows_carry_trials(self):
        rank = self.get_json("/api/rank?project=trialdemo&dataset=main&split=test")
        self.assertEqual([(r["target"], r["trials"], r["run"]) for r in rank["rows"]],
                         [("steady", 3, 3), ("flaky", 3, 6)])
        self.assertEqual(rank["rows"][0]["n_passed"], 4)
        self.assertEqual(rank["rows"][1]["n_passed"], 2)
        self.assertEqual(rank["skipped"], [])  # the ordinary runs are older runs of targets already ranked

    def test_page_shows_trials_in_words_and_stays_text_only(self):
        source = (Path(ui.__file__).with_name("ui_page.html")).read_text(encoding="utf-8")
        for needle in ("trial_runs", "trials_finished", "passed all", "Passed at least once", "Per-trial pass rate",
                       "Unstable cases", "this trial alone", "per passing case result", "Provisional",
                       "per_case", "1 in 20", "not counted as instability", "every trial"):
            self.assertIn(needle, source, needle)
        self.assertNotRegex(source, r"\|\|\s*0\b")

    def test_page_labels_figures_by_the_finished_trials(self):
        source = (Path(ui.__file__).with_name("ui_page.html")).read_text(encoding="utf-8")
        # the old helper took only the planned number; figure labels must use the finished count
        self.assertNotIn("allTrials(k)", source)
        self.assertNotIn('"Passed all " + k', source)
        self.assertNotIn('"Usage, all " + plural(k', source)
        for needle in ("allTrials(finished, k)", "allTrials(tr.finished, k)", "trialsPhrase(tr.finished, k)",
                       " finished", " planned)"):
            self.assertIn(needle, source, needle)
        self.assertNotIn("trial run\"", source)  # no "passing trial run" label
        self.assertNotIn("trial result", source)
        self.assertIn("case results", source)
        self.assertIn("passing case result", source)
        # tile 4 shows the comparison's warnings as text: the condition, the heading and the list
        self.assertIn("b.warnings && b.warnings.length ?", source)
        self.assertIn('" about this comparison:"', source)
        self.assertIn("b.warnings.map(", source)
        # one finished trial reads as on the command line, and "that finished" only when some did not
        self.assertIn('"passed the " + trialsPhrase(k, planned)', source)
        self.assertNotIn('"passed all " + trialsPhrase(k, planned) : "passed its only trial"', source)
        self.assertIn('(tr.finished < k ? " that finished." : ".")', source)

    def test_report_of_a_single_run_carries_the_baseline_warnings(self):
        self.assertEqual(self.post("/api/baseline", {"run": 4})[0], 200)
        single = self.get_json("/api/runs/1")
        self.assertEqual(single["figures_describe"], "this run")
        self.assertTrue(any("trial(s) per case" in w for w in single["baseline"]["warnings"]))
        grouped = self.get_json("/api/runs/7")
        self.assertTrue(grouped["figures_describe"].startswith("trial 2 of 3 (run 7) alone;"))


if __name__ == "__main__":
    unittest.main()
