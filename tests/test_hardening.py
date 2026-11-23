"""Regression tests for defects found by an independent check: store migration race, usage states,
rank selection, and the OpenRouter adapter's redirects, key masking, deadline and odd inputs.

Offline: stubs listen on 127.0.0.1 only; the adapter runs in a temp cwd with a temp HOME and a fake key.
"""

import contextlib
import http.server
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_cost_rank as tcr  # noqa: E402
import test_openrouter as tor  # noqa: E402

from llmtuner import report  # noqa: E402
from llmtuner.adapters.openrouter import find_key  # noqa: E402
from llmtuner.cli import build_parser  # noqa: E402
from llmtuner.config import ConfigError, load_config  # noqa: E402
from llmtuner.store import _SCHEMA, Store  # noqa: E402

SRC, PY = tor.SRC, sys.executable
BIG_KEY = "sk-or-v1-" + "".join("abcdefghijklmnopqrstuvwxyz0123456789"[(i * 7) % 36] for i in range(48))


def key_pieces(key, size=8):
    return [key[i:i + size] for i in range(len(key) - size + 1)]


# --- 1. migration race --------------------------------------------------------

OPEN_SCRIPT = """\
import os, sys, time
from llmtuner.store import Store
while not os.path.exists(sys.argv[2]):
    time.sleep(0.001)
Store(sys.argv[1]).close()
"""


class MigrationRaceTests(unittest.TestCase):
    def make_old_store(self, home: Path):
        old_schema = _SCHEMA.replace("  meta TEXT,\n", "")
        self.assertNotEqual(old_schema, _SCHEMA)
        db = sqlite3.connect(home / "store.db")
        db.executescript(old_schema)
        db.commit()
        db.close()

    def test_eight_processes_open_an_old_store_together(self):
        env = dict(os.environ, PYTHONPATH=SRC)
        with tempfile.TemporaryDirectory() as tmp:
            for round_no in range(4):
                home = Path(tmp) / f"store{round_no}"
                home.mkdir()
                self.make_old_store(home)
                go = Path(tmp) / f"go{round_no}"
                procs = [subprocess.Popen([PY, "-c", OPEN_SCRIPT, str(home), str(go)], env=env,
                                          stderr=subprocess.PIPE, text=True) for _ in range(8)]
                time.sleep(0.5)  # let every process reach the starting line
                go.write_text("go")
                for proc in procs:
                    _, err = proc.communicate(timeout=60)
                    self.assertEqual(proc.returncode, 0, err)
                db = sqlite3.connect(home / "store.db")
                names = [r[1] for r in db.execute("PRAGMA table_info(cases)")]
                db.close()
                self.assertEqual(names.count("meta"), 1)

    def test_duplicate_column_error_counts_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.make_old_store(home)
            real_connect = sqlite3.connect

            # Another process adds the column between our check and our ALTER.
            class Racing(sqlite3.Connection):
                def execute(self, sql, *args):
                    if sql.startswith("BEGIN IMMEDIATE"):
                        raise sqlite3.OperationalError("duplicate column name: meta")
                    return super().execute(sql, *args)

            with mock.patch("sqlite3.connect", lambda *a, **k: real_connect(*a, factory=Racing, **k)):
                Store(home).close()  # must not raise


# --- 2-4. usage has three states ----------------------------------------------

UNIT_STUB = """\
import json, os, sys
text = sys.stdin.read().strip()
n = int(text[1:])
kind = sys.argv[1]
path = os.environ["LLMTUNER_META"]
if kind == "mixed":
    meta = {"cost": 0.01, "cost_unit": "credits" if n % 2 else "usd", "prompt_tokens": 5, "completion_tokens": 1}
elif kind == "usd":
    meta = {"cost": 0.01, "cost_unit": "usd"}
elif kind == "credits":
    meta = {"cost": 0.01, "cost_unit": "credits"}
elif kind == "nan":
    open(path, "w").write('{"cost": NaN, "cost_unit": "credits", "prompt_tokens": 3}')
    meta = None
elif kind == "huge":
    open(path, "w").write('{"cost": 1e999, "cost_unit": "credits", "prompt_tokens": 3}')
    meta = None
elif kind == "inf":
    open(path, "w").write('{"cost": -Infinity, "prompt_tokens": Infinity}')
    meta = None
if meta is not None:
    open(path, "w").write(json.dumps(meta))
print(text)
"""


class UnitCase(tcr.ProjectCase):
    N = 4

    def setUp(self):
        super().setUp()
        (self.root / "unit_stub.py").write_text(UNIT_STUB)

    def add_unit_target(self, name, kind):
        self.add_target(name, [PY, "unit_stub.py", kind])

    def use(self, run_id):
        return report.build_report(self.store, run_id)["usage"]


class UsageStateTests(UnitCase):
    def test_mixed_units_in_one_run(self):
        self.add_unit_target("m", "mixed")
        run_id = self.run_target("m")
        use = self.use(run_id)
        self.assertTrue(use["mixed_units"])
        self.assertEqual(use["cost_units"], ["credits", "usd"])
        self.assertIsNone(use["cost"])
        self.assertIsNone(use["cost_unit"])
        self.assertIsNone(use["cost_per_pass"])
        self.assertEqual(use["cases_with_cost"], 4)
        text = report.format_report(report.build_report(self.store, run_id))
        self.assertIn("cost reported in different units (credits, usd), so it is not added up", text)
        self.assertNotIn("not reported by the target", text)

    def test_single_unit_keeps_working(self):
        self.add_unit_target("c", "credits")
        use = self.use(self.run_target("c"))
        self.assertFalse(use["mixed_units"])
        self.assertEqual(use["cost_units"], ["credits"])
        self.assertAlmostEqual(use["cost"], 0.04)
        self.assertEqual(use["cost_unit"], "credits")

    def test_missing_unit_counts_as_unknown(self):
        use = report.usage([{"meta": {"cost": 1, "cost_unit": "usd"}, "duration_s": 1.0},
                            {"meta": {"cost": 2}, "duration_s": 1.0}], 2)
        self.assertTrue(use["mixed_units"])
        self.assertEqual(use["cost_units"], ["unknown", "usd"])
        self.assertIsNone(use["cost"])

    def test_non_finite_costs_are_not_reported(self):
        for value in (float("nan"), float("inf"), -float("inf"), 10 ** 400):
            use = report.usage([{"meta": {"cost": value, "prompt_tokens": value}, "duration_s": 1.0}], 1)
            self.assertIsNone(use["cost"], value)
            self.assertEqual(use["cases_with_cost"], 0)
            self.assertIsNone(use["prompt_tokens"])
            self.assertFalse(use["mixed_units"])

    def test_nan_and_1e999_never_reach_the_store(self):
        for kind in ("nan", "huge", "inf"):
            self.add_unit_target(kind, kind)
            run_id = self.run_target(kind)
            for case in self.store.get_cases(run_id):
                self.assertIsNone(case["meta"], kind)
            use = self.use(run_id)
            self.assertIsNone(use["cost"])
            text = report.format_report(report.build_report(self.store, run_id))
            self.assertIn("cost not reported by the target", text)
            json.dumps(report.json_safe(report.build_report(self.store, run_id)), allow_nan=False)

    def test_one_side_of_tokens_unknown(self):
        use = report.usage([{"meta": {"prompt_tokens": 710}, "duration_s": 1.0}], 1)
        self.assertEqual(use["prompt_tokens"], 710)
        self.assertIsNone(use["completion_tokens"])
        line = report._usage_line(use, 1)
        self.assertIn("tokens 710 in, out not reported", line)
        self.assertNotIn(" 0 out", line)
        both = report._usage_line(report.usage([{"meta": {"prompt_tokens": 710, "completion_tokens": 33},
                                                 "duration_s": 1.0}], 1), 1)
        self.assertIn("tokens 710 in, 33 out", both)
        only_out = report._usage_line(report.usage([{"meta": {"completion_tokens": 4}, "duration_s": 1.0}], 1), 1)
        self.assertIn("tokens in not reported, 4 out", only_out)


class RankUnitTests(UnitCase):
    def rank(self):
        return report.rank(self.store, "costdemo", "main")

    def test_different_units_are_not_ordered_by_cost(self):
        # Same pass rate. By cost "usd_t" (0.04 usd) would beat "cred_t" (0.04 credits) or not: no way to say.
        self.add_unit_target("zz_usd", "usd")
        self.add_unit_target("aa_credits", "credits")
        first = self.run_target("zz_usd")
        second = self.run_target("aa_credits")
        result = self.rank()
        self.assertEqual([r["run"] for r in result["rows"]], [first, second])  # pass rate, then run id
        self.assertIn("different units", result["cost_note"])
        self.assertIn("different units", report.format_rank(result))

    def test_mixed_row_blocks_cost_ordering(self):
        self.add_unit_target("m", "mixed")
        self.add_unit_target("c", "credits")
        self.run_target("m")
        self.run_target("c")
        result = self.rank()
        self.assertIn("cost_note", result)
        self.assertIsNone(result["rows"][0]["usage"]["cost_per_pass"])

    def test_same_unit_has_no_note(self):
        self.add_unit_target("a", "credits")
        self.add_unit_target("b", "credits")
        self.run_target("a")
        self.run_target("b")
        self.assertNotIn("cost_note", self.rank())


# --- 5. rank selection --------------------------------------------------------


class RankSelectionTests(tcr.ProjectCase):
    N = 40

    def rank(self):
        return report.rank(self.store, "costdemo", "main")

    def test_target_with_newer_limit_run_is_ranked_by_its_full_run(self):
        self.add_rank_target("a", fails="c1", cost="0.01")
        self.add_rank_target("b", fails="c1,c2", cost="0.01")
        full = self.run_target("a")
        self.run_target("b")
        self.run_target("a", limit=5)  # newest run of "a", but only 5 cases
        result = self.rank()
        self.assertEqual({r["target"]: r["run"] for r in result["rows"]}["a"], full)
        self.assertEqual(result["skipped"], [])
        self.assertTrue(all(r["n"] == 40 for r in result["rows"]))

    def test_target_with_only_a_limit_run_is_skipped(self):
        self.add_rank_target("a", fails="", cost="0.01")
        self.add_rank_target("b", fails="c1", cost="0.01")
        self.run_target("a")
        limited = self.run_target("b", limit=10)
        result = self.rank()
        self.assertEqual([r["target"] for r in result["rows"]], ["a"])
        self.assertEqual(result["skipped"], [{"run": limited, "target": "b",
                                              "reason": "ran on a different set of cases (for example with --limit)"}])

    def test_leader_on_20_cases_cannot_contradict_a_run_on_40(self):
        # Seen before: 70% on 20 cases led a 67.5% run on 40 cases, and the two were then compared.
        fails_20 = ",".join(f"c{i}" for i in range(1, 7))  # 14 of the first 20 pass = 70%
        fails_40 = ",".join(f"c{i}" for i in range(1, 14))  # 27 of 40 pass = 67.5%
        self.add_rank_target("quick", fails=fails_20, cost="0.01")
        self.add_rank_target("full", fails=fails_40, cost="0.01")
        quick = self.run_target("quick", limit=20)
        full = self.run_target("full")
        result = self.rank()
        self.assertEqual([(r["target"], r["run"], r["n"]) for r in result["rows"]], [("full", full, 40)])
        self.assertEqual(result["rows"][0]["vs_leader"], "leader")
        self.assertEqual([s["run"] for s in result["skipped"]], [quick])

    def test_reference_group_is_the_biggest_even_if_older(self):
        self.add_rank_target("a", fails="", cost="0.01")
        big = self.run_target("a")
        self.run_target("a", limit=3)
        self.assertEqual([r["run"] for r in self.rank()["rows"]], [big])

    def test_vs_leader_maps_better_too(self):
        self.add_rank_target("good", fails="", cost="0.01")
        self.add_rank_target("bad", fails=",".join(f"c{i}" for i in range(1, 21)), cost="0.01")
        self.run_target("good")
        self.run_target("bad")
        rows = {r["target"]: r["vs_leader"] for r in self.rank()["rows"]}
        self.assertEqual(rows["bad"], "measurably worse than the leader")
        # The verdict mapping itself, for a candidate that beats the leader.
        with mock.patch.object(report, "compare_runs", return_value={"verdict": "candidate is better",
                                                                    "p_value": 0.001}):
            rows = {r["target"]: r["vs_leader"] for r in self.rank()["rows"]}
        self.assertEqual(rows["bad"], "measurably better than the leader")

    def test_older_dataset_and_graders_reasons_still_work(self):
        self.add_rank_target("old", fails="", cost="0.01")
        self.add_rank_target("new", fails="", cost="0.01")
        self.run_target("old")
        self.write_cases(41)
        self.run_target("new")
        self.assertEqual([s["reason"] for s in self.rank()["skipped"]], ["ran on a different version of the cases"])

    def test_ranking_follows_the_dataset_after_it_shrinks(self):
        # Seen before: after cases were removed, rank kept showing the runs on the old, bigger file.
        self.add_rank_target("a", fails="", cost="0.01")
        self.add_rank_target("b", fails="c1", cost="0.01")
        self.run_target("a")
        self.run_target("b")
        self.write_cases(30)
        new_a = self.run_target("a")
        new_b = self.run_target("b")
        result = self.rank()
        self.assertEqual(sorted(r["run"] for r in result["rows"]), sorted([new_a, new_b]))
        self.assertEqual({r["n"] for r in result["rows"]}, {30})
        self.assertEqual(result["skipped"], [])


# --- 6-9. adapter ----------------------------------------------------------------


class RawStub:
    """Calls `fn(handler)` for every request of any method; records every request line."""

    def __init__(self, fn):
        self.hits = []
        self.stop = threading.Event()
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def handle_any(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                stub.hits.append((self.command, self.path, dict(self.headers)))
                fn(self, stub)

            do_POST = do_GET = do_PUT = handle_any

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01},
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


class RedirectTests(tor.AdapterCase):
    def test_redirects_are_never_followed(self):
        for code in (301, 302, 307, 308):
            def redirect(handler, stub):
                handler.send_response(code)
                handler.send_header("Location", f"http://localhost:{stub.port}/steal")
                handler.send_header("Content-Length", "0")
                handler.end_headers()

            with self.subTest(code=code), RawStub(redirect) as stub:
                proc = self.run_adapter(stub.url)
                self.assert_clean_failure(proc)
                self.assertEqual(len(stub.hits), 1, stub.hits)  # no second request, to localhost or anywhere
                self.assertIn(f"the API answered with a redirect (HTTP {code}) to another address; not followed",
                              proc.stderr)
                self.assertNotIn(tor.FAKE, proc.stderr)


class MaskBeforeCutTests(tor.AdapterCase):
    def serve(self, offset, as_json):
        filler = "x" * offset

        def reply(handler, stub):
            text = f"{filler}{BIG_KEY} and more"
            data = json.dumps({"error": {"message": text}}).encode() if as_json else text.encode()
            handler.send_response(500)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)

        return RawStub(reply)

    def test_key_straddling_the_cut_does_not_survive(self):
        for offset in (0, 270, 280, 285, 290, 295, 298, 299, 300):
            for as_json in (True, False):
                with self.subTest(offset=offset, json=as_json), self.serve(offset, as_json) as stub:
                    proc = self.run_adapter(stub.url, key=BIG_KEY, stdin="q")
                    self.assert_clean_failure(proc)
                    for piece in key_pieces(BIG_KEY):
                        self.assertNotIn(piece, proc.stderr)


class StoreMaskTests(tcr.ProjectCase):
    N = 1

    def test_stored_run_has_no_piece_of_the_key(self):
        helper = MaskBeforeCutTests("test_key_straddling_the_cut_does_not_survive")
        self.add_target("or", ["{python}", "-m", "llmtuner.adapters.openrouter", "--model", "m",
                               "--base-url", "http://127.0.0.1:9", "--timeout", "20"])
        home = self.tmp / "home"
        home.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER_", "LLMTUNER_"))}
        env.update({"OPENROUTER_API_KEY": BIG_KEY, "PYTHONPATH": SRC, "HOME": str(home)})
        for as_json in (True, False):
            for offset in (275, 290, 297):
                with self.subTest(offset=offset, json=as_json), helper.serve(offset, as_json) as stub:
                    self.add_target("or", ["{python}", "-m", "llmtuner.adapters.openrouter", "--model", "m",
                                           "--base-url", stub.url, "--timeout", "20"])
                    with mock.patch.dict(os.environ, env, clear=True):
                        run_id = self.run_target("or")
                    case = self.store.get_cases(run_id)[0]
                    self.assertFalse(case["passed"])
                    self.assertIn("500", case["error"])
        self.store.close()
        data = b"".join(p.read_bytes() for p in sorted(self.store_dir.iterdir()) if p.is_file())
        for piece in key_pieces(BIG_KEY):
            self.assertNotIn(piece.encode(), data)


class DeadlineTests(tor.AdapterCase):
    def drip(self, handler, stub):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", "100000")
        handler.end_headers()
        try:
            while not stub.stop.is_set():
                handler.wfile.write(b" ")
                handler.wfile.flush()
                stub.stop.wait(1)
        except OSError:
            pass

    def test_dripping_body_hits_the_one_deadline(self):
        with RawStub(self.drip) as stub:
            started = time.monotonic()
            proc = self.run_adapter(stub.url, "--timeout", "2")
            elapsed = time.monotonic() - started
        self.assert_clean_failure(proc)
        self.assertIn("timed out", proc.stderr)
        self.assertLess(elapsed, 5, elapsed)  # 2s deadline plus interpreter start-up

    def test_dripping_headers_hit_the_one_deadline(self):
        def drip_headers(handler, stub):
            try:
                for byte in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"a" * 100:
                    handler.wfile.write(bytes([byte]))
                    handler.wfile.flush()
                    stub.stop.wait(0.5)
            except OSError:
                pass

        with RawStub(drip_headers) as stub:
            started = time.monotonic()
            proc = self.run_adapter(stub.url, "--timeout", "2")
            elapsed = time.monotonic() - started
        self.assert_clean_failure(proc)
        self.assertIn("timed out", proc.stderr)
        self.assertLess(elapsed, 5, elapsed)

    def test_chunked_reply_is_read(self):
        body = json.dumps(tor.ok_body("chunky")).encode()

        def chunked(handler, stub):
            handler.protocol_version = "HTTP/1.1"
            handler.send_response(200)
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            for i in range(0, len(body), 20):
                part = body[i:i + 20]
                handler.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            handler.wfile.write(b"0\r\n\r\n")

        with RawStub(chunked) as stub:
            proc = self.run_adapter(stub.url, "--timeout", "10")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "chunky")

    def test_silent_server_hits_the_deadline(self):
        def silent(handler, stub):
            stub.stop.wait(30)

        with RawStub(silent) as stub:
            started = time.monotonic()
            proc = self.run_adapter(stub.url, "--timeout", "2")
            elapsed = time.monotonic() - started
        self.assert_clean_failure(proc)
        self.assertIn("timed out", proc.stderr)
        self.assertLess(elapsed, 5, elapsed)

    def test_retries_share_the_deadline(self):
        def slow_503(handler, stub):
            stub.stop.wait(1.2)
            data = b'{"error": {"message": "busy"}}'
            handler.send_response(503)
            handler.send_header("Retry-After", "0")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            with contextlib.suppress(OSError):  # the client may have given up already
                handler.wfile.write(data)

        with RawStub(slow_503) as stub:
            started = time.monotonic()
            proc = self.run_adapter(stub.url, "--timeout", "2")
            elapsed = time.monotonic() - started
        self.assert_clean_failure(proc)
        self.assertLess(elapsed, 5, elapsed)  # three tries of 1.2s each would not fit in 2s
        self.assertLess(len(stub.hits), 3)

    def test_huge_retry_after_returns_within_the_deadline(self):
        def busy(handler, stub):
            data = b'{"error": {"message": "slow down"}}'
            handler.send_response(429)
            handler.send_header("Retry-After", "99999")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)

        with RawStub(busy) as stub:
            started = time.monotonic()
            proc = self.run_adapter(stub.url, "--timeout", "3")
            elapsed = time.monotonic() - started
        self.assert_clean_failure(proc)
        self.assertIn("HTTP 429", proc.stderr)
        self.assertLess(elapsed, 5, elapsed)
        self.assertEqual(len(stub.hits), 1)


class OddsAndEndsTests(tor.AdapterCase):
    def test_malformed_base_url_is_a_clean_error(self):
        for url in ("http://[::1", "http://host:notaport", "http://exa mple.com"):
            with self.subTest(url=url):
                proc = self.run_adapter(url)
                self.assert_clean_failure(proc)
                self.assertTrue(proc.stderr.startswith("error:"), proc.stderr)

    def test_bad_timeout_values_are_refused(self):
        for value in ("nan", "inf", "1e999", "0", "-1", "abc"):
            with self.subTest(value=value):
                proc = self.run_adapter("http://127.0.0.1:9", "--timeout", value)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)

    def test_tiny_timeout_is_a_clean_error(self):
        proc = self.run_adapter("http://127.0.0.1:9", "--timeout", "0.001")
        self.assert_clean_failure(proc)

    def test_key_with_line_break_is_not_echoed(self):
        proc = self.run_adapter("http://127.0.0.1:9", key="SECRETPART1\nSECRETPART2")
        self.assert_clean_failure(proc)
        self.assertNotIn("SECRETPART", proc.stderr)
        proc = self.run_adapter("http://127.0.0.1:9", key="ключ-SECRET")
        self.assert_clean_failure(proc)
        self.assertNotIn("SECRET", proc.stderr)

    def test_error_json_without_error_field_is_not_none(self):
        with tor.Stub((500, {}, {"detail": "boom"})) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertNotIn("None", proc.stderr)
        self.assertIn("boom", proc.stderr)

    def test_http_base_url_ignores_proxy_environment(self):
        with tor.Stub() as stub, RawStub(lambda h, s: h.send_error(502)) as proxy:
            proc = self.run_adapter(stub.url, env_extra={"http_proxy": proxy.url, "HTTP_PROXY": proxy.url})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(stub.requests), 1)
        self.assertEqual(proxy.hits, [])

    def test_env_file_comments(self):
        cases = {
            "OPENROUTER_API_KEY=abc # note": "abc",
            "OPENROUTER_API_KEY=abc\t# note": "abc",
            "OPENROUTER_API_KEY=abc#frag": "abc#frag",
            "OPENROUTER_API_KEY='abc # kept'": "abc # kept",
            'export OPENROUTER_API_KEY="abc#kept"': "abc#kept",
            "OPENROUTER_API_KEY= # nothing": None,
            'OPENROUTER_API_KEY="abc" # note': "abc",
            "OPENROUTER_API_KEY='abc'   # note": "abc",
        }
        for line, expected in cases.items():
            with self.subTest(line=line), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / ".env").write_text(line + "\n")
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("OPENROUTER_API_KEY", None)
                    self.assertEqual(find_key(Path(tmp), Path(tmp)), expected)


# --- 15 (parts) ---------------------------------------------------------------------


class ConfigTimeoutTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "llmtuner.toml").write_text('project = "p"\n' + text)
            return load_config(tmp)

    def test_target_timeout_must_be_finite_and_at_most_a_day(self):
        for value in ("inf", "nan", "86401", "1e9"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(f'[targets.t]\ncommand = ["x"]\ntimeout = {value}\n')
        self.assertEqual(self.load('[targets.t]\ncommand = ["x"]\ntimeout = 86400\n').targets["t"].timeout, 86400)

    def test_huge_integer_timeout_is_a_config_error(self):
        with self.assertRaises(ConfigError):
            self.load('[targets.t]\ncommand = ["x"]\ntimeout = ' + "9" * 400 + "\n")

    def test_judge_timeout_must_be_finite_and_at_most_a_day(self):
        for value in ("inf", "nan", "100000"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(f'[graders.j]\ntype = "judge"\nquestion = "q"\ncommand = ["x"]\ntimeout = {value}\n')
        self.assertTrue(math.isfinite(self.load(
            '[graders.j]\ntype = "judge"\nquestion = "q"\ncommand = ["x"]\ntimeout = 60\n'
        ).graders["j"].options["timeout"]))


class PortTests(unittest.TestCase):
    def parse(self, port):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                return build_parser().parse_args(["ui", "--port", port]).port, err.getvalue()
            except SystemExit as exc:
                return exc.code, err.getvalue()

    def test_port_range(self):
        for port in ("0", "8765", "65535"):
            self.assertEqual(self.parse(port)[0], int(port))
        for port in ("-1", "65536", "99999", "abc"):
            code, err = self.parse(port)
            self.assertEqual(code, 2, port)
            self.assertIn("port", err)


@unittest.skipUnless(hasattr(__import__("signal"), "setitimer"), "needs interval timers")
class CallerTimerTests(unittest.TestCase):
    def test_a_callers_own_timer_survives_the_call(self):
        # Seen before: used as a library, complete() switched off a timer the caller had running.
        import signal

        from llmtuner.adapters import openrouter

        fired = []
        mine = lambda signum, frame: fired.append(signum)  # noqa: E731
        previous = signal.signal(signal.SIGALRM, mine)
        try:
            signal.setitimer(signal.ITIMER_REAL, 30.0)
            with mock.patch.object(openrouter, "_complete", return_value={"ok": True}):
                self.assertEqual(openrouter.complete(timeout=2.0), {"ok": True})
            left, _ = signal.getitimer(signal.ITIMER_REAL)
            self.assertGreater(left, 20.0)
            self.assertIs(signal.getsignal(signal.SIGALRM), mine)
            self.assertEqual(fired, [])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    unittest.main()
