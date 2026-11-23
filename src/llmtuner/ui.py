"""The local web UI: a read-mostly window onto the store, served on 127.0.0.1 only.

It holds no eval or statistics logic. Every number comes from `report` and `stats`
functions and is passed through `report.json_safe`.

Safety model:
- bound to 127.0.0.1; any request whose Host header is not this server's own is refused
  (this blocks DNS rebinding);
- every POST needs the per-launch token in `X-LLMTuner-Token` and, when the browser sends
  an Origin header, that origin must be this server's own;
- the labelling queue never carries a grader's verdict, so a person labelling cannot see it.
"""

from __future__ import annotations

import hmac
import json
import random
import re
import secrets
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlsplit

from llmtuner import report, stats
from llmtuner.config import ConfigError
from llmtuner.dataset import DatasetError
from llmtuner.store import CASE_LABEL, Store, StoreError

MAX_BODY = 64 * 1024
MAX_TEXT = 20_000
MAX_CASES = 500
MAX_QUEUE = 50
MAX_ID_DIGITS = 18
MAX_CASE_ID = 1000
MAX_GRADER = 1000
MAX_NOTE = 10_000
REQUEST_TIMEOUT = 10.0  # seconds a client may take to send a request; tests lower it
SPLITS = ("test", "dev", "holdout", "all")
_ID = re.compile(r"[0-9]{1,18}")
_PAGE = Path(__file__).with_name("ui_page.html")


class BadRequest(Exception):
    pass


class NotFound(Exception):
    pass


class Running(NamedTuple):
    httpd: ThreadingHTTPServer
    url: str
    token: str


def _clip(text, truncated: dict, key: str):
    """Cap a text field at MAX_TEXT characters, recording its full length when it was cut."""
    if isinstance(text, str) and len(text) > MAX_TEXT:
        truncated[key] = len(text)
        return text[:MAX_TEXT]
    return text


def _int(value, name: str) -> int:
    """Parse an id or limit from a query string: ASCII digits only, at most 18 of them."""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise BadRequest(f"{name} must be a whole number")
    return int(value)


def _body_run(body: dict) -> int:
    run = body.get("run")
    if type(run) is not int or not 0 < run < 10**MAX_ID_DIGITS:
        raise BadRequest("run must be a positive whole number")
    return run


def _body_text(body: dict, name: str, limit: int, required: bool = False) -> str | None:
    value = body.get(name)
    if value is None:
        if required:
            raise BadRequest(f"{name} must be a non-empty string")
        return None
    if not isinstance(value, str) or (required and not value):
        raise BadRequest(f"{name} must be {'a non-empty ' if required else ''}text")
    if len(value) > limit:
        raise BadRequest(f"{name} must be at most {limit:,} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise BadRequest(f"{name} contains characters that cannot be stored") from None
    return value


def _query_int(query: dict, name: str, default: int | None = None, low: int = 1, high: int = 10**9) -> int:
    values = query.get(name)
    if not values:
        if default is None:
            raise BadRequest(f"missing '{name}'")
        return default
    value = _int(values[0], name)
    if not low <= value <= high:
        raise BadRequest(f"{name} must be between {low} and {high}")
    return value


def _query_str(query: dict, name: str, default: str | None = None) -> str:
    values = query.get(name)
    if not values or not values[0]:
        if default is None:
            raise BadRequest(f"missing '{name}'")
        return default
    return values[0]


# --- API handlers (each takes an open Store and returns a JSON-able value) ------


def api_projects(store: Store, query: dict):
    projects: dict[str, dict] = {}
    for row in store.list_datasets():
        project = projects.setdefault(row["project"], {"project": row["project"], "datasets": {}, "latest_run": 0})
        project["latest_run"] = max(project["latest_run"], row["latest_run"])
        dataset = project["datasets"].setdefault(row["dataset"], {"dataset": row["dataset"], "splits": [], "runs": 0})
        dataset["splits"].append({"split": row["split"], "runs": row["runs"]})
        dataset["runs"] += row["runs"]
    out = sorted(projects.values(), key=lambda p: -p["latest_run"])
    for project in out:
        project["datasets"] = list(project["datasets"].values())
    return {"projects": out}


_ROW_KEYS = ("id", "project", "dataset", "target", "model", "split", "note", "started_at",
             "finished_at", "trial_group", "trial", "trials_planned")


def _group_row(store: Store, first_id: int, baseline_id: int | None) -> dict:
    """One list row for a whole group of repeated trials, represented by its first run.

    Every number comes from `report.trial_summary`, the same value `build_report` returns under
    "trials", so the row and the run page can never disagree.
    """
    first = store.get_run(first_id)
    summary = report.trial_summary(store, first_id)
    use = summary["usage"]
    row = {k: first[k] for k in _ROW_KEYS}
    # An unfinished group has no single finish time; the page shows "k of K finished" instead.
    row["finished_at"] = max(r["finished_at"] for r in store.group_runs(first_id)) if summary["complete"] else None
    row.update(
        trials=summary["planned"],
        trials_finished=summary["finished"],
        trial_runs=summary["runs"],
        n=summary["n"],
        n_passed=summary["passed_all"]["n_passed"],
        pass_rate=summary["passed_all"]["rate"],
        cost_per_pass=use["cost_per_pass"],
        cost_unit=use["cost_unit"],
        cost_units=use.get("cost_units", []),
        mixed_units=bool(use.get("mixed_units", False)),
        is_baseline=baseline_id in summary["runs"],
    )
    return row


def api_runs(store: Store, query: dict):
    project = query.get("project", [None])[0] or None
    limit = _query_int(query, "limit", default=50, high=500)
    # A group of trials is several stored runs but one row, so `limit` counts rows: fetch more
    # runs until enough rows are collected or the store has no more. Only the rows that are
    # kept are then summarised.
    fetch = limit
    while True:
        runs = store.list_runs(project, fetch)
        picked, seen = [], set()
        for run in runs:  # newest first; a group sits where its newest trial is
            group = run["trial_group"]
            if group is not None:
                if group in seen:
                    continue
                seen.add(group)
            picked.append(run)
            if len(picked) == limit:
                break
        if len(picked) == limit or len(runs) < fetch:
            break
        fetch *= 2
    baselines: dict[tuple, int | None] = {}
    rows = []
    for run in picked:
        key = (run["project"], run["dataset"])
        if key not in baselines:
            baselines[key] = store.get_baseline(*key)
        if run["trial_group"] is not None:
            rows.append(_group_row(store, run["trial_group"], baselines[key]))
            continue
        cases = store.get_cases(run["id"])
        passed = [bool(c["passed"]) for c in cases]
        use = report.usage(cases, sum(passed))
        run["trials"] = 1
        run["pass_rate"] = stats.pass_rate(passed).as_dict()
        run["cost_per_pass"] = use["cost_per_pass"]
        run["cost_unit"] = use["cost_unit"]
        run["cost_units"] = use.get("cost_units", [])
        run["mixed_units"] = bool(use.get("mixed_units", False))
        run["is_baseline"] = baselines[key] == run["id"]
        rows.append(run)
    return {"runs": rows, "project": project, "limit": limit}


# The page does not know the project folder, so it cannot read the change log; say that, rather than
# the report's "change log was not available", which would read as a fault.
NOT_CHECKED = "changes to cases were not checked against the change log here (llmtuner report and compare do)"


def _not_checked(comparison: dict | None) -> None:
    if comparison:
        comparison["warnings"] = [NOT_CHECKED if w == report.LOG_UNAVAILABLE else w for w in comparison["warnings"]]


def api_report(store: Store, run_id: int, query: dict):
    result = report.build_report(store, run_id)
    _not_checked(result["baseline"])
    return result


def api_cases(store: Store, run_id: int, query: dict):
    store.get_run(run_id)
    failed_only = query.get("failed", ["0"])[0] in ("1", "true")
    only_case = query.get("case", [None])[0]
    grades: dict[str, list[dict]] = {}
    for g in store.get_grades(run_id):
        grades.setdefault(g["case_id"], []).append(g)
    cases = store.get_cases(run_id)
    if only_case is not None:
        cases = [c for c in cases if c["case_id"] == only_case]
        if not cases:
            raise StoreError(f"run {run_id} has no case '{only_case}'")
    if failed_only:
        cases = [c for c in cases if not c["passed"]]
    total = len(cases)
    out = []
    for c in cases[:MAX_CASES]:
        cut: dict[str, int] = {}
        item = {
            "case_id": c["case_id"],
            "passed": bool(c["passed"]),
            "input": _clip(c["input"], cut, "input"),
            "expected": _clip(c["expected"], cut, "expected"),
            "context": _clip(c["context"], cut, "context"),
            "output": _clip(c["output"], cut, "output"),
            "error": _clip(c["error"], cut, "error"),
            "exit_code": c["exit_code"],
            "timed_out": bool(c["timed_out"]),
            "duration_s": c["duration_s"],
            "meta": c["meta"],
            "grades": [
                {"grader": g["grader"], "passed": bool(g["passed"]), "error": bool(g["error"]),
                 "detail": _clip(g["detail"], cut, f"detail:{g['grader']}")}
                for g in grades.get(c["case_id"], [])
            ],
        }
        item["truncated"] = cut
        out.append(item)
    return {"run": run_id, "total": total, "shown": len(out), "max_text": MAX_TEXT, "cases": out}


def api_compare(store: Store, query: dict):
    result = report.compare_runs(store, _query_int(query, "baseline"), _query_int(query, "candidate"))
    _not_checked(result)
    return result


def api_rank(store: Store, query: dict):
    split = _query_str(query, "split", "test")
    if split not in SPLITS:
        raise BadRequest("split must be test, dev, holdout or all")
    return report.rank(store, _query_str(query, "project"), _query_str(query, "dataset"), split)


def _resolve_grader(run: dict, name: str | None) -> tuple[str, dict | None]:
    grader = name or CASE_LABEL
    spec = next((g for g in run["graders"] if g["name"] == grader), None)
    if name and name != CASE_LABEL and spec is None:
        names = ", ".join(g["name"] for g in run["graders"])
        raise StoreError(f"run {run['id']} has no grader '{name}' (it has: {names})")
    return grader, spec


def api_queue(store: Store, run_id: int, query: dict):
    """Cases still to label. Carries ONLY case_id, input, context, expected and output."""
    run = store.get_run(run_id)
    grader, spec = _resolve_grader(run, query.get("grader", [None])[0] or None)
    labels = store.get_labels(run_id)
    done = {l["case_id"] for l in labels if l["grader"] == grader}
    cases = [c for c in store.get_cases(run_id) if c["error"] is None]
    todo = [c for c in cases if c["case_id"] not in done]
    random.Random(run_id).shuffle(todo)  # same fixed per-run order as `llmtuner label`
    question = "Is this output acceptable overall?"
    if spec and spec["type"] == "judge":
        question = f"The grader asks: \"{spec['question']}\"  Does this output PASS that check?"
    elif spec:
        question = f"Should this output PASS the '{spec['name']}' check?"
    items, truncated_any = [], False
    for c in todo[:MAX_QUEUE]:
        cut: dict[str, int] = {}
        items.append({
            "case_id": c["case_id"],
            "input": _clip(c["input"], cut, "input"),
            "context": _clip(c["context"], cut, "context"),
            "expected": _clip(c["expected"], cut, "expected"),
            "output": _clip(c["output"], cut, "output"),
        })
        truncated_any = truncated_any or bool(cut)
    return {
        "run": run_id,
        "grader": None if grader == CASE_LABEL else grader,
        "graders": [g["name"] for g in run["graders"]],
        "question": question,
        "done": len(done & {c["case_id"] for c in cases}),
        "remaining": len(todo),
        "items": items,
        "truncated": truncated_any,
        "max_text": MAX_TEXT,
    }


def api_post_label(store: Store, body: dict):
    # Validate everything before the store is touched.
    run_id = _body_run(body)
    case_id = _body_text(body, "case_id", MAX_CASE_ID, required=True)
    passed = body.get("passed")
    if not isinstance(passed, bool):
        raise BadRequest("passed must be true or false")
    note = _body_text(body, "note", MAX_NOTE)
    grader_name = _body_text(body, "grader", MAX_GRADER)
    grader, _ = _resolve_grader(store.get_run(run_id), grader_name)
    store.set_label(run_id, case_id, passed, grader, note or None)
    return {"run": run_id, "case_id": case_id, "grader": None if grader == CASE_LABEL else grader,
            "passed": passed, "note": note or None}


def api_post_baseline(store: Store, body: dict):
    run_id = _body_run(body)
    run = store.set_baseline(run_id)
    # For a trial of a group the store keeps the group's first run; report that one.
    return {"project": run["project"], "dataset": run["dataset"],
            "baseline_run": store.get_baseline(run["project"], run["dataset"])}


_ROUTES_GET = [
    (re.compile(r"/api/projects"), api_projects),
    (re.compile(r"/api/runs"), api_runs),
    (re.compile(r"/api/runs/([0-9]{1,18})"), api_report),
    (re.compile(r"/api/runs/([0-9]{1,18})/cases"), api_cases),
    (re.compile(r"/api/runs/([0-9]{1,18})/queue"), api_queue),
    (re.compile(r"/api/compare"), api_compare),
    (re.compile(r"/api/rank"), api_rank),
]
_ROUTES_POST = [
    (re.compile(r"/api/label"), api_post_label),
    (re.compile(r"/api/baseline"), api_post_baseline),
]


# --- server --------------------------------------------------------------------


def _make_handler(store_home, token: str, page: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "llmtuner-ui"
        sys_version = ""
        protocol_version = "HTTP/1.0"

        def setup(self):
            self.timeout = REQUEST_TIMEOUT  # read at connection time so a test can lower it
            super().setup()

        def log_message(self, format, *args):  # quiet
            pass

        # -- helpers
        def _port(self) -> int:
            return self.server.server_address[1]

        def _allowed_hosts(self) -> set[str]:
            return {f"127.0.0.1:{self._port()}", f"localhost:{self._port()}"}

        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data):
            self._send(status, self._encode(data), "application/json; charset=utf-8")

        @staticmethod
        def _encode(data) -> bytes:
            # ensure_ascii: a lone surrogate becomes a \\u escape instead of an encode error.
            text = json.dumps(report.json_safe(data), ensure_ascii=True, allow_nan=False, default=str)
            return text.encode("ascii")

        def _guarded(self, work):
            """Run one request; whatever goes wrong, answer (500) or close, never leave a thread hanging."""
            try:
                work()
            except OSError:  # client went away or timed out; nothing to answer
                self.close_connection = True
            except Exception:
                traceback.print_exc()
                try:
                    self._error(500, "internal error; see the terminal running llmtuner ui")
                except OSError:
                    self.close_connection = True

        def _error(self, status: int, message: str):
            self._json(status, {"error": message})

        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") in self._allowed_hosts():
                return True
            self._error(403, "this server only answers requests addressed to 127.0.0.1 or localhost")
            return False

        # -- verbs
        def do_GET(self):
            self._guarded(self._get)

        def do_POST(self):
            self._guarded(self._post)

        def _get(self):
            if not self._host_ok():
                return
            url = urlsplit(self.path)
            if url.path in ("/", "/index.html"):
                return self._page()
            query = parse_qs(url.query)
            for pattern, func in _ROUTES_GET:
                match = pattern.fullmatch(url.path)
                if match:
                    args = tuple(_int(g, "id") for g in match.groups())
                    return self._call(func, args, query)
            self._error(404, "not found")

        def _post(self):
            if not self._host_ok():
                return
            supplied = self.headers.get("X-LLMTuner-Token", "")
            if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
                return self._error(403, "missing or wrong token; reload the page")
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{h}" for h in self._allowed_hosts()}:
                return self._error(403, "request came from another origin")
            url = urlsplit(self.path)
            route = next((f for p, f in _ROUTES_POST if p.fullmatch(url.path)), None)
            if route is None:
                return self._error(404, "not found")
            lengths = self.headers.get_all("Content-Length") or []
            if len(lengths) != 1 or not _ID.fullmatch(lengths[0]):
                return self._error(400, "exactly one numeric Content-Length header is required")
            length = int(lengths[0])
            if length > MAX_BODY:
                return self._error(413, f"request body is over {MAX_BODY // 1024} KB")
            raw = self._read_body(length)
            if raw is None:
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, RecursionError):
                return self._error(400, "body must be JSON")
            if not isinstance(body, dict):
                return self._error(400, "body must be a JSON object")
            self._call(route, (), body)

        def _read_body(self, length: int) -> bytes | None:
            """Read exactly `length` bytes within REQUEST_TIMEOUT in total, or answer and give up."""
            deadline = time.monotonic() + REQUEST_TIMEOUT
            chunks, got = [], 0
            try:
                while got < length:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError
                    self.connection.settimeout(left)
                    chunk = self.rfile.read1(length - got)
                    if not chunk:
                        self.close_connection = True
                        self._error(400, "request body is shorter than its Content-Length")
                        return None
                    chunks.append(chunk)
                    got += len(chunk)
            except TimeoutError:  # socket.timeout is an alias since Python 3.10
                self.close_connection = True
                try:
                    self._error(408, "timed out waiting for the request body")
                except OSError:
                    pass
                return None
            self.connection.settimeout(REQUEST_TIMEOUT)  # leave a full timeout for writing the reply
            return b"".join(chunks)

        def _call(self, func, ids: tuple, arg):
            try:
                with Store(store_home) as store:
                    result = func(store, *ids, arg)
                payload = self._encode(result)
            except BadRequest as exc:
                return self._error(400, str(exc))
            except OverflowError:
                return self._error(404, "no such run")
            except (StoreError, NotFound) as exc:
                return self._error(404, str(exc))
            except (ConfigError, DatasetError) as exc:
                return self._error(400, str(exc))
            except Exception:  # never leak a traceback to the page; it goes to the terminal
                traceback.print_exc()
                return self._error(500, "internal error; see the terminal running llmtuner ui")
            self._send(200, payload, "application/json; charset=utf-8")

        def _page(self):
            nonce = secrets.token_urlsafe(16)
            html = page.replace("__NONCE__", nonce).replace("__TOKEN__", token)
            csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8", {"Content-Security-Policy": csp})

    return Handler


def make_server(store_home=None, port: int = 8765) -> Running:
    """Create the server without serving. Raises OSError when the port is busy."""
    token = secrets.token_urlsafe(32)
    page = _PAGE.read_text(encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(store_home, token, page))
    httpd.daemon_threads = True
    return Running(httpd, f"http://127.0.0.1:{httpd.server_address[1]}/", token)


def main(store_home=None, port: int = 8765, open_browser: bool = True) -> int:
    if type(port) is not int or not 0 <= port <= 65535:
        print(f"error: port must be a whole number from 0 to 65535, not {port!r}", file=sys.stderr)
        return 1
    try:
        running = make_server(store_home, port)
    except OSError as exc:
        print(f"error: cannot listen on 127.0.0.1:{port} ({exc.strerror or exc}); "
              "pick another with --port, or --port 0 for a free one", file=sys.stderr)
        return 1
    print(f"llmtuner ui: {running.url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(running.url)).start()
    try:
        running.httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        running.httpd.server_close()
    return 0
