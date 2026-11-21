"""Tests for the OpenRouter adapter against a local stub HTTP server. Fully offline.

The adapter always runs as a subprocess in a fresh temp working directory, with HOME in a temp
dir and a fake API key (or none), so it can never find the project's real `.env` and can never
reach anything but the stub on 127.0.0.1.
"""

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from llmtuner.adapters.openrouter import find_key

SRC = str(Path(__file__).resolve().parents[1] / "src")
FAKE = "sk-test-FAKE-0000"


def ok_body(content="hello", usage=True, finish="stop", **extra):
    body = {"model": "vendor/m-resolved", "choices": [{"message": {"role": "assistant", "content": content},
                                                          "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 3, "cost": 0.00042}
    body.update(extra)
    return body


class Stub:
    """A scripted HTTP server on 127.0.0.1:<free port>. Records every request it gets."""

    def __init__(self, *responses):
        # Each response: (status, headers dict, body) where body is dict/bytes, or a callable(request) -> tuple.
        self.responses = list(responses) or [(200, {}, ok_body())]
        self.requests = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = None
                request = {"method": "POST", "path": self.path, "headers": dict(self.headers), "body": parsed}
                stub.requests.append(request)
                index = min(len(stub.requests), len(stub.responses)) - 1
                response = stub.responses[index]
                status, headers, body = response(request) if callable(response) else response
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


def adapter_env(home: Path, key: str | None = FAKE, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENROUTER_", "LLMTUNER_")) and k not in ("HOME", "PYTHONPATH")}
    env["HOME"] = str(home)
    env["PYTHONPATH"] = SRC
    if key is not None:
        env["OPENROUTER_API_KEY"] = key
    env.update(extra or {})
    return env


class AdapterCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.cwd = self.tmp / "work"
        self.home = self.tmp / "home"
        self.cwd.mkdir()
        self.home.mkdir()

    def run_adapter(self, stub_url, *args, stdin="the prompt", key=FAKE, meta=False, env_extra=None):
        extra = dict(env_extra or {})
        if meta:
            self.meta_path = self.tmp / "meta.json"
            extra["LLMTUNER_META"] = str(self.meta_path)
        proc = subprocess.run(
            [sys.executable, "-m", "llmtuner.adapters.openrouter", "--model", "vendor/m",
             "--base-url", stub_url, *args],
            input=stdin, capture_output=True, text=True, cwd=self.cwd, timeout=30,
            env=adapter_env(self.home, key, extra),
        )
        return proc

    def assert_clean_failure(self, proc, code=1):
        self.assertEqual(proc.returncode, code, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(proc.stderr.strip())


class RequestTests(AdapterCase):
    def test_happy_path(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url, "--max-tokens", "77", "--temperature", "0.5")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "hello")
        self.assertEqual(len(stub.requests), 1)
        req = stub.requests[0]
        self.assertEqual(req["method"], "POST")
        self.assertTrue(req["path"].endswith("/chat/completions"), req["path"])
        self.assertEqual(req["headers"]["Authorization"], f"Bearer {FAKE}")
        body = req["body"]
        self.assertEqual(body["model"], "vendor/m")
        self.assertEqual(body["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["max_tokens"], 77)

    def test_defaults_are_sent(self):
        with Stub() as stub:
            self.run_adapter(stub.url)
        body = stub.requests[0]["body"]
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 512)

    def test_base_url_trailing_slash(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url + "/")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(stub.requests[0]["path"], "/chat/completions")

    def test_system_prompt_goes_first(self):
        with Stub() as stub:
            self.run_adapter(stub.url, "--system", "be brief")
        self.assertEqual(stub.requests[0]["body"]["messages"],
                         [{"role": "system", "content": "be brief"}, {"role": "user", "content": "the prompt"}])

    def test_system_file(self):
        path = self.tmp / "system.txt"
        path.write_text("from a file\nline two", encoding="utf-8")
        with Stub() as stub:
            self.run_adapter(stub.url, "--system-file", str(path))
        messages = stub.requests[0]["body"]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "from a file\nline two"})
        self.assertEqual(messages[1], {"role": "user", "content": "the prompt"})

    def test_missing_system_file_is_a_clean_error(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url, "--system-file", str(self.tmp / "nope.txt"))
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(stub.requests, [])

    def test_non_http_base_url_is_refused(self):
        secret = self.tmp / "secret.txt"
        secret.write_text("TOP-SECRET-CONTENT")
        for url in ("file:///etc/passwd", secret.as_uri(), "ftp://127.0.0.1/x"):
            proc = self.run_adapter(url)
            self.assert_clean_failure(proc)
            self.assertIn("http", proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("TOP-SECRET-CONTENT", proc.stderr)
            self.assertNotIn("root:", proc.stderr)


class MetaTests(AdapterCase):
    def test_meta_written_with_usage(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url, meta=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        meta = json.loads(self.meta_path.read_text())
        self.assertEqual(meta["prompt_tokens"], 11)
        self.assertEqual(meta["completion_tokens"], 3)
        self.assertEqual(meta["cost"], 0.00042)
        self.assertEqual(meta["cost_unit"], "credits")
        self.assertEqual(meta["finish_reason"], "stop")
        self.assertEqual(meta["model"], "vendor/m-resolved")

    def test_no_usage_means_no_fields_not_zeros(self):
        with Stub((200, {}, ok_body(usage=False))) as stub:
            proc = self.run_adapter(stub.url, meta=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        meta = json.loads(self.meta_path.read_text())
        for field in ("cost", "cost_unit", "prompt_tokens", "completion_tokens"):
            self.assertNotIn(field, meta)

    def test_usage_without_cost_keeps_tokens_only(self):
        body = ok_body(usage=False)
        body["usage"] = {"prompt_tokens": 5, "completion_tokens": 2}
        with Stub((200, {}, body)) as stub:
            self.run_adapter(stub.url, meta=True)
        meta = json.loads(self.meta_path.read_text())
        self.assertEqual((meta["prompt_tokens"], meta["completion_tokens"]), (5, 2))
        self.assertNotIn("cost", meta)
        self.assertNotIn("cost_unit", meta)

    def test_no_meta_env_writes_nothing_and_succeeds(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sorted(p.name for p in self.cwd.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), [])

    def test_unwritable_meta_path_does_not_fail_the_call(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url, env_extra={"LLMTUNER_META": str(self.tmp / "no" / "dir" / "m.json")})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "hello")


class KeyLookupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.cwd = self.tmp / "a" / "b"
        self.cwd.mkdir(parents=True)
        self.home = self.tmp / "home"
        (self.home / ".config" / "llmtuner").mkdir(parents=True)
        # Keep the environment free of a real key for every test here.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OPENROUTER_API_KEY", None)

    def find(self):
        return find_key(cwd=self.cwd, home=self.home)

    def user_file(self, text):
        (self.home / ".config" / "llmtuner" / "env").write_text(text)

    def test_nothing_found(self):
        self.assertIsNone(self.find())

    def test_env_var_wins(self):
        (self.cwd / ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n")
        self.user_file("OPENROUTER_API_KEY=from-user\n")
        os.environ["OPENROUTER_API_KEY"] = "  from-env  "
        self.assertEqual(self.find(), "from-env")

    def test_dotenv_in_cwd(self):
        (self.cwd / ".env").write_text("OPENROUTER_API_KEY=in-cwd\n")
        self.user_file("OPENROUTER_API_KEY=from-user\n")
        self.assertEqual(self.find(), "in-cwd")

    def test_dotenv_in_parent(self):
        (self.tmp / "a" / ".env").write_text("OPENROUTER_API_KEY=in-parent\n")
        self.assertEqual(self.find(), "in-parent")

    def test_nearer_dotenv_beats_parent(self):
        (self.tmp / "a" / ".env").write_text("OPENROUTER_API_KEY=far\n")
        (self.cwd / ".env").write_text("OPENROUTER_API_KEY=near\n")
        self.assertEqual(self.find(), "near")

    def test_user_config_file(self):
        self.user_file("OPENROUTER_API_KEY=from-user\n")
        self.assertEqual(self.find(), "from-user")

    def test_line_forms(self):
        forms = [
            "OPENROUTER_API_KEY=plain",
            "OPENROUTER_API_KEY='single'",
            'OPENROUTER_API_KEY="double"',
            "export OPENROUTER_API_KEY='exported'",
            "  OPENROUTER_API_KEY = spaced  ",
        ]
        expected = ["plain", "single", "double", "exported", "spaced"]
        for form, want in zip(forms, expected):
            with self.subTest(form=form):
                (self.cwd / ".env").write_text(f"# a comment\nOTHER=1\n\n{form}\nMORE=2\n")
                self.assertEqual(self.find(), want)

    def test_comment_and_other_variables_ignored(self):
        (self.cwd / ".env").write_text("# OPENROUTER_API_KEY=commented\nOTHER_OPENROUTER_API_KEY=x\nFOO=bar\n")
        self.assertIsNone(self.find())

    def test_empty_value_continues_to_next_place(self):
        for empty in ("OPENROUTER_API_KEY=''", 'OPENROUTER_API_KEY=""', "OPENROUTER_API_KEY="):
            with self.subTest(empty=empty):
                (self.cwd / ".env").write_text(empty + "\n")
                (self.tmp / "a" / ".env").write_text("OPENROUTER_API_KEY=parent-value\n")
                self.assertEqual(self.find(), "parent-value")
                (self.tmp / "a" / ".env").unlink()
                self.user_file("OPENROUTER_API_KEY=user-value\n")
                self.assertEqual(self.find(), "user-value")
                self.user_file("OPENROUTER_API_KEY=''\n")
                self.assertIsNone(self.find())

    def test_empty_env_var_counts_as_unset(self):
        os.environ["OPENROUTER_API_KEY"] = "   "
        self.user_file("OPENROUTER_API_KEY=from-user\n")
        self.assertEqual(self.find(), "from-user")


class NoKeyTests(AdapterCase):
    def test_no_key_anywhere(self):
        with Stub() as stub:
            proc = self.run_adapter(stub.url, key=None)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("OPENROUTER_API_KEY", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(stub.requests, [])

    def test_key_from_dotenv_in_cwd_is_used(self):
        (self.cwd / ".env").write_text("OPENROUTER_API_KEY='sk-test-FAKE-dotenv'\n")
        with Stub() as stub:
            proc = self.run_adapter(stub.url, key=None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(stub.requests[0]["headers"]["Authorization"], "Bearer sk-test-FAKE-dotenv")

    def test_key_from_user_file_under_home(self):
        (self.home / ".config" / "llmtuner").mkdir(parents=True)
        (self.home / ".config" / "llmtuner" / "env").write_text("export OPENROUTER_API_KEY=sk-test-FAKE-home\n")
        with Stub() as stub:
            proc = self.run_adapter(stub.url, key=None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(stub.requests[0]["headers"]["Authorization"], "Bearer sk-test-FAKE-home")


class ErrorTests(AdapterCase):
    def test_http_401_message_on_stderr(self):
        with Stub((401, {}, {"error": {"message": "invalid credentials, sorry"}})) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertIn("invalid credentials, sorry", proc.stderr)
        self.assertIn("401", proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(len(stub.requests), 1)  # 401 is not retried

    def test_http_200_with_error_object(self):
        with Stub((200, {}, {"error": {"message": "provider fell over", "code": 502}})) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertIn("provider fell over", proc.stderr)

    def test_non_json_body(self):
        with Stub((200, {}, b"<html>not json</html>")) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)

    def test_non_json_error_body(self):
        with Stub((500, {}, b"plain text failure")) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertIn("500", proc.stderr)

    def test_response_without_choices(self):
        with Stub((200, {}, {"id": "x"})) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertIn("choices", proc.stderr)

    def test_json_list_body(self):
        with Stub((200, {}, [1, 2])) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)

    def test_connection_refused(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        proc = self.run_adapter(f"http://127.0.0.1:{port}")
        self.assert_clean_failure(proc)
        self.assertEqual(proc.stdout, "")


class RetryTests(AdapterCase):
    def test_429_then_success(self):
        with Stub((429, {"Retry-After": "0"}, {"error": {"message": "slow down"}}),
                  (200, {}, ok_body("recovered"))) as stub:
            proc = self.run_adapter(stub.url)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "recovered")
        self.assertEqual(len(stub.requests), 2)

    def test_three_429s_give_up(self):
        with Stub((429, {"Retry-After": "0"}, {"error": {"message": "slow down"}})) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertEqual(len(stub.requests), 3)
        self.assertIn("slow down", proc.stderr)


class ContentShapeTests(AdapterCase):
    def test_null_content_with_length(self):
        with Stub((200, {}, ok_body(content=None, finish="length"))) as stub:
            proc = self.run_adapter(stub.url, meta=True)
        self.assert_clean_failure(proc)
        self.assertIn("--max-tokens", proc.stderr)
        self.assertTrue(self.meta_path.exists())
        self.assertEqual(json.loads(self.meta_path.read_text())["finish_reason"], "length")
        self.assertEqual(proc.stdout, "")

    def test_empty_string_is_like_null(self):
        with Stub((200, {}, ok_body(content="", finish="length"))) as stub:
            proc = self.run_adapter(stub.url, meta=True)
        self.assert_clean_failure(proc)
        self.assertIn("--max-tokens", proc.stderr)
        self.assertTrue(self.meta_path.exists())

    def test_list_of_text_parts_is_joined(self):
        parts = [{"type": "text", "text": "Hello, "}, {"type": "image", "url": "x"}, {"type": "text", "text": "world"}]
        with Stub((200, {}, ok_body(content=parts))) as stub:
            proc = self.run_adapter(stub.url)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "Hello, world")


class KeyLeakTests(AdapterCase):
    def test_echoed_key_is_masked(self):
        def echo(request):
            auth = request["headers"]["Authorization"]
            return 401, {}, {"error": {"message": f"bad credential {auth} and also {FAKE}"}}

        with Stub(echo) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertNotIn(FAKE, proc.stderr)
        self.assertNotIn(FAKE, proc.stdout)
        self.assertIn("***", proc.stderr)

    def test_echoed_key_in_200_error_is_masked(self):
        def echo(request):
            return 200, {}, {"error": {"message": f"nope {request['headers']['Authorization']}"}}

        with Stub(echo) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertNotIn(FAKE, proc.stderr)
        self.assertIn("***", proc.stderr)

    def test_echoed_key_in_plain_text_error_is_masked(self):
        def echo(request):
            return 500, {}, f"internal error for {FAKE}".encode()

        with Stub(echo) as stub:
            proc = self.run_adapter(stub.url)
        self.assert_clean_failure(proc)
        self.assertNotIn(FAKE, proc.stderr)


if __name__ == "__main__":
    unittest.main()
