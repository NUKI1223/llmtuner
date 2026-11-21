"""Target adapter for OpenRouter and other OpenAI-compatible chat APIs.

Reads the prompt on stdin and prints the model's answer on stdout, so it can be
a target or a judge command:

    command = ["{python}", "-m", "llmtuner.adapters.openrouter", "--model", "vendor/model"]

The key comes from the OPENROUTER_API_KEY environment variable, else from a
`.env` file in the working directory or one of its parents, else from
~/.config/llmtuner/env. It is never taken from the command line, so it never
lands in a stored run.

When llmtuner runs this as a target it also reports the tokens and cost of the
call, which is what `llmtuner rank` uses for cost per passed case.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from llmtuner.targets import META_ENV

KEY_NAME = "OPENROUTER_API_KEY"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_RETRY_STATUSES = {429, 502, 503}
_MAX_RETRY_WAIT = 30.0
_MAX_BODY = 16 * 1024 * 1024  # a chat reply is never this large; stops a runaway body


class AdapterError(Exception):
    pass


def _value_from_env_file(path: Path, name: str) -> str | None:
    """Read NAME=value (optionally `export NAME='value'`) from a dotenv-style file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        line = line.strip().removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep or key.strip() != name:
            continue
        raw, value = value, value.strip()
        quoted = re.fullmatch(r"""(["'])(.*?)\1(?:\s+#.*)?""", value)
        if quoted:
            value = quoted.group(2)
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            # Unquoted: a `#` after whitespace starts a comment.
            value = re.split(r"\s#", raw, maxsplit=1)[0]
        return value.strip() or None
    return None


def find_key(cwd: Path | None = None, home: Path | None = None) -> str | None:
    """The API key, or None. Environment first, then `.env` upward from cwd, then the user file."""
    if os.environ.get(KEY_NAME, "").strip():
        return os.environ[KEY_NAME].strip()
    cwd = (cwd or Path.cwd()).resolve()
    for folder in (cwd, *cwd.parents):
        key = _value_from_env_file(folder / ".env", KEY_NAME)
        if key:
            return key
    return _value_from_env_file((home or Path.home()) / ".config" / "llmtuner" / "env", KEY_NAME)


def _mask(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def _error_message(raw: bytes, key: str = "") -> str:
    """The API's error text with the key masked first, then cut, so no piece of the key survives the cut."""
    text = _mask(raw.decode("utf-8", "replace"), key)
    try:
        error = json.loads(text).get("error")
        if error is not None:
            return _mask(str(error.get("message") if isinstance(error, dict) else error), key)
    except (ValueError, AttributeError):
        pass
    return text.strip()[:300] or "no details"


def _retry_wait(header: str | None, attempt: int) -> float:
    try:
        wait = float(header)
        if wait != wait:  # NaN
            raise ValueError
        return min(_MAX_RETRY_WAIT, max(0.0, wait))
    except (TypeError, ValueError):
        return min(_MAX_RETRY_WAIT, 2.0 * attempt)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: urllib would forward the Authorization header to the new address."""

    def redirect_request(self, *args, **kwargs):
        return None


def _opener(base_url: str) -> urllib.request.OpenerDirector:
    handlers: list = [_NoRedirect()]
    if base_url.startswith("http://"):
        handlers.append(urllib.request.ProxyHandler({}))  # a plain-http API is local or trusted: no env proxy
    return urllib.request.build_opener(*handlers)


def _read_all(response, deadline: float, timeout: float) -> bytes:
    """Read a body in chunks, so one deadline covers a server that sends it a byte at a time."""
    chunks = []
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise AdapterError(f"timed out after {timeout:g} seconds")
        try:  # make the next blocking read stop at the deadline, not at the first attempt's timeout
            response.fp.raw._sock.settimeout(left)
        except AttributeError:
            pass
        chunk = response.read1(65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        if sum(map(len, chunks)) > _MAX_BODY:
            raise AdapterError("the API response is too large")


def complete(*, timeout: float, **kwargs) -> dict:
    """One chat completion. Returns the decoded response or raises AdapterError.

    `timeout` is one deadline for the whole call: every attempt, every retry wait and every body read.
    Reads are bounded chunk by chunk; where signals work (the main thread of a command line run) a
    timer also stops a server that drips the status line and headers a byte at a time.
    """
    if not (hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()):
        return _complete(timeout=timeout, **kwargs)

    def expired(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, min(max(timeout, 0.001), 10 ** 7))
    try:
        return _complete(timeout=timeout, **kwargs)
    except TimeoutError:  # the timer fired outside the request itself
        raise AdapterError(f"timed out after {timeout:g} seconds") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        if old_delay > 0:  # a caller using this as a library had its own timer running: give back what is left
            left = max(old_delay - (time.monotonic() - started), 0.001)
            signal.setitimer(signal.ITIMER_REAL, left, old_interval)


def _complete(*, base_url: str, key: str, model: str, prompt: str, system: str | None,
              max_tokens: int, temperature: float, timeout: float, attempts: int = 3) -> dict:
    deadline = time.monotonic() + timeout
    if any(ord(c) > 255 or ord(c) < 32 or ord(c) == 127 for c in key):
        raise AdapterError("the API key contains a line break or a character that cannot be sent in a header")
    if not base_url.startswith(("https://", "http://")):
        raise AdapterError("the base URL must start with https:// or http://")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}).encode("utf-8")
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        opener = _opener(base_url)
    except ValueError as exc:
        raise AdapterError(f"the base URL is not valid: {_mask(str(exc), key)}") from None
    for attempt in range(1, attempts + 1):
        left = deadline - time.monotonic()
        if left <= 0:
            raise AdapterError(f"timed out after {timeout:g} seconds")
        try:
            with opener.open(request, timeout=left) as response:
                payload = json.loads(_read_all(response, deadline, timeout).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise AdapterError(f"the API answered with a redirect (HTTP {exc.code}) to another address;"
                                   " not followed") from None
            wait = _retry_wait(exc.headers.get("Retry-After"), attempt)
            if exc.code in _RETRY_STATUSES and attempt < attempts and time.monotonic() + wait < deadline:
                time.sleep(wait)
                continue
            try:
                raw = _read_all(exc, deadline, timeout)
            except (AdapterError, OSError, http.client.HTTPException):
                raw = b""
            raise AdapterError(f"HTTP {exc.code}: {_error_message(raw, key)}") from None
        except AdapterError:
            raise
        except ValueError as exc:
            if isinstance(exc, UnicodeError) or isinstance(exc, json.JSONDecodeError):
                raise AdapterError("the API returned something that is not JSON") from None
            raise AdapterError(f"the base URL is not valid: {_mask(str(exc), key)}") from None
        except TimeoutError:
            raise AdapterError(f"timed out after {timeout:g} seconds") from None
        except OSError as exc:  # URLError, refused connections
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise AdapterError(f"timed out after {timeout:g} seconds") from None
            raise AdapterError(f"request failed: {_mask(str(reason), key)}") from None
        except http.client.HTTPException as exc:  # bad URL parts, truncated or malformed replies
            raise AdapterError(f"request failed: {_mask(str(exc) or type(exc).__name__, key)}") from None
        # A failure during generation can arrive with HTTP 200 and an error object in the body.
        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise AdapterError(f"API error: {_mask(str(message), key)}")
        if not isinstance(payload, dict):
            raise AdapterError("the API returned an unexpected response")
        return payload
    raise AdapterError("no attempts were made")


def _text(content) -> str:
    """Answer text from a message's content, which may be a string, a list of parts or null."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_meta(payload: dict, model: str, finish_reason) -> dict:
    """What llmtuner stores about the call. Fields the API did not return are left out."""
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    meta = {"model": payload.get("model") or model}
    for field in ("prompt_tokens", "completion_tokens"):
        if _number(usage.get(field)) is not None:
            meta[field] = usage[field]
    if _number(usage.get("cost")) is not None:
        meta["cost"] = usage["cost"]
        meta["cost_unit"] = "credits"  # OpenRouter reports cost in its credits
    if finish_reason:
        meta["finish_reason"] = finish_reason
    return meta


def _write_meta(meta: dict) -> None:
    path = os.environ.get(META_ENV)
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(meta), encoding="utf-8")
    except OSError:
        pass  # reporting usage is optional; never fail the call over it


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 < value <= 86400:
        raise argparse.ArgumentTypeError("must be more than 0 and at most 86400 seconds")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m llmtuner.adapters.openrouter",
        description="Send stdin to a model on OpenRouter and print its answer.",
    )
    parser.add_argument("--model", required=True, help="model id, e.g. the one shown on the model's page")
    parser.add_argument("--system", help="system prompt text")
    parser.add_argument("--system-file", help="file holding the system prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="cap on the answer length (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0, help="default: 0")
    parser.add_argument("--timeout", type=_seconds, default=90.0, help="seconds to wait for the API (default: 90)")
    parser.add_argument("--base-url", default=os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
                        help="API base URL, for other OpenAI-compatible services")
    args = parser.parse_args(argv)

    system = args.system
    if args.system_file:
        try:
            system = Path(args.system_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read --system-file: {exc}", file=sys.stderr)
            return 2
    key = find_key()
    if not key:
        print(f"error: no API key: set {KEY_NAME} or put {KEY_NAME}='...' in a .env file", file=sys.stderr)
        return 2

    prompt = sys.stdin.read()
    try:
        payload = complete(base_url=args.base_url, key=key, model=args.model, prompt=prompt, system=system,
                           max_tokens=args.max_tokens, temperature=args.temperature, timeout=args.timeout)
        choice = payload["choices"][0]
        text = _text(choice["message"].get("content"))
        finish_reason = choice.get("finish_reason")
    except AdapterError as exc:
        # Belt and braces: the key must never reach output that llmtuner stores.
        print("error: " + str(exc).replace(key, "***"), file=sys.stderr)
        return 1
    except (KeyError, IndexError, TypeError, AttributeError):
        print("error: the API response had no choices[0].message", file=sys.stderr)
        return 1

    _write_meta(build_meta(payload, args.model, finish_reason))
    if not text.strip():
        print(f"error: the model returned no text (finish_reason: {finish_reason});"
              " a reasoning model may need a higher --max-tokens", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
