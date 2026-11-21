"""Run a system under test as a command: input on stdin, output on stdout.

This one adapter is how any project or agent connects. A script, a CLI agent
(`claude -p`, `codex exec`) or a curl call to a service all fit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 200_000
_STDERR_TAIL = 500

# A command may use this token in place of an interpreter. It becomes the Python that is
# running llmtuner, which is the one guaranteed to be able to import llmtuner's adapters.
PYTHON_TOKEN = "{python}"

# Name of the environment variable that tells a target where it may write a small JSON
# object about the call it just made: {"model", "prompt_tokens", "completion_tokens",
# "cost", "cost_unit"}. Writing it is optional.
META_ENV = "LLMTUNER_META"

# Which repeat of the case this call is (1 when the run has no repeated trials).
TRIAL_ENV = "LLMTUNER_TRIAL"


@dataclass(frozen=True)
class CommandResult:
    output: str
    exit_code: int | None
    timed_out: bool
    duration_s: float
    error: str | None  # None when the command ran and exited 0

    @property
    def ok(self) -> bool:
        return self.error is None


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_command(command, stdin_text: str, timeout: float, cwd: Path,
                env: dict[str, str] | None = None) -> CommandResult:
    """Run `command` without a shell. Failures are returned, never raised.

    `env` adds variables on top of the current environment.
    """
    start = time.monotonic()
    argv = [sys.executable if arg == PYTHON_TOKEN else arg for arg in command]
    try:
        proc = subprocess.Popen(
            argv,
            env={**os.environ, **env} if env else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            start_new_session=True,  # so a timeout can kill the whole process group
        )
    except OSError as exc:
        return CommandResult("", None, False, 0.0, f"could not start {command[0]}: {exc}")

    timed_out = False
    try:
        out, err = proc.communicate(stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            # A detached child may still hold the pipes open; give up on its output.
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            out, err = "", ""
    duration = time.monotonic() - start

    error = None
    if timed_out:
        error = f"timed out after {timeout:g}s"
    elif proc.returncode != 0:
        tail = (err or "").strip()[-_STDERR_TAIL:]
        error = f"exit code {proc.returncode}" + (f": {tail}" if tail else "")
    return CommandResult((out or "")[:MAX_OUTPUT_CHARS], proc.returncode, timed_out, duration, error)
