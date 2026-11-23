"""The change log of a dataset, and the one safe way to edit a case.

Every edit of a case's `expected`, `input`, `context` or `split` goes through `edit_case`,
which rewrites only that case's line and appends an entry to the dataset's log
(`<dataset>.changes.jsonl` unless the project config says otherwise). An edit made by hand
leaves no entry; comparisons then show the case as changed "unrecorded".

An entry is one JSON object per line:

    {"ts": 1700000000.0, "at": "2023-11-14T22:13:20Z", "case_id": "t1",
     "action": "edit" | "split", "by": "human" | "agent", "via": "cli" | "mcp",
     "agent_signal": null | "CLAUDECODE" | "AI_AGENT" | "flag" | "mcp",
     "reason": str | null, "basis": null | {"kind": "rule", "ref": str} | {"kind": "owner", "quote": str},
     "before": {field: old value}, "after": {field: new value}}

`before` and `after` hold only the fields that changed.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from llmtuner.config import Config, Dataset
from llmtuner.dataset import SPLITS, DatasetError, load_cases

try:
    import fcntl
except ImportError:  # no flock on this platform (Windows): editing is then only safe within one process
    fcntl = None

FIELDS = ("expected", "input", "context", "split")
# Environment variables that mark the shell of an AI agent, checked in this order.
AGENT_ENV = ("CLAUDECODE", "AI_AGENT")
# Splits where an edit changes the numbers a result is reported on.
GUARDED_SPLITS = ("test", "holdout")


class ChangeError(DatasetError):
    """An edit was refused or could not be written."""


_MCP_HOLDOUT = ("no such case, or it is in the holdout: holdout cases cannot be viewed, edited or moved"
                " through MCP (a person does it with `llmtuner case edit`)")


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()  # "not given", as opposed to None, which is a real value for expected and context


def who_is_editing(flag: str | None, via: str) -> tuple[str, str | None]:
    """Who this edit is recorded as: ("human" | "agent", the signal that said so).

    Through MCP it is always an agent. In the CLI an agent's shell is recognised by its
    environment, and `--by human` cannot override that; `--by agent` marks an edit as the agent's.
    """
    if via == "mcp":
        return "agent", "mcp"
    for name in AGENT_ENV:
        if os.environ.get(name):
            return "agent", name
    if flag == "agent":
        return "agent", "flag"
    return "human", None


def canonical(value) -> str:
    """A form in which equal JSON values are equal text: key order does not matter, and
    true, 1 and 1.0 stay different. Used wherever two values are compared."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


# --- the log -----------------------------------------------------------------


def read_log_with_skipped(path: Path) -> tuple[list[dict], int]:
    """The entries of a log and how many lines were skipped as malformed. A missing file is empty."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return [], 0
    except OSError as exc:
        raise ChangeError(f"cannot read change log {path}: {exc}") from exc
    entries, skipped = [], 0
    for line in raw.decode("utf-8", "replace").split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str) \
                and isinstance(entry.get("ts"), (int, float)):
            entries.append(entry)
        else:
            skipped += 1
    return entries, skipped


def read_log(path: Path) -> list[dict]:
    return read_log_with_skipped(path)[0]


def log_for_run(config: Config | None, run: dict) -> list[dict] | None:
    """The change log that applies to a stored run, or None when this project config does not
    describe the run's project and dataset (then nobody can say who changed what)."""
    if config is None or run["project"] != config.project or run["dataset"] not in config.datasets:
        return None
    return read_log(config.datasets[run["dataset"]].changes)


def append_entry(path: Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    # A hand-edited log may lack its final newline; do not glue the new entry onto the last line.
    try:
        with open(path, "rb") as existing:
            existing.seek(0, os.SEEK_END)
            if existing.tell():
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) != b"\n":
                    line = "\n" + line
    except FileNotFoundError:
        pass
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


# --- editing -----------------------------------------------------------------


def _atomic_write(path: Path, data: bytes, check=None) -> None:
    """Replace `path` with `data` in one step, keeping its file mode. `check(temp_path)` may
    raise to stop before the real file is touched."""
    mode = path.stat().st_mode & 0o7777
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, mode)
        if check is not None:
            check(Path(temp))
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def _clean_basis(basis) -> dict | None:
    if basis is None:
        return None
    kind = basis.get("kind") if isinstance(basis, dict) else None
    key = {"rule": "ref", "owner": "quote"}.get(kind)
    text = basis.get(key) if key else None
    if key is None or not isinstance(text, str) or not text.strip():
        raise ChangeError('basis must be {"kind": "rule", "ref": NAME} or {"kind": "owner", "quote": WORDS},'
                          " with the name or the owner's own words filled in")
    return {"kind": kind, key: text.strip()}


_held = threading.local()  # lock files this thread already holds, so locking can nest
_process_locks: dict[str, threading.Lock] = {}  # the fallback when fcntl is missing


@contextlib.contextmanager
def dataset_lock(dataset: Dataset):
    """Hold an exclusive lock on a dataset while it is read, rewritten and logged, so two edits
    cannot overwrite each other. The lock is flock() on `<change log>.lock`, which the operating
    system drops when the process ends, so a leftover file never blocks anything. Nests within a thread."""
    key = str(dataset.path)
    held = getattr(_held, "keys", None)
    if held is None:
        held = _held.keys = set()
    if key in held:
        yield
        return
    handle = None
    if fcntl is not None:
        try:
            # Next to the change log, not the dataset: a project that keeps its log out of git
            # (the `changes` setting) keeps the lock file there too.
            lock_path = dataset.changes.with_name(dataset.changes.name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a")
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:
            if handle:
                handle.close()
            raise ChangeError(f"cannot lock the dataset for editing: {exc}") from exc
        lock = None
    else:
        lock = _process_locks.setdefault(key, threading.Lock())
        lock.acquire()
    held.add(key)
    try:
        yield
    finally:
        held.discard(key)
        if handle:
            handle.close()  # closing releases the flock
        if lock:
            lock.release()


def edit_case(config: Config, dataset_name: str, case_id: str, **kwargs) -> dict:
    """Change one case under the dataset's lock; see `_edit_case` for the arguments."""
    with dataset_lock(config.dataset(dataset_name)):
        return _edit_case(config, dataset_name, case_id, **kwargs)


def _edit_case(config: Config, dataset_name: str, case_id: str, *, expected=UNSET, input=UNSET,
               context=UNSET, split=UNSET, reason: str | None = None, basis=None, by: str, via: str,
               signal: str | None = None, now: float | None = None) -> dict:
    """Change fields of one case in the dataset file and log it. Returns the log entry.

    Only the case's own line is rewritten (every other byte stays, and so does the line's
    ending), atomically. If the log cannot be written the dataset file is put back.
    Fields left as UNSET are not touched.
    """
    dataset = config.dataset(dataset_name)
    path = dataset.path
    load_cases(path)  # refuse to touch a file the loader cannot read
    raw = path.read_bytes()
    lines = raw.split(b"\n")  # "\n" only, like the loader; a trailing "\r" belongs to its line
    index = row = None
    for number, line in enumerate(lines):
        if line.strip():
            candidate = json.loads(line.decode("utf-8"))
            if str(candidate["id"]) == case_id:
                index, row = number, candidate
                break
    if via == "mcp" and (row is None or row.get("split", "test") == "holdout" or split == "holdout"):
        # One answer for a missing id and a holdout case, and before any value is compared, so an
        # agent cannot probe holdout ids or values.
        raise ChangeError(_MCP_HOLDOUT)
    if row is None:
        raise ChangeError(f"dataset '{dataset_name}' has no case '{case_id}'")

    given = {"expected": expected, "input": input, "context": context, "split": split}
    given = {k: v for k, v in given.items() if v is not UNSET}
    if "split" in given and given["split"] not in SPLITS:
        raise ChangeError(f"split must be one of {', '.join(SPLITS)}")
    if "context" in given and given["context"] is not None and not isinstance(given["context"], str):
        raise ChangeError("context must be text")
    if "input" in given and given["input"] is None:
        raise ChangeError("input cannot be empty")
    current = {"expected": row.get("expected"), "input": row.get("input"),
               "context": row.get("context"), "split": row.get("split", "test")}
    changed = [k for k, v in given.items() if canonical(v) != canonical(current[k])]
    if not changed:
        raise ChangeError(f"nothing to change: case '{case_id}' already has these values")

    split_before, split_after = current["split"], given.get("split", current["split"])
    guarded = split_before in GUARDED_SPLITS or split_after in GUARDED_SPLITS
    if guarded and not (reason or "").strip():
        raise ChangeError("a reason is required to edit a test or holdout case (say why the expectation was wrong)")
    clean_basis = _clean_basis(basis)
    if by == "agent" and guarded and clean_basis is None:
        raise ChangeError("an agent editing a test or holdout case must give its basis: a rule"
                          " (name of the rule or file) or the owner's words (a verbatim quote),"
                          " not the model's answers")

    new_row = dict(row)  # keeps the key order of the line
    for key in changed:
        new_row[key] = given[key]
    try:
        new_line = json.dumps(new_row, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ChangeError(f"the new value is not valid text: {exc.reason}") from exc
    if lines[index].endswith(b"\r"):
        new_line += b"\r"
    new_raw = b"\n".join(lines[:index] + [new_line] + lines[index + 1:])

    ts = time.time() if now is None else now
    entry = {
        "ts": ts,
        "at": datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "case_id": case_id,
        "action": "split" if changed == ["split"] else "edit",
        "by": by,
        "via": via,
        "agent_signal": signal,
        "reason": reason.strip() if reason and reason.strip() else None,
        "basis": clean_basis,
        "before": {k: current[k] for k in changed},
        "after": {k: given[k] for k in changed},
    }
    _atomic_write(path, new_raw, check=load_cases)  # a bad edit never reaches the real file
    try:
        load_cases(path)
        append_entry(dataset.changes, entry)
    except BaseException as exc:
        _atomic_write(path, raw)  # no entry, no edit
        if isinstance(exc, (OSError, DatasetError)):
            raise ChangeError(f"could not record the change, so the dataset was left as it was: {exc}") from exc
        raise
    return entry


def agent_edit(config: Config, dataset_name: str, case_id: str, *, reason: str | None,
               basis_kind: str | None = None, basis_text: str | None = None, expected_json: str | None = None, input_json: str | None = None,
               context: str | None = None, split: str | None = None, now: float | None = None) -> dict:
    """The edit an agent can make through MCP: always an agent, via MCP. Values for `expected`
    and `input` arrive as JSON text; a field left as None is not touched."""
    fields = {}
    for name, text in (("expected", expected_json), ("input", input_json)):
        if text is not None:
            try:
                fields[name] = json.loads(text)
            except ValueError as exc:
                raise ChangeError(f"{name}_json is not valid JSON ({exc})") from exc
    if context is not None:
        fields["context"] = context
    if split is not None:
        fields["split"] = split
    basis = None  # only a test or holdout edit needs one; edit_case insists when it does
    if basis_kind or (basis_text or "").strip():
        basis_key = {"rule": "ref", "owner": "quote"}.get(basis_kind)
        if basis_key is None:
            raise ChangeError('basis_kind must be "rule" or "owner"')
        basis = {"kind": basis_kind, basis_key: basis_text}
    return edit_case(config, dataset_name, case_id, reason=reason, basis=basis,
                     by="agent", via="mcp", signal="mcp", now=now, **fields)


# --- proposing a split -------------------------------------------------------


def _largest_remainder(n: int, values: list[float]) -> list[int]:
    """Whole counts adding up to `n`, as close to the shares `values` (which sum to 1) as possible.
    The leftovers go to the largest fractional parts; ties go in order."""
    quotas = [round(n * v, 9) for v in values]
    counts = [math.floor(q) for q in quotas]
    order = sorted(range(len(values)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


def split_counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """How many of `n` cases each split gets for these fractions (dev, test, holdout)."""
    values = [fractions.get(s, 0.0) for s in SPLITS]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ChangeError("split fractions must be numbers of 0 or more")
    if abs(sum(values) - 1) > 1e-9:
        raise ChangeError(f"split fractions must add up to 1 (they add up to {sum(values):g})")
    return dict(zip(SPLITS, _largest_remainder(n, values)))


def propose_split(case_ids: list[str], fractions: dict[str, float], seed: int,
                  seen: frozenset[str] | set[str] = frozenset()) -> dict[str, str]:
    """A random division of the cases into splits, by `fractions` (dev, test, holdout).

    The ids are shuffled with `random.Random(seed)`; counts follow the largest remainder method, so
    they add up to the number of cases and are as close to the fractions as whole cases allow.

    `seen` are cases already shown in some run outside the holdout. They never go to holdout, since an
    agent may have read their inputs and outputs there; if too few cases are left, holdout gets fewer
    and dev and test share the rest by their fractions. With no `seen` cases the division is the
    plain one: dev, test and holdout take consecutive stretches of the shuffled ids.
    """
    counts = split_counts(len(case_ids), fractions)
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    fresh = [c for c in shuffled if c not in seen]
    take = min(counts["holdout"], len(fresh))
    short = counts["holdout"] - take
    if short:
        dev, test = fractions.get("dev", 0.0), fractions.get("test", 0.0)
        shares = [dev / (dev + test), test / (dev + test)] if dev + test > 0 else [0.0, 1.0]
        extra = _largest_remainder(short, shares)
        counts["dev"] += extra[0]
        counts["test"] += extra[1]
    # Holdout takes the last fresh ids in shuffled order, so with nothing seen it is the final stretch.
    holdout = set(fresh[len(fresh) - take:]) if take else set()
    others = [c for c in shuffled if c not in holdout]
    result = {c: "holdout" for c in holdout}
    for i, case_id in enumerate(others):
        result[case_id] = "dev" if i < counts["dev"] else "test"
    return result
