"""The one place every project reports to: a user-level SQLite store.

Location: $LLMTUNER_HOME, else $XDG_DATA_HOME/llmtuner, else
~/.local/share/llmtuner. Runs are keyed by project name, so many projects
share one store.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

# Label key for a human verdict on the whole case, as opposed to one grader.
CASE_LABEL = "__case__"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  dataset TEXT NOT NULL,
  dataset_hash TEXT NOT NULL,
  target TEXT NOT NULL,
  command TEXT NOT NULL,
  model TEXT,
  tool_version TEXT,
  split TEXT NOT NULL,
  graders TEXT NOT NULL,
  note TEXT,
  started_at REAL NOT NULL,
  finished_at REAL,
  trial_group INTEGER,
  trial INTEGER,
  trials_planned INTEGER
);
CREATE TABLE IF NOT EXISTS cases (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  case_id TEXT NOT NULL,
  input TEXT NOT NULL,
  expected TEXT,
  context TEXT,
  output TEXT NOT NULL,
  exit_code INTEGER,
  timed_out INTEGER NOT NULL,
  duration_s REAL NOT NULL,
  error TEXT,
  passed INTEGER NOT NULL,
  meta TEXT,
  PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS grades (
  run_id INTEGER NOT NULL,
  case_id TEXT NOT NULL,
  grader TEXT NOT NULL,
  passed INTEGER NOT NULL,
  error INTEGER NOT NULL,
  detail TEXT,
  PRIMARY KEY (run_id, case_id, grader),
  FOREIGN KEY (run_id, case_id) REFERENCES cases(run_id, case_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS labels (
  run_id INTEGER NOT NULL,
  case_id TEXT NOT NULL,
  grader TEXT NOT NULL,
  passed INTEGER NOT NULL,
  note TEXT,
  labeled_at REAL NOT NULL,
  PRIMARY KEY (run_id, case_id, grader),
  FOREIGN KEY (run_id, case_id) REFERENCES cases(run_id, case_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS baselines (
  project TEXT NOT NULL,
  dataset TEXT NOT NULL,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  PRIMARY KEY (project, dataset)
);
-- An agent's holdout attempt, held from the moment it is allowed until its run row exists.
CREATE TABLE IF NOT EXISTS holdout_reservations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  dataset TEXT NOT NULL,
  at REAL NOT NULL
);
"""


# Columns added after the first version: (table, column, type). All are nullable, so a store
# made by an older version is upgraded in place by adding them.
_ADDED_COLUMNS = (
    ("cases", "meta", "TEXT"),
    ("runs", "trial_group", "INTEGER"),
    ("runs", "trial", "INTEGER"),
    ("runs", "trials_planned", "INTEGER"),
    # Where a run was started from ("cli", "mcp" or "ui"); NULL for runs made before it was recorded.
    ("runs", "origin", "TEXT"),
    # The `limit` a run used, NULL when it ran every case of its split.
    ("runs", "case_limit", "INTEGER"),
    # The case's own split at run time. NULL on rows made before splits were recorded; that NULL
    # is how a comparison recognises a baseline from "before the change log".
    ("cases", "split", "TEXT"),
    # How many cases the split held before `limit` cut them; with case_limit it tells whether the limit cut anything.
    ("runs", "cases_selected", "INTEGER"),
)


class StoreError(LookupError):
    pass


def _utf8(text: str | None) -> str | None:
    """Replace lone surrogates, which SQLite cannot store as UTF-8."""
    return None if text is None else text.encode("utf-8", "replace").decode("utf-8")


def default_home() -> Path:
    if os.environ.get("LLMTUNER_HOME"):
        return Path(os.environ["LLMTUNER_HOME"]).expanduser()
    data = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(data).expanduser() / "llmtuner"


class Store:
    def __init__(self, home: str | Path | None = None):
        self.home = Path(home) if home else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "store.db"
        # The MCP server may call tools from worker threads; calls are short and serialised by SQLite.
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Bring a store made by an older version up to date. CREATE IF NOT EXISTS adds no columns.

        Several processes may open an old store at once, so the check and the ALTER run under one
        write lock, and the columns are read again after the lock is taken.
        """
        def missing() -> list[tuple[str, str, str]]:
            found = {table: {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
                     for table in {t for t, _, _ in _ADDED_COLUMNS}}
            return [(t, c, kind) for t, c, kind in _ADDED_COLUMNS if c not in found[t]]

        if not missing():
            return
        try:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for table, column, kind in missing():
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- runs ---------------------------------------------------------------

    def create_run(self, *, project, dataset, dataset_hash, target, command, model,
                   tool_version, split, graders, note, trial_group: int | None = None,
                   trial: int | None = None, trials_planned: int | None = None,
                   origin: str | None = None, case_limit: int | None = None,
                   cases_selected: int | None = None) -> int:
        """Record a run. Repeated trials are ordinary runs that share a `trial_group`: the id of
        the group's first run. Pass `trials_planned` without `trial_group` to start a group."""
        with self._db:
            cursor = self._db.execute(
                "INSERT INTO runs (project, dataset, dataset_hash, target, command, model,"
                " tool_version, split, graders, note, started_at, trial_group, trial, trials_planned,"
                " origin, case_limit, cases_selected) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (project, dataset, dataset_hash, target, json.dumps(list(command)), model,
                 tool_version, split, json.dumps(graders), note, time.time(),
                 trial_group, trial, trials_planned, origin, case_limit, cases_selected),
            )
            if trials_planned is not None and trial_group is None:
                self._db.execute("UPDATE runs SET trial_group = id WHERE id = ?", (cursor.lastrowid,))
        return cursor.lastrowid

    def group_runs(self, run_id: int) -> list[dict]:
        """The runs that are trials of the same group as `run_id`, in trial order.

        A run that is not part of a group is returned alone.
        """
        run = self.get_run(run_id)
        if run["trial_group"] is None:
            return [run]
        rows = self._db.execute(
            "SELECT id FROM runs WHERE trial_group = ? ORDER BY trial, id", (run["trial_group"],))
        return [self.get_run(row[0]) for row in rows]

    def add_case(self, run_id: int, case, result, passed: bool, grades, meta: dict | None = None) -> None:
        expected = None if case.expected is None else json.dumps(case.expected, ensure_ascii=False)
        meta_text = None if meta is None else _utf8(json.dumps(meta, ensure_ascii=False))
        with self._db:
            # Columns are named, so a store migrated from an older layout still lines up.
            self._db.execute(
                "INSERT INTO cases (run_id, case_id, input, expected, context, output, exit_code,"
                " timed_out, duration_s, error, passed, meta, split) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, _utf8(case.id), _utf8(case.input_text), _utf8(expected), _utf8(case.context), _utf8(result.output),
                 result.exit_code, int(result.timed_out), result.duration_s, _utf8(result.error), int(passed), meta_text,
                 getattr(case, "split", None)),
            )
            self._db.executemany(
                "INSERT INTO grades (run_id, case_id, grader, passed, error, detail) VALUES (?,?,?,?,?,?)",
                [(run_id, _utf8(case.id), g.grader, int(g.passed), int(g.error), _utf8(g.detail)) for g in grades],
            )

    def finish_run(self, run_id: int) -> None:
        with self._db:
            self._db.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (time.time(), run_id))

    def get_run(self, run_id: int) -> dict:
        row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError(f"no run with id {run_id}")
        run = dict(row)
        run["command"] = json.loads(run["command"])
        run["graders"] = json.loads(run["graders"])
        return run

    def latest_run_id(self, project: str | None = None) -> int:
        query, args = "SELECT MAX(id) FROM runs", ()
        if project:
            query, args = query + " WHERE project = ?", (project,)
        run_id = self._db.execute(query, args).fetchone()[0]
        if run_id is None:
            raise StoreError("no runs recorded yet" + (f" for project '{project}'" if project else ""))
        return run_id

    def list_runs(self, project: str | None = None, limit: int = 20) -> list[dict]:
        query = (
            "SELECT r.id, r.project, r.dataset, r.target, r.model, r.split, r.note, r.started_at,"
            " r.finished_at, r.trial_group, r.trial, r.trials_planned,"
            " COUNT(c.case_id) AS n, COALESCE(SUM(c.passed), 0) AS n_passed"
            " FROM runs r LEFT JOIN cases c ON c.run_id = r.id"
        )
        args: tuple = ()
        if project:
            query, args = query + " WHERE r.project = ?", (project,)
        query += " GROUP BY r.id ORDER BY r.id DESC LIMIT ?"
        return [dict(r) for r in self._db.execute(query, args + (limit,))]

    def get_cases(self, run_id: int) -> list[dict]:
        rows = self._db.execute("SELECT * FROM cases WHERE run_id = ? ORDER BY rowid", (run_id,))
        cases = [dict(r) for r in rows]
        for case in cases:
            case["meta"] = json.loads(case["meta"]) if case["meta"] else None
        return cases

    def finished_runs(self, project: str, dataset: str, split: str = "test") -> list[dict]:
        """Every finished run of one dataset and split, newest first, with the set of case ids it holds."""
        rows = self._db.execute(
            "SELECT id FROM runs WHERE project = ? AND dataset = ? AND split = ?"
            " AND finished_at IS NOT NULL ORDER BY id DESC",
            (project, dataset, split),
        ).fetchall()
        runs = []
        for (run_id,) in rows:
            run = self.get_run(run_id)
            run["case_ids"] = frozenset(
                r[0] for r in self._db.execute("SELECT case_id FROM cases WHERE run_id = ?", (run_id,)))
            runs.append(run)
        return runs

    def groups_without_finished_trial(self, project: str, dataset: str, split: str = "test") -> list[dict]:
        """The first run of every group of repeated trials of one dataset and split in which no trial
        has finished, newest first. `finished_runs` cannot see these. Read-only."""
        rows = self._db.execute(
            "SELECT id FROM runs WHERE project = ? AND dataset = ? AND split = ? AND trial_group = id"
            " AND NOT EXISTS (SELECT 1 FROM runs m WHERE m.trial_group = runs.trial_group"
            " AND m.finished_at IS NOT NULL) ORDER BY id DESC",
            (project, dataset, split),
        ).fetchall()
        return [self.get_run(run_id) for (run_id,) in rows]

    def previous_finished_run(self, project: str, dataset: str, split: str, before_id: int) -> int | None:
        """The id of the newest finished run of this dataset and split that is older than `before_id`."""
        row = self._db.execute(
            "SELECT MAX(id) FROM runs WHERE project = ? AND dataset = ? AND split = ? AND id < ?"
            " AND finished_at IS NOT NULL", (project, dataset, split, before_id)).fetchone()
        return row[0]

    def case_ids_seen_outside_holdout(self, project: str, dataset: str) -> set[str]:
        """Every case id stored in some run of this dataset where it was not a holdout case: an
        agent may have read its input and output there. An old row without its own split goes by
        its run's split. Read-only."""
        rows = self._db.execute(
            "SELECT DISTINCT c.case_id FROM cases c JOIN runs r ON r.id = c.run_id"
            " WHERE r.project = ? AND r.dataset = ? AND COALESCE(c.split, r.split) != 'holdout'",
            (project, dataset))
        return {row[0] for row in rows}

    def holdout_attempts(self, project: str, dataset: str, since: float) -> list[float]:
        """Start times of the holdout runs an agent started through MCP since `since`, newest first.

        A group of repeated trials counts once, by its first run. Unfinished and failed runs count
        too: starting one used up the attempt. Reservations that are still held count as well.
        Read-only.
        """
        rows = self._db.execute(
            "SELECT started_at FROM runs WHERE project = ? AND dataset = ? AND split = 'holdout'"
            " AND origin = 'mcp' AND started_at >= ? AND (trial_group IS NULL OR trial_group = id)"
            " UNION ALL SELECT at FROM holdout_reservations WHERE project = ? AND dataset = ? AND at >= ?"
            " ORDER BY 1 DESC",
            (project, dataset, since, project, dataset, since),
        )
        return [row[0] for row in rows]

    def reserve_holdout(self, project: str, dataset: str, now: float, window: float) -> tuple[int | None, float | None]:
        """Take the holdout attempt if none is within `window` seconds before `now`: (reservation id, None).
        Otherwise (None, the time of the latest attempt). The check and the reservation happen in one
        write transaction, so several processes sharing this store cannot both get it."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            recent = self.holdout_attempts(project, dataset, now - window)
            if recent:
                self._db.execute("ROLLBACK")
                return None, recent[0]
            cursor = self._db.execute(
                "INSERT INTO holdout_reservations (project, dataset, at) VALUES (?,?,?)", (project, dataset, now))
            self._db.execute("COMMIT")
            return cursor.lastrowid, None
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def release_holdout(self, reservation: int | None) -> None:
        """Let go of a reservation. Once the run row exists it carries the attempt itself."""
        if reservation is None:
            return
        with self._db:
            self._db.execute("DELETE FROM holdout_reservations WHERE id = ?", (reservation,))

    def list_datasets(self) -> list[dict]:
        """Every (project, dataset, split) that has runs, with its run count, newest first. Read-only."""
        rows = self._db.execute(
            "SELECT project, dataset, split, COUNT(*) AS runs, MAX(id) AS latest_run FROM runs"
            " GROUP BY project, dataset, split ORDER BY MAX(id) DESC"
        )
        return [dict(r) for r in rows]

    def get_grades(self, run_id: int) -> list[dict]:
        rows = self._db.execute("SELECT * FROM grades WHERE run_id = ? ORDER BY rowid", (run_id,))
        return [dict(r) for r in rows]

    # --- human labels -------------------------------------------------------

    def set_label(self, run_id: int, case_id: str, passed: bool, grader: str = CASE_LABEL,
                  note: str | None = None) -> None:
        self.get_run(run_id)
        known = self._db.execute(
            "SELECT 1 FROM cases WHERE run_id = ? AND case_id = ?", (run_id, case_id)
        ).fetchone()
        if known is None:
            raise StoreError(f"run {run_id} has no case '{case_id}'")
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO labels (run_id, case_id, grader, passed, note, labeled_at)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, case_id, grader, int(passed), _utf8(note), time.time()),
            )

    def get_labels(self, run_id: int) -> list[dict]:
        rows = self._db.execute("SELECT * FROM labels WHERE run_id = ? ORDER BY rowid", (run_id,))
        return [dict(r) for r in rows]

    # --- baselines ----------------------------------------------------------

    def set_baseline(self, run_id: int) -> dict:
        run = self.get_run(run_id)
        # A group of trials is one baseline: store its first run, whichever trial was named.
        baseline_id = run["trial_group"] if run["trial_group"] is not None else run_id
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO baselines VALUES (?,?,?)", (run["project"], run["dataset"], baseline_id)
            )
        return run

    def get_baseline(self, project: str, dataset: str) -> int | None:
        row = self._db.execute(
            "SELECT run_id FROM baselines WHERE project = ? AND dataset = ?", (project, dataset)
        ).fetchone()
        return row[0] if row else None
