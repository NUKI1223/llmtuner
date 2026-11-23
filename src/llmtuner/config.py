"""Project config: which targets, datasets and graders a project has.

A project describes itself in one `llmtuner.toml` file. Commands are always
lists of arguments and never go through a shell.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "llmtuner.toml"

# Pseudo-grader name the runner records when the target itself fails; no real grader may use it.
TARGET_GRADER = "target"

MAX_TIMEOUT = 86400  # one day, in seconds

# Grader type -> option names it accepts (besides `type` and `required`).
GRADER_OPTIONS: dict[str, set[str]] = {
    "exact": {"field", "ignore_case"},
    "contains_all": {"field", "values"},
    "not_contains": {"values"},
    "regex": {"pattern"},
    "max_words": {"limit"},
    "nonempty": set(),
    "json": {"required_keys"},
    "json_match": {"field", "ordered"},
    "judge": {"command", "question", "fail_on", "timeout", "include_expected"},
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Target:
    name: str
    command: tuple[str, ...]
    timeout: float = 120.0
    model: str | None = None
    version_command: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Grader:
    name: str
    type: str
    required: bool = True
    options: dict = field(default_factory=dict)

    def describe(self) -> dict:
        info = {"name": self.name, "type": self.type, "required": self.required}
        if self.type == "judge":
            info["question"] = self.options["question"]
        return info


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    graders: tuple[str, ...] | None = None  # None means every grader in the config
    # The change log of this dataset's cases; defaults to the dataset path with `.changes.jsonl`.
    changes: Path | None = None

    def __post_init__(self) -> None:
        if self.changes is None:
            object.__setattr__(self, "changes", self.path.with_suffix(".changes.jsonl"))


@dataclass(frozen=True)
class Config:
    project: str
    root: Path
    targets: dict[str, Target]
    datasets: dict[str, Dataset]
    graders: dict[str, Grader]

    def target(self, name: str) -> Target:
        return _lookup(self.targets, name, "target")

    def dataset(self, name: str) -> Dataset:
        return _lookup(self.datasets, name, "dataset")

    def graders_for(self, dataset: Dataset) -> list[Grader]:
        names = dataset.graders if dataset.graders is not None else tuple(self.graders)
        return [self.graders[n] for n in names]

    def describe(self) -> dict:
        return {
            "project": self.project,
            "root": str(self.root),
            "targets": {
                t.name: {"command": list(t.command), "model": t.model, "timeout": t.timeout}
                for t in self.targets.values()
            },
            "datasets": {
                d.name: {"path": str(d.path), "changes": str(d.changes), "graders": [g.name for g in self.graders_for(d)]}
                for d in self.datasets.values()
            },
            "graders": [g.describe() for g in self.graders.values()],
        }


def _lookup(table: dict, name: str, kind: str):
    if name not in table:
        known = ", ".join(sorted(table)) or "none defined"
        raise ConfigError(f"unknown {kind} '{name}' (available: {known})")
    return table[name]


def _command(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(a, str) and a for a in value):
        raise ConfigError(f"{where}: command must be a non-empty list of strings, e.g. [\"python3\", \"app.py\"]")
    return tuple(value)


def _timeout(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ConfigError(f"{where}: timeout must be a positive number of seconds")
    try:
        too_big = not math.isfinite(value) or value > MAX_TIMEOUT
    except OverflowError:  # an integer too large for a float
        too_big = True
    if too_big:
        raise ConfigError(f"{where}: timeout must be a finite number of at most {MAX_TIMEOUT} seconds")
    return float(value)


def _table(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _grader(name: str, raw: dict) -> Grader:
    where = f"[graders.{name}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table")
    if name == TARGET_GRADER:
        raise ConfigError(f"{where}: '{TARGET_GRADER}' is reserved for target failures; pick another name")
    if not isinstance(raw.get("required", True), bool):
        raise ConfigError(f"{where}: required must be true or false")
    kind = raw.get("type")
    if kind not in GRADER_OPTIONS:
        raise ConfigError(f"{where}: type must be one of {', '.join(sorted(GRADER_OPTIONS))}")
    options = {k: v for k, v in raw.items() if k not in ("type", "required")}
    unknown = set(options) - GRADER_OPTIONS[kind]
    if unknown:
        raise ConfigError(f"{where}: unknown option(s) for type '{kind}': {', '.join(sorted(unknown))}")

    def need(key: str, kinds) -> None:
        if not isinstance(options.get(key), kinds):
            raise ConfigError(f"{where}: type '{kind}' needs '{key}'")

    if kind == "contains_all" and ("values" in options) == ("field" in options):
        raise ConfigError(f"{where}: give exactly one of 'values' (fixed list) or 'field' (from each case)")
    if kind == "not_contains" or (kind == "contains_all" and "values" in options):
        need("values", list)
    if kind == "regex":
        need("pattern", str)
        try:
            re.compile(options["pattern"])
        except re.error as exc:
            raise ConfigError(f"{where}: bad pattern: {exc}") from exc
    if kind == "max_words":
        need("limit", int)
    if kind == "json_match":
        if not isinstance(options.get("ordered", False), bool):
            raise ConfigError(f"{where}: ordered must be true or false")
        if "field" in options:
            need("field", str)
    if kind == "judge":
        need("question", str)
        options["command"] = _command(options.get("command"), where)
        if "timeout" in options:
            options["timeout"] = _timeout(options["timeout"], where)
        options.setdefault("fail_on", "yes")
        if options["fail_on"] not in ("yes", "no"):
            raise ConfigError(f"{where}: fail_on must be \"yes\" or \"no\"")
    return Grader(name=name, type=kind, required=bool(raw.get("required", True)), options=options)


def find_config(path: str | Path | None = None) -> Path:
    """Resolve a config file from a file path, a directory, or the current directory."""
    candidate = Path(path) if path else Path.cwd()
    if candidate.is_dir():
        candidate = candidate / CONFIG_NAME
    if not candidate.is_file():
        raise ConfigError(f"no config found at {candidate}")
    return candidate.resolve()


def load_config(path: str | Path | None = None) -> Config:
    config_path = find_config(path)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    root = config_path.parent

    project = raw.get("project")
    if not isinstance(project, str) or not project.strip():
        raise ConfigError(f"{config_path}: set project = \"name\" at the top")

    targets = {}
    for name, body in _table(raw, "targets").items():
        where = f"[targets.{name}]"
        if not isinstance(body, dict):
            raise ConfigError(f"{where} must be a table")
        version = body.get("version_command")
        targets[name] = Target(
            name=name,
            command=_command(body.get("command"), where),
            timeout=_timeout(body.get("timeout", 120.0), where),
            model=body.get("model"),
            version_command=_command(version, where) if version is not None else None,
        )

    graders = {name: _grader(name, body) for name, body in _table(raw, "graders").items()}

    datasets = {}
    for name, body in _table(raw, "datasets").items():
        where = f"[datasets.{name}]"
        if not isinstance(body, dict) or not isinstance(body.get("path"), str):
            raise ConfigError(f"{where}: needs path = \"file.jsonl\"")
        chosen = body.get("graders")
        if chosen is not None:
            if not isinstance(chosen, list) or not all(isinstance(g, str) for g in chosen):
                raise ConfigError(f"{where}: graders must be a list of grader names")
            missing = [g for g in chosen if g not in graders]
            if missing:
                raise ConfigError(f"{where}: unknown grader(s): {', '.join(missing)}")
            chosen = tuple(chosen)
        changes = body.get("changes")
        if changes is not None and (not isinstance(changes, str) or not changes):
            raise ConfigError(f"{where}: changes must be a file path like \"artifacts/changes.jsonl\"")
        path = (root / body["path"]).resolve()
        changes_path = (root / changes).resolve() if changes is not None else None
        if changes_path == path:
            raise ConfigError(f"{where}: changes must not be the dataset file itself")
        datasets[name] = Dataset(name=name, path=path, graders=chosen, changes=changes_path)

    # Edits lock on the change log, so datasets that read one file must share one log.
    logs: dict[Path, Dataset] = {}
    files = {d.path: d for d in datasets.values()}
    for d in datasets.values():
        if d.changes in files:
            raise ConfigError(f"[datasets.{d.name}]: changes points at the file of [datasets.{files[d.changes].name}];"
                              " a change log must be a file of its own")
        first = logs.setdefault(d.path, d)
        if first.changes != d.changes:
            raise ConfigError(f"[datasets.{d.name}]: reads the same file as [datasets.{first.name}]"
                              " but keeps a different change log; give both the same `changes`")

    return Config(project=project.strip(), root=root, targets=targets, datasets=datasets, graders=graders)
