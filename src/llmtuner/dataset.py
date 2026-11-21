"""Datasets: one JSON object per line.

    {"id": "t1", "input": "...", "expected": "returns", "split": "test"}

`input` and `expected` may be a string or an object. `split` is "dev" (for
tuning prompts and graders) or "test" (for reporting) and defaults to "test".
`context` is optional text the system was given, shown to judge graders.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("dev", "test")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    expected: Any = None
    context: str | None = None
    split: str = "test"

    @property
    def input_text(self) -> str:
        return self.input if isinstance(self.input, str) else json.dumps(self.input, ensure_ascii=False)


def load_cases(path: Path) -> tuple[list[Case], str]:
    """Return the cases and a hash of the file, so runs record which data they used."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read dataset {path}: {exc}") from exc
    cases: list[Case] = []
    seen: set[str] = set()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path.name}: not valid UTF-8 ({exc.reason} at byte {exc.start})") from exc
    # Split on "\n" only: str.splitlines() also breaks on U+2028 and U+0085, which are legal inside JSON strings.
    for number, line in enumerate(text.split("\n"), start=1):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict) or "id" not in row or "input" not in row:
            raise DatasetError(f"{where}: each line needs \"id\" and \"input\"")
        case_id = str(row["id"])
        if case_id in seen:
            raise DatasetError(f"{where}: duplicate id '{case_id}'")
        seen.add(case_id)
        split = row.get("split", "test")
        if split not in SPLITS:
            raise DatasetError(f"{where}: split must be \"dev\" or \"test\"")
        context = row.get("context")
        if context is not None and not isinstance(context, str):
            raise DatasetError(f"{where}: context must be a string")
        cases.append(Case(case_id, row["input"], row.get("expected"), context, split))
    if not cases:
        raise DatasetError(f"{path}: dataset is empty")
    return cases, hashlib.sha256(raw).hexdigest()[:16]


def select_split(cases: list[Case], split: str) -> list[Case]:
    if split == "all":
        return cases
    if split not in SPLITS:
        raise DatasetError("split must be \"dev\", \"test\" or \"all\"")
    chosen = [c for c in cases if c.split == split]
    if not chosen:
        raise DatasetError(f"no cases in the '{split}' split")
    return chosen
