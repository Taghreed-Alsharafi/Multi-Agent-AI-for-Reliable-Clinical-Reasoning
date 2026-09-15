from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, TypeVar
from pydantic import BaseModel
from .schema import EvaluationCase, PredictionRecord

T = TypeVar("T", bound=BaseModel)

def _read_jsonl(path: str | Path, model: type[T]) -> list[T]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid {path}:{line_number}: {exc}") from exc
    return rows

def read_cases(path: str | Path) -> list[EvaluationCase]:
    return _read_jsonl(path, EvaluationCase)

def read_predictions(path: str | Path) -> list[PredictionRecord]:
    return _read_jsonl(path, PredictionRecord)

def append_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")

def write_json(path: str | Path, value: object) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
