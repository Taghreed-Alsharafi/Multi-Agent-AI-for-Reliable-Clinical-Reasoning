"""Exact-match verification against locally available original benchmarks.

This module never guesses a label and never calls a model. It updates a row only
when one normalized exact question match yields one unambiguous source label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re
import unicodedata

import pandas as pd

from .mcq_hybrid import parse_options


@dataclass(frozen=True)
class BenchmarkSource:
    path: str | Path
    source_dataset: str
    source_split: str = ""
    question_column: str = "question"
    answer_column: str = "answer"
    id_column: str = "id"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _match_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _clean(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "examples", "questions", "rows"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported JSON structure in {path}")
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported benchmark file type: {path}")


def _source_letter(value: Any, question: str) -> str:
    raw = _clean(value)
    direct = re.fullmatch(r"\(?\s*([A-H])\s*\)?[.)]?", raw, flags=re.I)
    if direct:
        return direct.group(1).upper()
    if re.fullmatch(r"[0-7]", raw):
        return chr(ord("A") + int(raw))
    options = parse_options(question)
    normalized = _match_key(raw)
    matches = [
        letter for letter, text in options.items() if _match_key(text) == normalized
    ]
    return matches[0] if len(matches) == 1 else ""


def verify_gold_against_originals(
    evaluation_frame: pd.DataFrame,
    sources: list[BenchmarkSource],
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify gold labels using exact local benchmark matches only."""
    result = evaluation_frame.copy()
    for column in (
        "source_dataset",
        "source_split",
        "source_question_id",
        "original_gold_letter",
    ):
        if column not in result:
            result[column] = ""
    if "gold_status" not in result:
        result["gold_status"] = "SOURCE_DERIVED_UNVERIFIED"

    index: dict[str, list[dict[str, str]]] = {}
    source_errors: list[dict[str, str]] = []
    for source in sources:
        path = Path(source.path).resolve()
        if not path.exists():
            source_errors.append(
                {
                    "source_dataset": source.source_dataset,
                    "path": str(path),
                    "error": "file_not_found",
                }
            )
            continue
        frame = _read_source(path)
        required = {source.question_column, source.answer_column}
        if not required.issubset(frame.columns):
            source_errors.append(
                {
                    "source_dataset": source.source_dataset,
                    "path": str(path),
                    "error": f"missing_columns:{sorted(required - set(frame.columns))}",
                }
            )
            continue
        for row_number, row in frame.iterrows():
            question = _clean(row[source.question_column])
            letter = _source_letter(row[source.answer_column], question)
            if not question or not letter:
                continue
            key = _match_key(question)
            index.setdefault(key, []).append(
                {
                    "source_dataset": source.source_dataset,
                    "source_split": source.source_split,
                    "source_question_id": _clean(
                        row.get(source.id_column, row_number)
                    ),
                    "original_gold_letter": letter,
                }
            )

    audit_rows: list[dict[str, Any]] = []
    for row_index, row in result.iterrows():
        question = _clean(row.get("question", ""))
        if not question:
            question = _clean(row.get("instruction", ""))
        matches = index.get(_match_key(question), [])
        distinct = {
            (match["source_dataset"], match["source_question_id"], match["original_gold_letter"])
            for match in matches
        }
        supplied_options = parse_options(_clean(row.get("instruction", "")))
        matched_letter = matches[0]["original_gold_letter"] if matches else ""
        verified = len(distinct) == 1 and (
            not supplied_options or matched_letter in supplied_options
        )
        if verified:
            match = matches[0]
            result.at[row_index, "source_dataset"] = match["source_dataset"]
            result.at[row_index, "source_split"] = match["source_split"]
            result.at[row_index, "source_question_id"] = match["source_question_id"]
            result.at[row_index, "original_gold_letter"] = match["original_gold_letter"]
            result.at[row_index, "gold_letter"] = match["original_gold_letter"]
            result.at[row_index, "output"] = match["original_gold_letter"]
            result.at[row_index, "gold_status"] = "ORIGINAL_BENCHMARK_VERIFIED"
        audit_rows.append(
            {
                "fixed_case_id": row.get("fixed_case_id", ""),
                "exact_match_count": len(matches),
                "distinct_match_count": len(distinct),
                "verified": verified,
                "result_status": (
                    "ORIGINAL_BENCHMARK_VERIFIED"
                    if verified
                    else "SOURCE_DERIVED_UNVERIFIED"
                ),
            }
        )

    audit = pd.DataFrame(audit_rows)
    if source_errors:
        audit.attrs["source_errors"] = source_errors
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(path, index=False)
    return result, audit
