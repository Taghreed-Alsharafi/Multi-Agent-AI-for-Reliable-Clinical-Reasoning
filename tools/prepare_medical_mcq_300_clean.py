#!/usr/bin/env python
"""Build the direct-label 300-case evaluation and QC master CSVs.

The source-derived labels are preserved but deliberately marked unverified.
No model is used and no original-benchmark verification is claimed.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mcq_hybrid import parse_options
from evaluation.mcq_hybrid_v2 import (
    _MISSING_IMAGE_PATTERN,
    _normalize_option_text,
    assign_case_ids,
    build_question,
)


SOURCE = ROOT / "data" / "medical_mcq.csv"
EVAL_READY = ROOT / "data" / "medical_mcq_300_EVAL_READY.csv"
QC_MASTER = ROOT / "data" / "medical_mcq_300_QC_MASTER.csv"


def _question_stem(instruction: str) -> str:
    text = str(instruction or "").strip()
    text = re.sub(
        r"(?i)^please answer the following multiple-choice question:\s*",
        "",
        text,
    )
    return re.split(r"(?mi)^\s*(?:\(A\)|A[.)\-:])\s+", text, maxsplit=1)[0].strip()


def main() -> None:
    source = pd.read_csv(SOURCE, keep_default_na=False)
    source = assign_case_ids(source)
    duplicate_sizes = source.groupby("base_question_hash")["base_question_hash"].transform("size")
    rows: list[dict[str, object]] = []

    for _, row in source.iterrows():
        question_for_model = build_question(row)
        options = parse_options(question_for_model)
        gold = str(row.get("output", "")).strip().upper()
        answer_text = str(row.get("answer_text", "")).strip()
        reasons: list[str] = []
        if not question_for_model:
            reasons.append("MISSING_QUESTION")
        if len(options) < 2 or list(options) != [
            chr(ord("A") + index) for index in range(len(options))
        ]:
            reasons.append("MALFORMED_OPTIONS")
        if gold not in options:
            reasons.append("INVALID_DIRECT_GOLD")
        if gold in options and answer_text:
            expected = _normalize_option_text(options[gold])
            observed = _normalize_option_text(answer_text)
            if not (
                expected == observed
                or (expected and expected in observed)
                or (observed and observed in expected)
            ):
                reasons.append("GOLD_TEXT_CONFLICT")
        if _MISSING_IMAGE_PATTERN.search(question_for_model):
            reasons.append("MISSING_REQUIRED_IMAGE")
        if int(row["duplicate_index"]) > 0:
            reasons.append("EXACT_DUPLICATE")
        if question_for_model.lower().count(
            "please answer the following multiple-choice question"
        ) > 1:
            reasons.append("EMBEDDED_SECOND_QUESTION")

        record: dict[str, object] = {
            "fixed_case_id": row["fixed_case_id"],
            "row_id": row.get("row_id", ""),
            "source_case_id": row.get("case_id", ""),
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": gold,
            "gold_letter": gold,
            "gold_answer_text": answer_text,
            "gold_status": "SOURCE_DERIVED_UNVERIFIED",
            "include_for_evaluation": int(not reasons),
            "qc_reasons": ";".join(reasons),
            "correction_note": "",
            "correction_source": "",
            "source_dataset": "",
            "source_split": "",
            "source_question_id": "",
            "original_gold_letter": "",
            "question": _question_stem(row.get("instruction", "")),
            "duplicate_index": int(row["duplicate_index"]),
            "duplicate_group_size": int(duplicate_sizes.loc[row.name]),
            "legacy_answer_status": row.get("answer_status", ""),
            "legacy_answer_method": row.get("answer_method", ""),
            "legacy_answer_confidence": row.get("answer_confidence", ""),
        }
        for letter in "ABCDEFGH":
            record[f"option_{letter}"] = options.get(letter, "")
        rows.append(record)

    master = pd.DataFrame(rows)
    evaluation = master[master["include_for_evaluation"].astype(bool)].copy()
    master.to_csv(QC_MASTER, index=False)
    evaluation.to_csv(EVAL_READY, index=False)

    print(f"QC master rows: {len(master)}")
    print(f"Evaluation-ready rows: {len(evaluation)}")
    print(f"Excluded rows: {len(master) - len(evaluation)}")
    print("Gold status counts:", master["gold_status"].value_counts().to_dict())
    print("QC reasons:", master["qc_reasons"].replace("", "NONE").value_counts().to_dict())
    print("Saved:", EVAL_READY)
    print("Saved:", QC_MASTER)


if __name__ == "__main__":
    main()
