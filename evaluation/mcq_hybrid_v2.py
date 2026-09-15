"""V2 adaptive clinical MCQ evaluation pipeline.

The V2 architecture is intentionally separate from ``mcq_hybrid.py`` so prior
results remain reproducible:

    dataset QC -> router -> independent specialists -> optional critic
    -> exactly one GPT-5 judge -> deterministic validator

No model is called after the GPT-5 judge.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
import asyncio
import hashlib
import json
import math
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

from .mcq_hybrid import parse_options, reference_letter
from .mcq_v2_prompts import (
    CRITIC_PROMPT,
    JUDGE_PROMPT,
    PROMPT_VERSIONS,
    ROUTER_PROMPT,
    SINGLE_PROMPT,
    SPECIALIST_PROMPT,
    specialty_instruction,
)


ARCHITECTURE_VERSION = "2.1.0"
MAX_SPECIALISTS_HARD = 4
DIFFICULTIES = ("simple", "moderate", "complex", "very_complex")
DISAGREEMENT_LEVELS = ("low", "moderate", "high")
SPECIALIST_ROLES = ("lead", "supporting")
ORIGINAL_GOLD_STATUSES = {
    "ORIGINAL_BENCHMARK_VERIFIED",
    "CORRECTED_WITH_AUTHORITATIVE_SOURCE",
}

# These rates are configuration defaults, not hidden billing assumptions. Every
# run stores the exact table and source label in metadata so it can be replaced.
DEFAULT_PRICING_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5": {"input": 1.25, "output": 10.00},
}
DEFAULT_PRICING_SOURCE = "OpenAI public model pages checked 2026-08-11; override before publication"


def _configure_platform_trust_store() -> bool:
    """Use the Windows certificate store without weakening TLS verification."""
    if os.name != "nt":
        return False
    try:
        import truststore
    except ImportError as exc:
        raise RuntimeError(
            "Windows HTTPS trust-store support is missing. Install project "
            "dependencies with `python -m pip install -e .[evaluation]`."
        ) from exc
    truststore.inject_into_ssl()
    return True


def _exception_chain_message(exc: BaseException) -> str:
    """Expose nested transport causes while avoiding a full noisy traceback."""
    messages = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    current: BaseException | None = exc
    while current is not None:
        current = current.__cause__ or current.__context__
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        detail = f"{type(current).__name__}: {current}"
        if detail not in messages:
            messages.append(detail)
    return " | caused by: ".join(messages)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def _normalize_identity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _clean_cell(value))
    return re.sub(r"\s+", " ", text).strip()


def _normalize_option_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    # Literal option text such as "None" is a valid answer choice, not a
    # missing-value sentinel.
    text = unicodedata.normalize("NFKC", str(value).strip()).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = _clean_cell(value).casefold()
    if text in {"1", "true", "yes", "y", "include", "included"}:
        return True
    if text in {"0", "false", "no", "n", "exclude", "excluded"}:
        return False
    return default


def _bool_column(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(lambda value: _parse_bool(value, default)).astype(bool)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def case_id_components(instruction: Any, additional_input: Any) -> tuple[str, str]:
    """Return normalized identity text and its label-independent base hash."""
    identity = f"{_normalize_identity(instruction)}\x1f{_normalize_identity(additional_input)}"
    return identity, _sha256_text(identity)


def assign_case_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic IDs that do not depend on the gold answer."""
    result = frame.copy().reset_index(drop=True)
    components = [
        case_id_components(row.get("instruction", ""), row.get("input", ""))
        for _, row in result.iterrows()
    ]
    result["normalized_question_identity"] = [item[0] for item in components]
    result["base_question_hash"] = [item[1] for item in components]
    result["duplicate_index"] = result.groupby("base_question_hash", sort=False).cumcount()
    result["fixed_case_id"] = [
        _sha256_text(f"{base_hash}:{duplicate_index}")
        for base_hash, duplicate_index in zip(
            result["base_question_hash"], result["duplicate_index"]
        )
    ]
    return result


def validate_or_assign_case_ids(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Use cleaned fixed IDs when valid; otherwise generate label-free IDs."""
    result = frame.copy().reset_index(drop=True)
    components = [
        case_id_components(row.get("instruction", ""), row.get("input", ""))
        for _, row in result.iterrows()
    ]
    result["normalized_question_identity"] = [item[0] for item in components]
    result["base_question_hash"] = [item[1] for item in components]
    result["duplicate_index"] = result.groupby("base_question_hash", sort=False).cumcount()

    if "fixed_case_id" in result:
        supplied = result["fixed_case_id"].map(_clean_cell)
        valid = (
            supplied.str.fullmatch(r"[0-9a-fA-F]{64}").all()
            and not supplied.duplicated().any()
            and supplied.ne("").all()
        )
        if valid:
            result["fixed_case_id"] = supplied.str.lower()
            return result, "cleaned_dataset_fixed_case_id"

    result["fixed_case_id"] = [
        _sha256_text(f"{base_hash}:{duplicate_index}")
        for base_hash, duplicate_index in zip(
            result["base_question_hash"], result["duplicate_index"]
        )
    ]
    return result, "generated_from_question_identity"


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint question identity only, never labels."""
    ids = sorted(frame["fixed_case_id"].astype(str))
    return _sha256_text("\n".join(ids))


def label_fingerprint(frame: pd.DataFrame) -> str:
    pairs = sorted(
        f"{case_id}:{_clean_cell(letter).upper()}"
        for case_id, letter in zip(frame["fixed_case_id"], frame["reference_letter"])
    )
    return _sha256_text("\n".join(pairs))


def build_question(row: Mapping[str, Any] | pd.Series) -> str:
    instruction = _clean_cell(row.get("instruction", ""))
    additional = _clean_cell(row.get("input", ""))
    if additional:
        return f"{instruction}\n\nAdditional input:\n{additional}".strip()
    return instruction


_MISSING_IMAGE_PATTERN = re.compile(
    r"(?i)(?:shown|pictured|depicted)\s+(?:above|below|here)|"
    r"(?:image|figure|photograph|radiograph|ecg|ekg)\s+(?:above|below)|"
    r"(?:following|provided|attached)\s+(?:image|figure|photograph|radiograph|ecg|ekg)|"
    r"see\s+(?:the\s+)?(?:image|figure|photograph|radiograph)"
)


def _gold_from_source(question: str, output: Any) -> tuple[str, str, str]:
    """Extract a gold letter conservatively and report ambiguity."""
    raw = _clean_cell(output)
    direct = re.fullmatch(r"\(?\s*([A-H])\s*\)?[.)]?", raw, flags=re.I)
    if direct:
        return direct.group(1).upper(), "literal_letter", ""

    explicit = {
        match.upper()
        for match in re.findall(
            r"(?i)\b(?:final|correct|best)\s+(?:answer|option|choice)\s*(?:is|:|=|-)?\s*\(?([A-H])\)?",
            raw,
        )
    }
    if len(explicit) > 1:
        return "", "unresolved", "conflicting_final_answers"
    if raw.lower().count("please answer the following multiple-choice question") > 0:
        return "", "unresolved", "reference_contains_embedded_question"

    parsed = reference_letter(question, raw)
    if parsed:
        return parsed, "deterministic_verbose_parse", ""
    return "", "unresolved", "unresolved_gold"


@dataclass
class DatasetQCResult:
    data: pd.DataFrame
    audit: pd.DataFrame
    eligible: pd.DataFrame
    excluded: pd.DataFrame
    summary: dict[str, Any]
    dataset_fingerprint: str
    label_fingerprint: str
    audit_path: Path | None = None
    summary_path: Path | None = None


def run_dataset_qc(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    qc_master_path: str | Path | None = None,
    require_original_gold: bool = False,
) -> DatasetQCResult:
    """Load and validate the shared evaluation dataframe before API calls.

    Direct ``gold_letter`` is authoritative when present. A one-letter ``output``
    is the second choice; verbose parsing is a legacy-only fallback.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path, keep_default_na=False)
    required = {"instruction", "input", "output"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    for column in required | {
        "gold_letter",
        "gold_answer_text",
        "gold_status",
        "qc_reasons",
        "correction_note",
        "correction_source",
    } & set(data.columns):
        data[column] = data[column].map(_clean_cell)
    if "source_row_number" not in data:
        data.insert(0, "source_row_number", np.arange(1, len(data) + 1))
    data, case_id_source = validate_or_assign_case_ids(data)

    if "gold_status" not in data:
        # Legacy confidence labels are extraction metadata, not independent
        # benchmark verification.
        data["gold_status"] = "SOURCE_DERIVED_UNVERIFIED"
    else:
        data["gold_status"] = data["gold_status"].replace(
            "", "SOURCE_DERIVED_UNVERIFIED"
        )

    duplicate_sizes = data.groupby("base_question_hash")["base_question_hash"].transform("size")
    audit_rows: list[dict[str, Any]] = []
    references: list[str] = []

    for _, row in data.iterrows():
        question = build_question(row)
        options = parse_options(question)
        letters = list(options)
        expected_letters = [chr(ord("A") + i) for i in range(len(letters))]
        option_values = [_normalize_option_text(value) for value in options.values()]
        direct_gold = _clean_cell(row.get("gold_letter", "")).upper()
        output_gold = _clean_cell(row.get("output", "")).upper()
        if direct_gold:
            gold, gold_method, gold_issue = direct_gold, "direct_gold_letter", ""
        elif re.fullmatch(r"[A-H]", output_gold):
            gold, gold_method, gold_issue = output_gold, "direct_output_letter", ""
        else:
            gold, gold_method, gold_issue = _gold_from_source(
                question, row.get("output", "")
            )
            if gold:
                gold_method = "legacy_verbose_parser"
        references.append(gold)

        valid_question = bool(_normalize_identity(question))
        valid_options = (
            2 <= len(options) <= 8
            and letters == expected_letters
            and len(option_values) == len(set(option_values))
            and all(option_values)
        )
        valid_gold = bool(gold and gold in options)
        missing_image = bool(_MISSING_IMAGE_PATTERN.search(question))
        embedded_second_question = (
            question.lower().count("please answer the following multiple-choice question") > 1
        )
        duplicate_question = int(row["duplicate_index"]) > 0
        source_included = (
            _parse_bool(row.get("include_for_evaluation"), default=True)
            if "include_for_evaluation" in data
            else True
        )
        gold_status = _clean_cell(row.get("gold_status", "")) or "SOURCE_DERIVED_UNVERIFIED"

        answer_text = _clean_cell(
            row.get("gold_answer_text", row.get("answer_text", ""))
        )
        gold_text_consistent = True
        if answer_text and valid_gold:
            expected_text = _normalize_option_text(options[gold])
            observed_text = _normalize_option_text(answer_text)
            gold_text_consistent = bool(
                observed_text == expected_text
                or (observed_text and observed_text in expected_text)
                or (expected_text and expected_text in observed_text)
            )

        reasons: list[str] = []
        if not source_included:
            source_reason = _clean_cell(row.get("qc_reasons", ""))
            reasons.append(f"source_excluded:{source_reason or 'include_for_evaluation_false'}")
        if not valid_question:
            reasons.append("missing_question")
        if not valid_options:
            reasons.append("malformed_options")
        if not valid_gold:
            reasons.append(gold_issue or "invalid_gold")
        if not gold_text_consistent:
            reasons.append("gold_answer_text_conflict")
        if missing_image:
            reasons.append("missing_required_image")
        if embedded_second_question:
            reasons.append("embedded_second_question")
        if duplicate_question:
            reasons.append("duplicate_question")
        if require_original_gold and gold_status not in ORIGINAL_GOLD_STATUSES:
            reasons.append("gold_not_originally_verified")

        eligible = not reasons
        audit_rows.append(
            {
                "source_row_number": int(row["source_row_number"]),
                "fixed_case_id": row["fixed_case_id"],
                "base_question_hash": row["base_question_hash"],
                "duplicate_index": int(row["duplicate_index"]),
                "duplicate_group_size": int(duplicate_sizes.loc[row.name]),
                "n_options": len(options),
                "option_letters": "".join(letters),
                "options_json": json.dumps(options, ensure_ascii=False),
                "reference_letter": gold,
                "gold_method": gold_method,
                "valid_question": valid_question,
                "valid_options": valid_options,
                "valid_gold": valid_gold,
                "gold_text_consistent": gold_text_consistent,
                "missing_required_image": missing_image,
                "embedded_second_question": embedded_second_question,
                "duplicate_question": duplicate_question,
                "include_for_evaluation_source": source_included,
                "gold_status": gold_status,
                "qc_reasons_source": _clean_cell(row.get("qc_reasons", "")),
                "manual_label_review_recommended": gold_status not in ORIGINAL_GOLD_STATUSES,
                "eligible_for_evaluation": eligible,
                "exclusion_reason": ";".join(reasons),
                "instruction_preview": question[:500],
            }
        )

    data["reference_letter"] = references
    data["reference_method"] = [row["gold_method"] for row in audit_rows]
    audit = pd.DataFrame(audit_rows)
    eligibility = audit.set_index("fixed_case_id")["eligible_for_evaluation"]
    data["eligible_for_evaluation"] = data["fixed_case_id"].map(eligibility).astype(bool)
    eligible = data[data["eligible_for_evaluation"]].copy().reset_index(drop=True)
    excluded = data[~data["eligible_for_evaluation"]].copy().reset_index(drop=True)

    # Fingerprints describe the actual evaluation cohort after all QC filters.
    fingerprint = dataset_fingerprint(eligible)
    current_label_fingerprint = label_fingerprint(eligible)
    data["dataset_fingerprint"] = fingerprint
    data["label_fingerprint"] = current_label_fingerprint
    eligible["dataset_fingerprint"] = fingerprint
    eligible["label_fingerprint"] = current_label_fingerprint

    reason_counts = (
        audit.loc[audit["exclusion_reason"].ne(""), "exclusion_reason"]
        .str.split(";")
        .explode()
        .value_counts()
        .to_dict()
    )
    answer_distribution = eligible["reference_letter"].value_counts().sort_index().to_dict()
    status_distribution = audit["gold_status"].replace("", "UNSPECIFIED").value_counts().to_dict()

    master_reporting: dict[str, Any] = {}
    if qc_master_path is not None and Path(qc_master_path).exists():
        master = pd.read_csv(qc_master_path, keep_default_na=False)
        include_values = (
            master.get("include_for_evaluation", pd.Series(True, index=master.index))
            .map(lambda value: _parse_bool(value, True))
        )
        qc_reason_counts = (
            master.get("qc_reasons", pd.Series("", index=master.index))
            .replace("", "NONE")
            .value_counts()
            .to_dict()
        )
        evaluation_ids = set(data["fixed_case_id"].astype(str))
        if "fixed_case_id" in master:
            master_eligible_ids = set(
                master.loc[include_values, "fixed_case_id"].astype(str)
            )
            master_matches_evaluation = master_eligible_ids == evaluation_ids
        else:
            master_matches_evaluation = False
        reported_master_exclusions = int((~include_values).sum())
        master_reporting = {
            "qc_master_file": str(Path(qc_master_path).resolve()),
            "qc_master_total_rows": int(len(master)),
            "qc_master_included_rows": int(include_values.sum()),
            "qc_master_excluded_rows": (
                reported_master_exclusions if master_matches_evaluation else None
            ),
            "qc_master_reported_excluded_rows": reported_master_exclusions,
            "qc_master_matches_evaluation": master_matches_evaluation,
            "qc_master_warning": (
                "" if master_matches_evaluation else
                "QC master eligible IDs do not match this evaluation-ready file; "
                "its exclusion counts are informational only."
            ),
            "qc_master_gold_status_distribution": {
                str(k): int(v)
                for k, v in master.get(
                    "gold_status", pd.Series("UNSPECIFIED", index=master.index)
                ).value_counts().to_dict().items()
            },
            "qc_master_reason_distribution": {
                str(k): int(v) for k, v in qc_reason_counts.items()
            },
        }
    summary: dict[str, Any] = {
        "source_file": str(path),
        "total_rows": int(len(data)),
        "source_include_for_evaluation_rows": int(
            audit["include_for_evaluation_source"].sum()
        ),
        "valid_questions": int(audit["valid_question"].sum()),
        "valid_answer_options": int(audit["valid_options"].sum()),
        "valid_gold_labels": int(audit["valid_gold"].sum()),
        "gold_text_consistent": int(audit["gold_text_consistent"].sum()),
        "duplicate_questions": int(audit["duplicate_question"].sum()),
        "missing_image_questions": int(audit["missing_required_image"].sum()),
        "embedded_second_questions": int(audit["embedded_second_question"].sum()),
        "unresolved_gold_answers": int((~audit["valid_gold"]).sum()),
        "eligible_rows": int(audit["eligible_for_evaluation"].sum()),
        "excluded_rows": int((~audit["eligible_for_evaluation"]).sum()),
        "manual_label_review_recommended": int(audit["manual_label_review_recommended"].sum()),
        "answer_distribution": {str(k): int(v) for k, v in answer_distribution.items()},
        "gold_status_distribution": {
            str(k): int(v) for k, v in status_distribution.items()
        },
        "exclusion_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "dataset_fingerprint": fingerprint,
        "label_fingerprint": current_label_fingerprint,
        "case_ids_exclude_gold": True,
        "case_id_source": case_id_source,
        "require_original_gold": require_original_gold,
        "original_gold_eligible_rows": int(
            audit["gold_status"].isin(ORIGINAL_GOLD_STATUSES).sum()
        ),
        **master_reporting,
    }

    audit_path: Path | None = None
    summary_path: Path | None = None
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        audit_path = output / "dataset_qc.csv"
        summary_path = output / "dataset_qc_summary.json"
        audit.to_csv(audit_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        eligible.to_csv(output / "dataset_qc_eligible.csv", index=False)
        if not excluded.empty:
            excluded_ids = set(excluded["fixed_case_id"])
            audit[audit["fixed_case_id"].isin(excluded_ids)].to_csv(
                output / "unresolved_reference_cases_all.csv", index=False
            )

    return DatasetQCResult(
        data=data,
        audit=audit,
        eligible=eligible,
        excluded=excluded,
        summary=summary,
        dataset_fingerprint=fingerprint,
        label_fingerprint=current_label_fingerprint,
        audit_path=audit_path,
        summary_path=summary_path,
    )


def load_evaluation_dataset(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    qc_master_path: str | Path | None = None,
    require_original_gold: bool = False,
) -> DatasetQCResult:
    """Shared cleaned-data loader used by both single and multi notebooks."""
    return run_dataset_qc(
        csv_path,
        output_dir,
        qc_master_path=qc_master_path,
        require_original_gold=require_original_gold,
    )


def _migrate_legacy_membership(
    eligible: pd.DataFrame,
    legacy_root: Path,
) -> tuple[list[str], list[str]] | None:
    dev_path = legacy_root / "development_20.csv"
    test_path = legacy_root / "test_80.csv"
    if not dev_path.exists() or not test_path.exists():
        return None

    dev = assign_case_ids(pd.read_csv(dev_path, keep_default_na=False))
    test = assign_case_ids(pd.read_csv(test_path, keep_default_na=False))
    current_map = dict(zip(eligible["normalized_question_identity"], eligible["fixed_case_id"]))
    if len(current_map) != len(eligible):
        return None

    dev_ids = [current_map.get(identity, "") for identity in dev["normalized_question_identity"]]
    test_ids = [current_map.get(identity, "") for identity in test["normalized_question_identity"]]
    if not all(dev_ids) or not all(test_ids):
        return None
    if set(dev_ids) | set(test_ids) != set(eligible["fixed_case_id"]):
        return None
    if set(dev_ids) & set(test_ids):
        return None
    return dev_ids, test_ids


def _safe_v2_split_project_root(root: Path, dataset_name: str, fingerprint: str) -> Path:
    """Select a Windows-safe shared split location for nested repository copies."""
    versioned = f"{dataset_name}__{fingerprint[:12]}"
    prospective = root / "data" / "fixed_splits_v2" / versioned / "split_manifest.json"
    if os.name != "nt" or len(str(prospective)) < 248:
        return root

    canonical_name = "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"
    candidates = [
        root.parent / canonical_name,
        root.parent.parent / canonical_name,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        safe_path = candidate / "data" / "fixed_splits_v2" / versioned / "split_manifest.json"
        if (candidate / "evaluation" / "mcq_hybrid_v2.py").exists() and len(str(safe_path)) < 248:
            print(f"Nested project path is too long for Windows split files. Using: {candidate}")
            return candidate
    raise RuntimeError(
        f"V2 split path is too long for reliable Windows writes: {prospective}. "
        "Move the repository to a shorter directory."
    )


def ensure_fixed_split_v2(
    qc: DatasetQCResult,
    project_root: str | Path,
    dataset_name: str = "medical_mcq",
    development_fraction: float = 0.20,
    seed: int = 2026,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path]:
    """Create or validate a leakage-safe V2 split manifest."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    data = qc.eligible.copy()
    if len(data) < 2:
        raise ValueError("At least two QC-eligible cases are required")

    requested_root = Path(project_root).resolve()
    root = _safe_v2_split_project_root(
        requested_root, dataset_name, qc.dataset_fingerprint
    )
    split_root = root / "data" / "fixed_splits_v2" / dataset_name
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_path = split_root / "split_manifest.json"
    redirected_from_manifest: Path | None = None

    # A cleaned file may legitimately be replaced while retaining its friendly
    # filename. Never overwrite or mix the protected manifest for the old
    # question set. Route the new fingerprint to its own stable namespace.
    if manifest_path.exists():
        base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if base_manifest.get("dataset_fingerprint") != qc.dataset_fingerprint:
            redirected_from_manifest = manifest_path
            versioned_name = f"{dataset_name}__{qc.dataset_fingerprint[:12]}"
            split_root = root / "data" / "fixed_splits_v2" / versioned_name
            split_root.mkdir(parents=True, exist_ok=True)
            manifest_path = split_root / "split_manifest.json"
            print(
                "Dataset question set changed under the same filename. "
                f"Preserving the old manifest and using: {manifest_path}"
            )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_fingerprint") != qc.dataset_fingerprint:
            raise RuntimeError(
                f"Question identity changed; protected V2 split is at {manifest_path}. "
                "Use a new dataset_name only for an intentional new dataset."
            )
        if manifest.get("seed") != seed or not math.isclose(
            float(manifest.get("development_fraction", -1)), development_fraction
        ):
            raise RuntimeError("The protected V2 split uses a different seed or fraction")
        prior_label_fingerprint = manifest.get(
            "label_fingerprint", manifest.get("label_fingerprint_at_creation", "")
        )
        if prior_label_fingerprint != qc.label_fingerprint:
            history = list(manifest.get("label_fingerprint_history", []))
            if prior_label_fingerprint and prior_label_fingerprint not in history:
                history.append(prior_label_fingerprint)
            manifest["label_fingerprint_history"] = history
            manifest["label_fingerprint"] = qc.label_fingerprint
            manifest["label_fingerprint_updated_utc"] = pd.Timestamp.utcnow().isoformat()
            # Only label metadata changes; protected case membership is untouched.
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        records = manifest.get("cases", [])
        development_ids = [r["fixed_case_id"] for r in records if r["split"] == "development"]
        test_ids = [r["fixed_case_id"] for r in records if r["split"] == "test"]
    else:
        migrated = _migrate_legacy_membership(
            data, root / "data" / "fixed_splits" / dataset_name
        )
        if migrated is not None:
            development_ids, test_ids = migrated
            membership_source = "migrated_from_v1_question_membership"
        else:
            ranked = data.assign(
                _rank=data["fixed_case_id"].map(
                    lambda case_id: _sha256_text(f"{seed}:{case_id}")
                )
            ).sort_values("_rank")
            n_development = max(1, round(len(data) * development_fraction))
            n_development = min(n_development, len(data) - 1)
            development_ids = ranked["fixed_case_id"].iloc[:n_development].tolist()
            test_ids = ranked["fixed_case_id"].iloc[n_development:].tolist()
            membership_source = "deterministic_hash_rank"

        split_by_id = {case_id: "development" for case_id in development_ids}
        split_by_id.update({case_id: "test" for case_id in test_ids})
        cases = [
            {"fixed_case_id": case_id, "split": split_by_id[case_id]}
            for case_id in sorted(split_by_id)
        ]
        manifest = {
            "architecture_version": ARCHITECTURE_VERSION,
            "dataset_name": split_root.name,
            "source_dataset_name": dataset_name,
            "dataset_fingerprint": qc.dataset_fingerprint,
            "label_fingerprint_at_creation": qc.label_fingerprint,
            "label_fingerprint": qc.label_fingerprint,
            "case_ids_exclude_gold": True,
            "seed": seed,
            "development_fraction": development_fraction,
            "n_total": len(data),
            "n_development": len(development_ids),
            "n_test": len(test_ids),
            "membership_source": membership_source,
            "redirected_from_manifest": (
                str(redirected_from_manifest) if redirected_from_manifest else ""
            ),
            "cases": cases,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    development_set = set(development_ids)
    test_set = set(test_ids)
    all_ids = set(data["fixed_case_id"])
    if development_set & test_set:
        raise RuntimeError("Development/test overlap detected")
    if development_set | test_set != all_ids:
        raise RuntimeError("Split manifest is missing or contains unknown case IDs")
    if len(development_ids) != int(manifest["n_development"]):
        raise RuntimeError("Development count does not match manifest")
    if len(test_ids) != int(manifest["n_test"]):
        raise RuntimeError("Test count does not match manifest")

    split_map = {case_id: "development" for case_id in development_ids}
    split_map.update({case_id: "test" for case_id in test_ids})
    manifest_csv = pd.DataFrame(
        {
            "fixed_case_id": sorted(split_map),
            "split": [split_map[case_id] for case_id in sorted(split_map)],
            "dataset_fingerprint": qc.dataset_fingerprint,
            "label_fingerprint": qc.label_fingerprint,
            "split_seed": seed,
            "development_fraction": development_fraction,
        }
    )
    manifest_csv.to_csv(split_root / "split_manifest.csv", index=False)

    development = data[data["fixed_case_id"].isin(development_set)].copy()
    test = data[data["fixed_case_id"].isin(test_set)].copy()
    development["split"] = "development"
    test["split"] = "test"
    development = development.sort_values("fixed_case_id").reset_index(drop=True)
    test = test.sort_values("fixed_case_id").reset_index(drop=True)
    development.to_csv(split_root / "development_20.csv", index=False)
    test.to_csv(split_root / "test_80.csv", index=False)
    return {"development": development, "test": test}, manifest, manifest_path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from either the root or notebooks directory."""
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve().parents[1]])
    for candidate in candidates:
        for path in [candidate, *candidate.parents]:
            if (path / "evaluation" / "mcq_hybrid_v2.py").exists() and (
                path / "data"
            ).exists():
                return path
    raise FileNotFoundError("Could not locate the clinical MCQ project root")


def load_project_api_key(project_root: str | Path | None = None) -> str:
    """Load OPENAI_API_KEY from the environment or project-root .env."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    root = find_project_root(project_root)
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required to load the project .env") from exc
    canonical_name = "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"
    env_candidates = [
        root / ".env",
        root.parent / ".env",
        root.parent.parent / canonical_name / ".env",
    ]
    env_path = next((path.resolve() for path in env_candidates if path.exists()), root / ".env")
    load_dotenv(env_path, override=False)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the environment or any checked .env: "
            + ", ".join(str(path) for path in env_candidates)
        )
    return key


def _validate_schema_value(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(schema.get("properties", {}))
            if extras:
                errors.append(f"{path} has unexpected keys: {sorted(extras)}")
        for key, child_schema in schema.get("properties", {}).items():
            if key in value:
                errors.extend(_validate_schema_value(value[key], child_schema, f"{path}.{key}"))
    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        if len(value) < int(schema.get("minItems", 0)):
            errors.append(f"{path} has too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path} has too many items")
        for index, item in enumerate(value):
            errors.extend(_validate_schema_value(item, schema.get("items", {}), f"{path}[{index}]"))
    elif expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path} must be a string")
        elif "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path} must be one of {schema['enum']}")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path} must be an integer")
        else:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path} is below minimum")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path} is above maximum")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path} must be numeric")
        else:
            numeric = float(value)
            if "minimum" in schema and numeric < schema["minimum"]:
                errors.append(f"{path} is below minimum")
            if "maximum" in schema and numeric > schema["maximum"]:
                errors.append(f"{path} is above maximum")
    elif expected_type == "boolean" and not isinstance(value, bool):
        errors.append(f"{path} must be boolean")
    return errors


@dataclass
class CallResult:
    role: str
    model: str
    parsed: dict[str, Any] | None
    raw_text: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    attempts: int = 1
    retries: int = 0
    status: str = "completed"
    incomplete_reason: str = ""
    error_type: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None and not self.error


class JSONCaller(Protocol):
    async def call_json(
        self,
        *,
        role: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        reasoning_effort: str | None = None,
        allow_retry: bool = True,
    ) -> CallResult: ...


class OpenAIJSONClient:
    """Central Responses API wrapper with strict schemas and recorded retries."""

    def __init__(
        self,
        api_key: str | None = None,
        project_root: str | Path | None = None,
        small_model_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        _configure_platform_trust_store()
        from openai import AsyncOpenAI

        key = api_key or load_project_api_key(project_root)
        # SDK-level retries are disabled. GPT-5 therefore cannot be retried
        # invisibly; small-model retries are explicit below and recorded.
        self.client = AsyncOpenAI(api_key=key, max_retries=0)
        self.small_model_retries = max(0, int(small_model_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    async def call_json(
        self,
        *,
        role: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        reasoning_effort: str | None = None,
        allow_retry: bool = True,
    ) -> CallResult:
        max_attempts = 1 + (self.small_model_retries if allow_retry else 0)
        last: CallResult | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "instructions": system_prompt,
                    "input": user_prompt,
                    "max_output_tokens": int(max_output_tokens),
                    "store": False,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                }
                if reasoning_effort:
                    kwargs["reasoning"] = {"effort": reasoning_effort}
                response = await self.client.responses.create(**kwargs)
                latency = time.perf_counter() - started
                raw = _clean_cell(getattr(response, "output_text", ""))
                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                details = getattr(usage, "output_tokens_details", None)
                reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
                status = _clean_cell(getattr(response, "status", "completed"))
                incomplete = getattr(response, "incomplete_details", None)
                incomplete_reason = _clean_cell(getattr(incomplete, "reason", ""))

                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("structured response was not an object")
                except Exception as exc:
                    last = CallResult(
                        role=role,
                        model=model,
                        parsed=None,
                        raw_text=raw,
                        latency_seconds=latency,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        attempts=attempt,
                        retries=attempt - 1,
                        status=status,
                        incomplete_reason=incomplete_reason,
                        error_type="invalid_output",
                        error=str(exc),
                    )
                else:
                    schema_errors = _validate_schema_value(parsed, schema)
                    if status != "completed" or incomplete_reason:
                        last = CallResult(
                            role=role,
                            model=model,
                            parsed=None,
                            raw_text=raw,
                            latency_seconds=latency,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                            attempts=attempt,
                            retries=attempt - 1,
                            status=status,
                            incomplete_reason=incomplete_reason,
                            error_type="incomplete",
                            error=f"Response incomplete: {incomplete_reason or status}",
                        )
                    elif schema_errors:
                        last = CallResult(
                            role=role,
                            model=model,
                            parsed=None,
                            raw_text=raw,
                            latency_seconds=latency,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                            attempts=attempt,
                            retries=attempt - 1,
                            status=status,
                            error_type="invalid_output",
                            error="; ".join(schema_errors),
                        )
                    else:
                        return CallResult(
                            role=role,
                            model=model,
                            parsed=parsed,
                            raw_text=raw,
                            latency_seconds=latency,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                            attempts=attempt,
                            retries=attempt - 1,
                            status=status,
                        )
            except Exception as exc:
                last = CallResult(
                    role=role,
                    model=model,
                    parsed=None,
                    raw_text="",
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempt,
                    retries=attempt - 1,
                    status="error",
                    error_type="api_error",
                    error=_exception_chain_message(exc),
                )

            if attempt < max_attempts and self.retry_backoff_seconds:
                await asyncio.sleep(self.retry_backoff_seconds * attempt)

        if last is None:
            raise RuntimeError("OpenAI wrapper exited without a call result")
        return last


def _object_schema(properties: dict[str, Any], required: Iterable[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required if required is not None else properties),
        "additionalProperties": False,
    }


def router_schema(max_specialists: int) -> dict[str, Any]:
    specialist = _object_schema(
        {
            "specialty": {"type": "string"},
            "role": {"type": "string", "enum": list(SPECIALIST_ROLES)},
            "focus": {"type": "string"},
            "why_needed": {"type": "string"},
        }
    )
    return _object_schema(
        {
            "difficulty": {"type": "string", "enum": list(DIFFICULTIES)},
            "difficulty_reason": {"type": "string"},
            "lead_specialty": {"type": "string"},
            "panel_size": {"type": "integer", "minimum": 1, "maximum": max_specialists},
            "specialists": {
                "type": "array",
                "items": specialist,
                "minItems": 1,
                "maxItems": max_specialists,
            },
            "expected_disagreement": {
                "type": "string",
                "enum": list(DISAGREEMENT_LEVELS),
            },
        }
    )


def specialist_schema(valid_letters: Iterable[str]) -> dict[str, Any]:
    letters = list(valid_letters)
    return _object_schema(
        {
            "answer_letter": {"type": "string", "enum": letters},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "decisive_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
            "domain_reasoning": {"type": "string"},
            "strongest_alternative": {"type": "string", "enum": letters},
            "why_alternative_is_weaker": {"type": "string"},
            "remaining_uncertainty": {"type": "string"},
        }
    )


def critic_schema() -> dict[str, Any]:
    issue = _object_schema(
        {
            "specialty": {"type": "string"},
            "issue": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "moderate", "high"]},
        }
    )
    return _object_schema(
        {
            "conflict_summary": {"type": "string"},
            "decisive_clues": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
            "issues": {"type": "array", "items": issue, "maxItems": 6},
            "resolution_guidance": {"type": "string"},
        }
    )


def judge_schema(valid_letters: Iterable[str]) -> dict[str, Any]:
    return _object_schema(
        {
            "answer_letter": {"type": "string", "enum": list(valid_letters)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "decisive_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
            "explanation": {"type": "string"},
            "disagreement_resolution": {"type": "string"},
        }
    )


@dataclass
class MultiAgentConfig:
    run_name: str = "multi_dynamic_one_gpt5_v2"
    architecture_version: str = ARCHITECTURE_VERSION
    router_model: str = "gpt-4o-mini"
    specialist_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-5"
    judge_reasoning_effort: str = "medium"
    max_tokens_small: int = 700
    max_tokens_judge: int = 8000
    max_specialists: int = 3
    allow_fourth_specialist: bool = False
    use_router: bool = True
    specialist_mode: Literal["one", "dynamic"] = "dynamic"
    use_critic: bool = True
    small_model_retries: int = 2
    pricing_per_million: dict[str, dict[str, float]] = field(
        default_factory=lambda: json.loads(json.dumps(DEFAULT_PRICING_PER_MILLION))
    )
    pricing_source: str = DEFAULT_PRICING_SOURCE
    prompt_versions: dict[str, str] = field(default_factory=lambda: dict(PROMPT_VERSIONS))

    def validate(self) -> None:
        if not self.judge_model.startswith("gpt-5"):
            raise ValueError("The V2 final judge must be GPT-5")
        non_judge_models = [self.router_model, self.specialist_model, self.critic_model]
        if any(model.startswith("gpt-5") for model in non_judge_models):
            raise ValueError("GPT-5 is restricted to the final Judge role")
        allowed_max = MAX_SPECIALISTS_HARD if self.allow_fourth_specialist else 3
        if not 1 <= self.max_specialists <= allowed_max:
            raise ValueError(f"max_specialists must be between 1 and {allowed_max}")
        if self.specialist_mode not in {"one", "dynamic"}:
            raise ValueError("specialist_mode must be 'one' or 'dynamic'")
        if self.max_tokens_judge < 1000:
            raise ValueError("max_tokens_judge is too small for GPT-5 reasoning")

    def hash_payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return _sha256_text(_stable_json(self.hash_payload()))


@dataclass
class SingleAgentConfig:
    run_name: str
    model: str
    architecture_version: str = ARCHITECTURE_VERSION
    reasoning_effort: str | None = None
    max_output_tokens: int = 8000
    small_model_retries: int = 2
    pricing_per_million: dict[str, dict[str, float]] = field(
        default_factory=lambda: json.loads(json.dumps(DEFAULT_PRICING_PER_MILLION))
    )
    pricing_source: str = DEFAULT_PRICING_SOURCE
    prompt_versions: dict[str, str] = field(default_factory=lambda: dict(PROMPT_VERSIONS))

    def validate(self) -> None:
        if self.model.startswith("gpt-5") and self.max_output_tokens < 1000:
            raise ValueError("GPT-5 baseline output budget is too small")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _sha256_text(_stable_json(asdict(self)))


_GENERIC_SPECIALTIES = {
    "clinical reasoner",
    "differential reviewer",
    "skeptical reviewer",
    "medical expert",
    "general medical expert",
}
_SPECIALTY_ALIASES = {
    "general dermatology": "Dermatology",
    "clinical dermatology": "Dermatology",
    "general cardiology": "Cardiology",
    "general neurology": "Neurology",
    "general internal medicine": "Internal Medicine",
}


def _canonical_specialty(value: Any) -> str:
    specialty = re.sub(r"\s+", " ", _clean_cell(value)).strip(" .,:;-")
    alias = _SPECIALTY_ALIASES.get(specialty.casefold())
    return alias or specialty


def sanitize_route(
    route: dict[str, Any] | None,
    max_specialists: int = 3,
    allow_fourth_specialist: bool = False,
    specialist_mode: Literal["one", "dynamic"] = "dynamic",
) -> tuple[dict[str, Any], bool, list[str]]:
    """Sanitize routing and fall back to one Internal Medicine specialist."""
    hard_max = MAX_SPECIALISTS_HARD if allow_fourth_specialist else 3
    max_allowed = min(max(1, int(max_specialists)), hard_max)
    errors: list[str] = []
    if not isinstance(route, dict):
        route = {}
        errors.append("route_not_object")

    difficulty = _clean_cell(route.get("difficulty", "")).lower()
    if difficulty not in DIFFICULTIES:
        errors.append("invalid_difficulty")
        difficulty = "moderate"
    expected = _clean_cell(route.get("expected_disagreement", "")).lower()
    if expected not in DISAGREEMENT_LEVELS:
        errors.append("invalid_expected_disagreement")
        expected = "moderate"

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    raw_specialists = route.get("specialists", [])
    if not isinstance(raw_specialists, list):
        raw_specialists = []
        errors.append("specialists_not_array")
    for raw in raw_specialists:
        if not isinstance(raw, dict):
            errors.append("specialist_not_object")
            continue
        specialty = _canonical_specialty(raw.get("specialty", ""))
        key = specialty.casefold()
        if not specialty or key in _GENERIC_SPECIALTIES:
            errors.append("invalid_or_generic_specialty")
            continue
        if key in seen:
            errors.append("duplicate_specialty_removed")
            continue
        seen.add(key)
        cleaned.append(
            {
                "specialty": specialty,
                "role": _clean_cell(raw.get("role", "supporting")).lower(),
                "focus": _clean_cell(raw.get("focus", "")) or "decisive clinical evidence",
                "why_needed": _clean_cell(raw.get("why_needed", "")) or "distinct clinical expertise",
            }
        )

    lead = _canonical_specialty(route.get("lead_specialty", ""))
    lead_index = next(
        (index for index, item in enumerate(cleaned) if item["specialty"].casefold() == lead.casefold()),
        None,
    )
    if lead_index is None and cleaned:
        errors.append("lead_not_in_panel")
        lead_index = 0
        lead = cleaned[0]["specialty"]
    if lead_index is not None:
        for index, item in enumerate(cleaned):
            item["role"] = "lead" if index == lead_index else "supporting"

    try:
        panel_size = int(route.get("panel_size", len(cleaned)))
    except Exception:
        panel_size = len(cleaned)
        errors.append("invalid_panel_size")
    if panel_size != len(cleaned):
        errors.append("panel_size_mismatch")

    cleaned = cleaned[:max_allowed]
    if specialist_mode == "one" and cleaned:
        lead_item = next((item for item in cleaned if item["role"] == "lead"), cleaned[0])
        cleaned = [lead_item]
        lead = lead_item["specialty"]
    if cleaned:
        cleaned[0]["role"] = "lead" if len(cleaned) == 1 else cleaned[0]["role"]

    fallback = not cleaned
    if fallback:
        cleaned = [
            {
                "specialty": "Internal Medicine",
                "role": "lead",
                "focus": "integrate the decisive clinical findings",
                "why_needed": "single conservative fallback after invalid routing",
            }
        ]
        lead = "Internal Medicine"
        difficulty = "moderate"
        expected = "moderate"
        errors.append("single_specialist_fallback")

    for item in cleaned:
        item["role"] = "lead" if item["specialty"] == lead else "supporting"
    sanitized = {
        "difficulty": difficulty,
        "difficulty_reason": _clean_cell(route.get("difficulty_reason", "")) or "Router output required sanitation.",
        "lead_specialty": lead,
        "panel_size": len(cleaned),
        "specialists": cleaned,
        "expected_disagreement": expected,
    }
    return sanitized, fallback, errors


def critic_triggers(
    route: Mapping[str, Any],
    specialist_outputs: list[Mapping[str, Any]],
) -> list[str]:
    """Return evidence-based critic triggers without numerical thresholds."""
    triggers: list[str] = []
    valid_letters = [
        _clean_cell(item.get("answer_letter", "")).upper()
        for item in specialist_outputs
        if item.get("valid_output", True) and _clean_cell(item.get("answer_letter", ""))
    ]
    if len(set(valid_letters)) > 1:
        triggers.append("specialist_answer_disagreement")
    if any(not bool(item.get("valid_output", True)) for item in specialist_outputs):
        triggers.append("invalid_specialist_output")
    if route.get("expected_disagreement") == "high":
        triggers.append("expected_disagreement_high")
    if route.get("difficulty") == "very_complex":
        triggers.append("very_complex_case")

    uncertainty_markers = (
        "meaningful",
        "substantial",
        "cannot distinguish",
        "conflicting",
        "uncertain between",
        "depends on",
    )
    uncertainty = " ".join(
        _clean_cell(item.get("remaining_uncertainty", "")).lower()
        for item in specialist_outputs
    )
    if any(marker in uncertainty for marker in uncertainty_markers):
        triggers.append("meaningful_residual_uncertainty")
    return list(dict.fromkeys(triggers))


def validate_forced_choice(
    parsed: Mapping[str, Any] | None,
    valid_letters: Iterable[str],
) -> dict[str, Any]:
    """Deterministically validate the final forced-choice output."""
    valid_set = set(valid_letters)
    if not isinstance(parsed, Mapping):
        return {
            "valid": False,
            "answer_letter": "",
            "confidence": float("nan"),
            "invalid_reason": "missing_structured_output",
        }
    letter = _clean_cell(parsed.get("answer_letter", "")).upper()
    try:
        confidence = float(parsed.get("confidence"))
    except Exception:
        confidence = float("nan")
    reasons: list[str] = []
    if letter not in valid_set:
        reasons.append("answer_not_in_supplied_options")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        reasons.append("invalid_confidence")
    return {
        "valid": not reasons,
        "answer_letter": letter if letter in valid_set else "",
        "confidence": confidence,
        "invalid_reason": ";".join(reasons),
    }


def usage_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Mapping[str, Mapping[str, float]],
) -> float:
    rates = pricing.get(model)
    if not rates:
        return float("nan")
    return (
        input_tokens * float(rates["input"])
        + output_tokens * float(rates["output"])
    ) / 1_000_000


def _call_trace(result: CallResult, prompt_version: str) -> dict[str, Any]:
    return {
        "role": result.role,
        "model": result.model,
        "prompt_version": prompt_version,
        "latency_seconds": result.latency_seconds,
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reasoning_tokens": result.reasoning_tokens,
        },
        "attempts": result.attempts,
        "retries": result.retries,
        "status": result.status,
        "incomplete_reason": result.incomplete_reason,
        "structured_response": result.parsed,
        "error_type": result.error_type,
        "error": result.error,
    }


def _case_ids_hash(frame: pd.DataFrame) -> str:
    return _sha256_text("\n".join(sorted(frame["fixed_case_id"].astype(str))))


def _run_paths(output_dir: Path, run_name: str, split: str) -> dict[str, Path]:
    prefix = output_dir / f"{run_name}_{split}"
    return {
        "predictions": Path(f"{prefix}_predictions.csv"),
        "traces": Path(f"{prefix}_traces.jsonl"),
        "metadata": Path(f"{prefix}_run_metadata.json"),
    }


def _prepare_run(
    frame: pd.DataFrame,
    output_dir: str | Path,
    run_name: str,
    split: str,
    config_hash: str,
    config_payload: dict[str, Any],
    resume: bool,
) -> tuple[dict[str, Path], pd.DataFrame, set[str], dict[str, Any]]:
    output = Path(output_dir).resolve()
    prospective = _run_paths(output, run_name, split)["predictions"]
    if os.name == "nt" and len(str(prospective)) >= 248:
        root = find_project_root()
        canonical = root.parent.parent / "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"
        candidate = canonical / "results" / output.name
        candidate_path = _run_paths(candidate, run_name, split)["predictions"]
        if (canonical / "evaluation" / "mcq_hybrid_v2.py").exists() and len(str(candidate_path)) < 248:
            output = candidate
            print(f"Nested output path is too long for Windows. Using: {output}")
        else:
            raise RuntimeError(f"V2 prediction path is too long for Windows: {prospective}")
    output.mkdir(parents=True, exist_ok=True)
    paths = _run_paths(output, run_name, split)
    probe = output / ".v2_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"V2 output directory is not writable: {output}") from exc
    data_fp_values = frame.get("dataset_fingerprint", pd.Series(dtype=str)).dropna().astype(str).unique()
    label_fp_values = frame.get("label_fingerprint", pd.Series(dtype=str)).dropna().astype(str).unique()
    dataset_fp = data_fp_values[0] if len(data_fp_values) == 1 else dataset_fingerprint(frame)
    current_label_fp = label_fp_values[0] if len(label_fp_values) == 1 else label_fingerprint(frame)
    metadata = {
        "run_name": run_name,
        "architecture_version": ARCHITECTURE_VERSION,
        "split": split,
        "dataset_fingerprint": dataset_fp,
        "label_fingerprint": current_label_fp,
        "case_ids_hash": _case_ids_hash(frame),
        "n_requested_cases": len(frame),
        "config_hash": config_hash,
        "config": config_payload,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    existing = pd.DataFrame()
    completed: set[str] = set()
    if resume and paths["predictions"].exists():
        if not paths["metadata"].exists():
            raise RuntimeError("Resume rejected: run metadata is missing. Use RESUME=False or a new RUN_NAME.")
        old = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        required_matches = (
            "run_name",
            "architecture_version",
            "split",
            "dataset_fingerprint",
            "label_fingerprint",
            "case_ids_hash",
            "config_hash",
        )
        mismatches = [key for key in required_matches if old.get(key) != metadata.get(key)]
        if mismatches:
            raise RuntimeError(
                "Resume rejected because configuration/data differ in "
                f"{mismatches}. Use RESUME=False or a new RUN_NAME."
            )
        existing = pd.read_csv(paths["predictions"], keep_default_na=False)
        unknown = set(existing.get("fixed_case_id", pd.Series(dtype=str)).astype(str)) - set(
            frame["fixed_case_id"].astype(str)
        )
        if unknown:
            raise RuntimeError("Resume rejected: predictions contain IDs outside the current dataframe")
        completed = set(existing["fixed_case_id"].astype(str))
    else:
        paths["traces"].write_text("", encoding="utf-8")
        paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return paths, existing, completed, metadata


def _append_trace(path: Path, trace: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def _save_predictions(path: Path, existing: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    new = pd.DataFrame(records)
    if existing.empty:
        combined = new
    elif new.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    if not combined.empty:
        combined = (
            combined.drop_duplicates("fixed_case_id", keep="last")
            .sort_values("fixed_case_id")
            .reset_index(drop=True)
        )
    combined.to_csv(path, index=False)
    return combined


def _aggregate_calls(results: Iterable[CallResult]) -> dict[str, int | float]:
    calls = list(results)
    return {
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "reasoning_tokens": sum(call.reasoning_tokens for call in calls),
        "total_calls": sum(call.attempts for call in calls),
        "total_retries": sum(call.retries for call in calls),
        "llm_latency_seconds": sum(call.latency_seconds for call in calls),
    }


def _validate_execution_frame(frame: pd.DataFrame) -> None:
    """Prevent excluded, unresolved, duplicate, or stale cases reaching models."""
    if frame.empty:
        raise ValueError("No cases were supplied for execution")
    if "eligible_for_evaluation" in frame and not _bool_column(
        frame, "eligible_for_evaluation"
    ).all():
        raise ValueError("Excluded QC rows cannot enter model execution")
    if frame["fixed_case_id"].astype(str).duplicated().any():
        raise ValueError("Execution dataframe contains duplicate fixed_case_id values")
    invalid: list[str] = []
    for _, row in frame.iterrows():
        options = parse_options(build_question(row))
        if _clean_cell(row.get("reference_letter", "")).upper() not in options:
            invalid.append(str(row.get("fixed_case_id", "")))
    if invalid:
        raise ValueError(f"Execution dataframe has invalid gold labels: {invalid[:5]}")


async def run_single_agent_v2(
    frame: pd.DataFrame,
    config: SingleAgentConfig,
    split: str,
    output_dir: str | Path,
    *,
    resume: bool = False,
    caller: JSONCaller | None = None,
) -> pd.DataFrame:
    """Run a forced-choice single-agent baseline on the supplied cases."""
    config.validate()
    _validate_execution_frame(frame)
    if caller is None:
        caller = OpenAIJSONClient(small_model_retries=config.small_model_retries)
    paths, existing, completed, metadata = _prepare_run(
        frame,
        output_dir,
        config.run_name,
        split,
        config.config_hash,
        asdict(config),
        resume,
    )
    records: list[dict[str, Any]] = []
    for case_number, (_, row) in enumerate(frame.iterrows(), start=1):
        case_id = str(row["fixed_case_id"])
        if case_id in completed:
            continue
        started = time.perf_counter()
        question = build_question(row)
        options = parse_options(question)
        valid_letters = list(options)
        is_gpt5 = config.model.startswith("gpt-5")
        call = await caller.call_json(
            role="single",
            model=config.model,
            system_prompt=SINGLE_PROMPT,
            user_prompt=question,
            schema_name="single_mcq_v2",
            schema=judge_schema(valid_letters),
            max_output_tokens=config.max_output_tokens,
            reasoning_effort=config.reasoning_effort if is_gpt5 else None,
            allow_retry=not is_gpt5,
        )
        gpt5_call_count = call.attempts if is_gpt5 else 0
        if is_gpt5 and gpt5_call_count != 1:
            raise RuntimeError(
                f"GPT-5 invariant violated for {case_id}: observed {gpt5_call_count} calls"
            )

        validator = validate_forced_choice(call.parsed, valid_letters)
        invalid_answer = not validator["valid"]
        api_error = call.error_type == "api_error"
        technical_failure = bool(call.error or invalid_answer)
        predicted = validator["answer_letter"]
        confidence = validator["confidence"]
        correct = bool(predicted == row["reference_letter"] and not technical_failure)
        estimated_cost = usage_cost(
            config.model,
            call.input_tokens,
            call.output_tokens,
            config.pricing_per_million,
        )
        trace = {
            "fixed_case_id": case_id,
            "single": _call_trace(call, config.prompt_versions["single"]),
            "validator": validator,
        }
        _append_trace(paths["traces"], trace)
        records.append(
            {
                "fixed_case_id": case_id,
                "split": split,
                "system": config.run_name,
                "model": config.model,
                "architecture_version": ARCHITECTURE_VERSION,
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "label_fingerprint": metadata["label_fingerprint"],
                "config_hash": config.config_hash,
                "prompt_version": config.prompt_versions["single"],
                "row_id": row.get("row_id", ""),
                "source_case_id": row.get("source_case_id", row.get("case_id", "")),
                "gold_status": row.get("gold_status", "SOURCE_DERIVED_UNVERIFIED"),
                "reference_letter": row["reference_letter"],
                "predicted_letter": predicted,
                "confidence": confidence,
                "correct": correct,
                "invalid_answer": invalid_answer,
                "abstain": False,
                "technical_failure": technical_failure,
                "api_error": api_error,
                "token_limit_failure": call.error_type == "incomplete",
                "error_type": call.error_type,
                "error": call.error or validator["invalid_reason"],
                "latency_seconds": time.perf_counter() - started,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "reasoning_tokens": call.reasoning_tokens,
                "total_tokens": call.input_tokens + call.output_tokens,
                "total_calls": call.attempts,
                "total_retries": call.retries,
                "gpt5_call_count": gpt5_call_count,
                "estimated_cost_usd": estimated_cost,
                "raw_response": call.raw_text,
            }
        )
        _save_predictions(paths["predictions"], existing, records)
        if case_number % 10 == 0:
            print(f"{config.run_name}: completed {case_number}/{len(frame)}")
    return _save_predictions(paths["predictions"], existing, records)


def _route_user_prompt(question: str, max_specialists: int, allow_fourth: bool) -> str:
    maximum = min(max_specialists, MAX_SPECIALISTS_HARD if allow_fourth else 3)
    return (
        f"CONFIGURED MAXIMUM SPECIALISTS: {maximum}\n"
        f"FOURTH SPECIALIST ALLOWED: {str(allow_fourth).lower()}\n\n"
        f"ORIGINAL MCQ:\n{question}"
    )


def _specialist_user_prompt(question: str, assignment: Mapping[str, str]) -> str:
    # Deliberately excludes all other specialist, critic, judge, and gold data.
    return (
        f"ORIGINAL MCQ:\n{question}\n\n"
        f"ASSIGNED SPECIALTY: {assignment['specialty']}\n"
        f"ASSIGNED FOCUS: {assignment['focus']}"
    )


def _majority_letter(opinions: list[Mapping[str, Any]]) -> str:
    letters = [
        _clean_cell(item.get("answer_letter", "")).upper()
        for item in opinions
        if item.get("valid_output") and _clean_cell(item.get("answer_letter", ""))
    ]
    if not letters:
        return ""
    counts = Counter(letters)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return ""
    return top[0][0]


async def run_multi_agent_v2(
    frame: pd.DataFrame,
    config: MultiAgentConfig,
    split: str,
    output_dir: str | Path,
    *,
    resume: bool = False,
    caller: JSONCaller | None = None,
) -> pd.DataFrame:
    """Run the adaptive V2 pipeline with exactly one GPT-5 call per case."""
    config.validate()
    _validate_execution_frame(frame)
    if caller is None:
        caller = OpenAIJSONClient(small_model_retries=config.small_model_retries)
    paths, existing, completed, metadata = _prepare_run(
        frame,
        output_dir,
        config.run_name,
        split,
        config.config_hash,
        config.hash_payload(),
        resume,
    )
    records: list[dict[str, Any]] = []

    for case_number, (_, row) in enumerate(frame.iterrows(), start=1):
        case_id = str(row["fixed_case_id"])
        if case_id in completed:
            continue
        started = time.perf_counter()
        question = build_question(row)
        options = parse_options(question)
        valid_letters = list(options)
        calls: list[CallResult] = []
        trace: dict[str, Any] = {
            "fixed_case_id": case_id,
            "router": None,
            "specialists": [],
            "critic": None,
            "judge": None,
            "validator": None,
        }

        if config.use_router:
            router_call = await caller.call_json(
                role="router",
                model=config.router_model,
                system_prompt=ROUTER_PROMPT,
                user_prompt=_route_user_prompt(
                    question, config.max_specialists, config.allow_fourth_specialist
                ),
                schema_name="clinical_router_v2",
                schema=router_schema(config.max_specialists),
                max_output_tokens=config.max_tokens_small,
                allow_retry=True,
            )
            calls.append(router_call)
            route, routing_fallback, route_errors = sanitize_route(
                router_call.parsed,
                config.max_specialists,
                config.allow_fourth_specialist,
                config.specialist_mode,
            )
            trace["router"] = {
                **_call_trace(router_call, config.prompt_versions["router"]),
                "sanitized_route": route,
                "routing_fallback": routing_fallback,
                "validation_issues": route_errors,
            }
        else:
            route, routing_fallback, route_errors = sanitize_route(
                None,
                config.max_specialists,
                config.allow_fourth_specialist,
                config.specialist_mode,
            )
            routing_fallback = True
            trace["router"] = {
                "disabled": True,
                "sanitized_route": route,
                "routing_fallback": True,
                "validation_issues": ["router_disabled_single_fallback"],
            }

        async def run_specialist(assignment: dict[str, str]) -> tuple[dict[str, Any], CallResult]:
            prompt = SPECIALIST_PROMPT.format(
                specialty=assignment["specialty"],
                focus=assignment["focus"],
                specialty_instruction=specialty_instruction(assignment["specialty"]),
            )
            result = await caller.call_json(
                role="specialist",
                model=config.specialist_model,
                system_prompt=prompt,
                user_prompt=_specialist_user_prompt(question, assignment),
                schema_name="clinical_specialist_v2",
                schema=specialist_schema(valid_letters),
                max_output_tokens=config.max_tokens_small,
                allow_retry=True,
            )
            parsed = dict(result.parsed or {})
            validator = validate_forced_choice(parsed, valid_letters)
            opinion = {
                "specialty": assignment["specialty"],
                "role": assignment["role"],
                "focus": assignment["focus"],
                **parsed,
                "answer_letter": validator["answer_letter"],
                "valid_output": bool(result.ok and validator["valid"]),
                "output_error": result.error or validator["invalid_reason"],
            }
            return opinion, result

        # Specialists see only the original case and their own assignment and run concurrently.
        specialist_runs = await asyncio.gather(
            *(run_specialist(assignment) for assignment in route["specialists"])
        )
        opinions: list[dict[str, Any]] = []
        for assignment, (opinion, specialist_call) in zip(
            route["specialists"], specialist_runs
        ):
            calls.append(specialist_call)
            opinions.append(opinion)
            trace["specialists"].append(
                {
                    "assignment": assignment,
                    **_call_trace(
                        specialist_call, config.prompt_versions["specialist"]
                    ),
                    "validated_opinion": opinion,
                }
            )

        triggers = critic_triggers(route, opinions)
        critic_used = bool(config.use_critic and triggers)
        critique: dict[str, Any] | None = None
        if critic_used:
            critic_input = (
                f"ORIGINAL MCQ:\n{question}\n\n"
                f"SPECIALIST OPINIONS:\n{json.dumps(opinions, ensure_ascii=False)}\n\n"
                f"TRIGGERS:\n{json.dumps(triggers)}"
            )
            critic_call = await caller.call_json(
                role="critic",
                model=config.critic_model,
                system_prompt=CRITIC_PROMPT,
                user_prompt=critic_input,
                schema_name="clinical_conflict_reviewer_v2",
                schema=critic_schema(),
                max_output_tokens=config.max_tokens_small,
                allow_retry=True,
            )
            calls.append(critic_call)
            critique = critic_call.parsed
            trace["critic"] = {
                **_call_trace(critic_call, config.prompt_versions["critic"]),
                "trigger": triggers,
            }

        judge_input = (
            f"ORIGINAL MCQ:\n{question}\n\n"
            f"ROUTING SUMMARY:\n{json.dumps({key: route[key] for key in ('difficulty', 'difficulty_reason', 'lead_specialty', 'specialists', 'expected_disagreement')}, ensure_ascii=False)}\n\n"
            f"INDEPENDENT SPECIALIST OPINIONS:\n{json.dumps(opinions, ensure_ascii=False)}"
        )
        if critique is not None:
            judge_input += f"\n\nCONFLICT REVIEW:\n{json.dumps(critique, ensure_ascii=False)}"

        # Exactly one attempted GPT-5 call. Retries are disabled both here and in the SDK.
        judge_call = await caller.call_json(
            role="judge",
            model=config.judge_model,
            system_prompt=JUDGE_PROMPT,
            user_prompt=judge_input,
            schema_name="final_clinical_judge_v2",
            schema=judge_schema(valid_letters),
            max_output_tokens=config.max_tokens_judge,
            reasoning_effort=config.judge_reasoning_effort,
            allow_retry=False,
        )
        calls.append(judge_call)
        gpt5_call_count = judge_call.attempts
        if gpt5_call_count != 1:
            raise RuntimeError(
                f"GPT-5 invariant violated for {case_id}: observed {gpt5_call_count} calls"
            )
        if any(call.model.startswith("gpt-5") for call in calls if call.role != "judge"):
            raise RuntimeError(f"GPT-5 role invariant violated for {case_id}")

        validator = validate_forced_choice(judge_call.parsed, valid_letters)
        trace["judge"] = _call_trace(judge_call, config.prompt_versions["judge"])
        trace["validator"] = validator
        invalid_answer = not validator["valid"]
        predicted = validator["answer_letter"]
        confidence = validator["confidence"]
        api_error = judge_call.error_type == "api_error"
        technical_failure = bool(judge_call.error or invalid_answer)
        correct = bool(predicted == row["reference_letter"] and not technical_failure)

        usage = _aggregate_calls(calls)
        role_costs: dict[str, float] = {}
        for role in ("router", "specialist", "critic", "judge"):
            role_calls = [call for call in calls if call.role == role]
            costs = [
                usage_cost(
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    config.pricing_per_million,
                )
                for call in role_calls
            ]
            role_costs[role] = (
                float("nan") if any(math.isnan(cost) for cost in costs) else sum(costs)
            )
        total_cost = (
            float("nan")
            if any(math.isnan(cost) for cost in role_costs.values())
            else sum(role_costs.values())
        )

        majority = _majority_letter(opinions)
        lead_opinion = next(
            (item for item in opinions if item["specialty"] == route["lead_specialty"]),
            {},
        )
        lead_letter = _clean_cell(lead_opinion.get("answer_letter", ""))
        answered_letters = [
            item["answer_letter"] for item in opinions if item.get("valid_output")
        ]
        actual_disagreement = len(set(answered_letters)) > 1
        any_api_error = any(call.error_type == "api_error" for call in calls)
        all_errors = [call.error for call in calls if call.error]
        selected_specialties = [item["specialty"] for item in route["specialists"]]
        _append_trace(paths["traces"], trace)

        records.append(
            {
                "fixed_case_id": case_id,
                "split": split,
                "system": config.run_name,
                "model": f"adaptive small agents + exactly one {config.judge_model} judge",
                "architecture_version": ARCHITECTURE_VERSION,
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "label_fingerprint": metadata["label_fingerprint"],
                "config_hash": config.config_hash,
                "router_prompt_version": config.prompt_versions["router"],
                "specialist_prompt_version": config.prompt_versions["specialist"],
                "critic_prompt_version": config.prompt_versions["critic"],
                "judge_prompt_version": config.prompt_versions["judge"],
                "row_id": row.get("row_id", ""),
                "source_case_id": row.get("source_case_id", row.get("case_id", "")),
                "gold_status": row.get("gold_status", "SOURCE_DERIVED_UNVERIFIED"),
                "reference_letter": row["reference_letter"],
                "predicted_letter": predicted,
                "confidence": confidence,
                "correct": correct,
                "invalid_answer": invalid_answer,
                "abstain": False,
                "technical_failure": technical_failure,
                "api_error": api_error or any_api_error,
                "token_limit_failure": judge_call.error_type == "incomplete",
                "error_type": judge_call.error_type,
                "error": " | ".join(all_errors) or validator["invalid_reason"],
                "difficulty": route["difficulty"],
                "difficulty_reason": route["difficulty_reason"],
                "panel_size": route["panel_size"],
                "n_specialists": len(route["specialists"]),
                "lead_specialty": route["lead_specialty"],
                "selected_specialties": json.dumps(selected_specialties),
                "specialist_assignments": json.dumps(route["specialists"], ensure_ascii=False),
                "specialist_answer_letters": json.dumps(answered_letters),
                "specialist_confidences": json.dumps(
                    [item.get("confidence") for item in opinions]
                ),
                "specialist_opinions": json.dumps(opinions, ensure_ascii=False),
                "expected_disagreement": route["expected_disagreement"],
                "actual_disagreement": actual_disagreement,
                "routing_fallback": routing_fallback,
                "routing_validation_issues": json.dumps(route_errors),
                "critic_used": critic_used,
                "critic_trigger": json.dumps(triggers),
                "gpt5_final_answer": predicted,
                "gpt5_confidence": confidence,
                "gpt5_overruled_majority": bool(majority and predicted and predicted != majority),
                "gpt5_agreed_with_lead": bool(lead_letter and predicted == lead_letter),
                "specialist_majority_letter": majority,
                "latency_seconds": time.perf_counter() - started,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                "total_calls": usage["total_calls"],
                "total_retries": usage["total_retries"],
                "gpt5_call_count": gpt5_call_count,
                "router_cost_usd": role_costs["router"],
                "specialist_cost_usd": role_costs["specialist"],
                "critic_cost_usd": role_costs["critic"],
                "judge_cost_usd": role_costs["judge"],
                "estimated_cost_usd": total_cost,
                "raw_response": judge_call.raw_text,
            }
        )
        _save_predictions(paths["predictions"], existing, records)
        if case_number % 5 == 0:
            print(f"{config.run_name}: completed {case_number}/{len(frame)}")

    result = _save_predictions(paths["predictions"], existing, records)
    successful = result[
        ~_bool_column(result, "api_error")
        & ~_bool_column(result, "invalid_answer")
    ]
    if not successful.empty and not (successful["gpt5_call_count"] == 1).all():
        raise RuntimeError("GPT-5 exactly-once invariant failed after execution")
    return result


def validate_v2_multi_results(
    result: pd.DataFrame,
    current_frame: pd.DataFrame,
    config: MultiAgentConfig,
) -> dict[str, Any]:
    """Validate the post-run invariants required by notebook 03."""
    successful = result[
        ~_bool_column(result, "api_error") & ~_bool_column(result, "invalid_answer")
    ]
    specialist_counts = pd.to_numeric(result["n_specialists"], errors="coerce")
    successful_gpt5_counts = pd.to_numeric(
        successful["gpt5_call_count"], errors="coerce"
    )
    checks = {
        "successful_gpt5_exactly_once": bool(
            successful.empty or (successful_gpt5_counts == 1).all()
        ),
        "at_least_one_specialist": bool((specialist_counts >= 1).all()),
        "within_specialist_maximum": bool(
            (specialist_counts <= config.max_specialists).all()
        ),
        "no_post_gpt5_llm_auditor": "audit_rejected" not in result.columns,
        "case_ids_in_current_dataframe": set(result["fixed_case_id"]).issubset(
            set(current_frame["fixed_case_id"])
        ),
        "config_hash_matches": bool(
            result.empty or (result["config_hash"] == config.config_hash).all()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"V2 validation failed: {failed}")
    return checks
