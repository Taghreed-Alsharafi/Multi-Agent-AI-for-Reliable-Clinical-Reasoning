"""Metrics and paired analyses for the V2 clinical MCQ experiment.

This module performs no model or API calls and is safe to use in notebook 04.
"""

from __future__ import annotations

from typing import Any
import json
import math

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "system",
    "split",
    "n",
    "accuracy",
    "macro_f1",
    "answered_accuracy",
    "coverage",
    "abstention_rate",
    "invalid_answer_rate",
    "high_confidence_error_rate",
    "high_confidence_error_numerator",
    "high_confidence_error_denominator",
    "brier_score",
    "expected_calibration_error",
    "api_error_rate",
    "mean_latency_seconds",
    "median_latency_seconds",
    "reference_extraction_rate",
    "mean_input_tokens",
    "mean_output_tokens",
    "mean_reasoning_tokens",
    "mean_total_tokens",
    "mean_estimated_cost_usd",
    "mean_total_calls",
    "mean_gpt5_call_count",
    "mean_n_specialists",
    "critic_used_rate",
    "specialist_disagreement_rate",
    "gpt5_majority_overrule_rate",
    # Retained as a zero-valued compatibility row for legacy presentation.
    # V2 has no auditor and prediction files contain no audit decision column.
    "audit_rejected_rate",
]


def _as_bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return _as_bool_series(frame[column], default)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _safe_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else float("nan")


def _valid_predictions(frame: pd.DataFrame) -> pd.Series:
    predicted = frame.get("predicted_letter", pd.Series("", index=frame.index)).astype(str)
    invalid = _bool_series(frame, "invalid_answer")
    api_error = _bool_series(frame, "api_error")
    return predicted.str.fullmatch(r"[A-H]") & ~invalid & ~api_error


def expected_calibration_error(
    confidence: pd.Series,
    correct: pd.Series,
    bins: int = 10,
) -> float:
    if confidence.empty:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        upper = high if high < 1 else 1.000001
        mask = (confidence >= low) & (confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def per_answer_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return precision, recall, and F1 for every gold answer letter."""
    valid = _valid_predictions(frame)
    references = frame["reference_letter"].astype(str)
    predictions = frame["predicted_letter"].astype(str)
    rows: list[dict[str, Any]] = []
    for label in sorted(references[references.str.fullmatch(r"[A-H]")].unique()):
        tp = int(((references == label) & (predictions == label) & valid).sum())
        fp = int(((references != label) & (predictions == label) & valid).sum())
        fn = int(((references == label) & ((predictions != label) | ~valid)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "answer_letter": label,
                "support": int((references == label).sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def performance_metrics(
    frame: pd.DataFrame,
    high_confidence_threshold: float = 0.80,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Compute forced-choice performance, reliability, and resource metrics."""
    if frame.empty:
        return {"n": 0.0}
    result = frame.copy()
    valid = _valid_predictions(result)
    references = result["reference_letter"].astype(str)
    predictions = result["predicted_letter"].astype(str)
    correct = (predictions == references) & valid
    if "correct" in result:
        reported_correct = _bool_series(result, "correct")
        if not (reported_correct == correct).all():
            raise ValueError("Stored correctness does not match letters/technical validity")
    confidence = _numeric_series(result, "confidence")
    high_valid = valid & confidence.ge(high_confidence_threshold)
    high_error_numerator = int((high_valid & ~correct).sum())
    high_error_denominator = int(high_valid.sum())

    per_class = per_answer_metrics(result)
    valid_confidence = confidence[valid].clip(0, 1)
    valid_correct = correct[valid].astype(float)
    brier = (
        float(np.mean((valid_confidence - valid_correct) ** 2))
        if not valid_confidence.empty
        else float("nan")
    )
    ece = expected_calibration_error(valid_confidence, valid_correct, calibration_bins)

    output: dict[str, float] = {
        "n": float(len(result)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(per_class["f1"].mean()) if not per_class.empty else 0.0,
        "answered_accuracy": float(correct[valid].mean()) if valid.any() else 0.0,
        "coverage": float(valid.mean()),
        "abstention_rate": float(_bool_series(result, "abstain").mean()),
        "invalid_answer_rate": float(_bool_series(result, "invalid_answer").mean()),
        "high_confidence_error_rate": (
            high_error_numerator / high_error_denominator
            if high_error_denominator
            else float("nan")
        ),
        "high_confidence_error_numerator": float(high_error_numerator),
        "high_confidence_error_denominator": float(high_error_denominator),
        "brier_score": brier,
        "expected_calibration_error": ece,
        "api_error_rate": float(_bool_series(result, "api_error").mean()),
        "mean_latency_seconds": _safe_mean(_numeric_series(result, "latency_seconds")),
        "median_latency_seconds": _safe_median(_numeric_series(result, "latency_seconds")),
        "reference_extraction_rate": float(references.str.fullmatch(r"[A-H]").mean()),
    }

    for column in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "total_calls",
        "gpt5_call_count",
    ):
        output[f"mean_{column}"] = _safe_mean(_numeric_series(result, column))

    output["mean_n_specialists"] = (
        _safe_mean(_numeric_series(result, "n_specialists"))
        if "n_specialists" in result
        else 0.0
    )
    output["critic_used_rate"] = (
        float(_bool_series(result, "critic_used").mean())
        if "critic_used" in result
        else 0.0
    )
    output["specialist_disagreement_rate"] = (
        float(_bool_series(result, "actual_disagreement").mean())
        if "actual_disagreement" in result
        else 0.0
    )
    output["gpt5_majority_overrule_rate"] = (
        float(_bool_series(result, "gpt5_overruled_majority").mean())
        if "gpt5_overruled_majority" in result
        else 0.0
    )
    output["audit_rejected_rate"] = 0.0
    return output


def summarize_system(
    frame: pd.DataFrame,
    system: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    system_name = system or str(frame["system"].iloc[0])
    split_name = split or str(frame["split"].iloc[0])
    row = {"system": system_name, "split": split_name, **performance_metrics(frame)}
    return pd.DataFrame([row]).reindex(columns=SUMMARY_COLUMNS)


def comparison_summary(systems: dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    rows = [summarize_system(frame, name, split) for name, frame in systems.items()]
    return pd.concat(rows, ignore_index=True).reindex(columns=SUMMARY_COLUMNS)


def distribution_table(
    frame: pd.DataFrame,
    column: str,
    label: str | None = None,
) -> pd.DataFrame:
    if column not in frame:
        return pd.DataFrame(columns=[label or column, "n", "rate"])
    counts = frame[column].fillna("UNAVAILABLE").astype(str).value_counts(dropna=False)
    result = counts.rename_axis(label or column).reset_index(name="n")
    result["rate"] = result["n"] / len(frame) if len(frame) else float("nan")
    return result


def routing_summary_tables(multi: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "specialist_count_distribution": distribution_table(
            multi, "n_specialists", "n_specialists"
        ),
        "difficulty_distribution": distribution_table(multi, "difficulty"),
        "lead_specialty_distribution": distribution_table(multi, "lead_specialty"),
        "critic_trigger_distribution": distribution_table(multi, "critic_trigger"),
    }


def _exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    n = baseline_only + candidate_only
    if n == 0:
        return 1.0
    k = min(baseline_only, candidate_only)
    tail = sum(math.comb(n, i) * 0.5**n for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def paired_accuracy_comparison(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate_name: str,
    baseline_name: str,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare systems only on matched fixed_case_id values."""
    if candidate["fixed_case_id"].duplicated().any() or baseline["fixed_case_id"].duplicated().any():
        raise ValueError("Paired comparison requires one row per fixed_case_id per system")
    candidate_cols = candidate[
        ["fixed_case_id", "reference_letter", "predicted_letter", "invalid_answer", "api_error"]
    ].copy()
    baseline_cols = baseline[
        ["fixed_case_id", "reference_letter", "predicted_letter", "invalid_answer", "api_error"]
    ].copy()
    paired = candidate_cols.merge(
        baseline_cols,
        on="fixed_case_id",
        how="inner",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(f"No matched cases for {candidate_name} vs {baseline_name}")
    if not (
        paired["reference_letter_candidate"] == paired["reference_letter_baseline"]
    ).all():
        raise ValueError("Gold labels differ across paired system files")

    candidate_valid = (
        paired["predicted_letter_candidate"].astype(str).str.fullmatch(r"[A-H]")
        & ~_as_bool_series(paired["invalid_answer_candidate"])
        & ~_as_bool_series(paired["api_error_candidate"])
    )
    baseline_valid = (
        paired["predicted_letter_baseline"].astype(str).str.fullmatch(r"[A-H]")
        & ~_as_bool_series(paired["invalid_answer_baseline"])
        & ~_as_bool_series(paired["api_error_baseline"])
    )
    paired["candidate_correct"] = candidate_valid & (
        paired["predicted_letter_candidate"] == paired["reference_letter_candidate"]
    )
    paired["baseline_correct"] = baseline_valid & (
        paired["predicted_letter_baseline"] == paired["reference_letter_baseline"]
    )
    differences = (
        paired["candidate_correct"].astype(float)
        - paired["baseline_correct"].astype(float)
    ).to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    bootstrap = differences[sampled].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    candidate_only = int((paired["candidate_correct"] & ~paired["baseline_correct"]).sum())
    baseline_only = int((~paired["candidate_correct"] & paired["baseline_correct"]).sum())
    stats = {
        "comparison": f"{candidate_name} - {baseline_name}",
        "n_paired": len(paired),
        "candidate_accuracy": float(paired["candidate_correct"].mean()),
        "baseline_accuracy": float(paired["baseline_correct"].mean()),
        "paired_accuracy_difference": float(differences.mean()),
        "bootstrap_95ci_low": float(ci_low),
        "bootstrap_95ci_high": float(ci_high),
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "mcnemar_exact_p": _exact_mcnemar_p(baseline_only, candidate_only),
    }
    return stats, paired


def difficulty_analysis(
    multi: pd.DataFrame,
    single_gpt5: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "fixed_case_id",
        "difficulty",
        "correct",
        "n_specialists",
        "critic_used",
        "total_tokens",
        "latency_seconds",
        "estimated_cost_usd",
    ]
    m = multi[[column for column in columns if column in multi]].copy()
    s = single_gpt5[["fixed_case_id", "correct"]].rename(
        columns={"correct": "single_gpt5_correct"}
    )
    merged = m.merge(s, on="fixed_case_id", how="inner", validate="one_to_one")
    merged["correct"] = _as_bool_series(merged["correct"])
    merged["single_gpt5_correct"] = _as_bool_series(merged["single_gpt5_correct"])
    grouped = merged.groupby("difficulty", dropna=False)
    rows: list[dict[str, Any]] = []
    for difficulty, group in grouped:
        rows.append(
            {
                "difficulty": difficulty,
                "n": len(group),
                "multi_accuracy": float(group["correct"].mean()),
                "single_gpt5_accuracy": float(group["single_gpt5_correct"].mean()),
                "accuracy_difference": float(
                    group["correct"].astype(float).mean()
                    - group["single_gpt5_correct"].astype(float).mean()
                ),
                "mean_specialists": float(pd.to_numeric(group["n_specialists"]).mean()),
                "critic_used_rate": float(group["critic_used"].astype(bool).mean()),
                "mean_total_tokens": float(pd.to_numeric(group["total_tokens"]).mean()),
                "mean_latency_seconds": float(pd.to_numeric(group["latency_seconds"]).mean()),
                "mean_estimated_cost_usd": float(
                    pd.to_numeric(group["estimated_cost_usd"]).mean()
                ),
            }
        )
    order = {"simple": 0, "moderate": 1, "complex": 2, "very_complex": 3}
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["difficulty"].map(order).fillna(99)
        result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return result


def panel_size_analysis(multi: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_size, group in multi.groupby("n_specialists"):
        rows.append(
            {
                "n_specialists": int(panel_size),
                "n": len(group),
                "accuracy": float(_as_bool_series(group["correct"]).mean()),
                "critic_used_rate": float(_bool_series(group, "critic_used").mean()),
                "mean_total_tokens": float(_numeric_series(group, "total_tokens").mean()),
                "mean_latency_seconds": float(_numeric_series(group, "latency_seconds").mean()),
                "mean_estimated_cost_usd": float(
                    _numeric_series(group, "estimated_cost_usd").mean()
                ),
                "interpretation_note": "Panel size is confounded with routed case difficulty.",
            }
        )
    return pd.DataFrame(rows).sort_values("n_specialists").reset_index(drop=True)


def case_disagreement_table(
    dataset: pd.DataFrame,
    single_gpt5: pd.DataFrame,
    multi: pd.DataFrame,
) -> pd.DataFrame:
    dataset_columns = [
        column
        for column in ("fixed_case_id", "instruction", "input", "reference_letter")
        if column in dataset
    ]
    base = dataset[dataset_columns].copy()
    single = single_gpt5[
        ["fixed_case_id", "predicted_letter", "correct", "confidence"]
    ].rename(
        columns={
            "predicted_letter": "single_gpt5_answer",
            "correct": "single_gpt5_correct",
            "confidence": "single_gpt5_confidence",
        }
    )
    multi_columns = [
        "fixed_case_id",
        "predicted_letter",
        "correct",
        "difficulty",
        "selected_specialties",
        "specialist_opinions",
        "critic_used",
        "gpt5_confidence",
    ]
    multi_table = multi[[column for column in multi_columns if column in multi]].rename(
        columns={
            "predicted_letter": "multi_answer",
            "correct": "multi_correct",
            "gpt5_confidence": "multi_gpt5_confidence",
        }
    )
    merged = base.merge(single, on="fixed_case_id", how="inner", validate="one_to_one")
    merged = merged.merge(multi_table, on="fixed_case_id", how="inner", validate="one_to_one")
    single_correct = _as_bool_series(merged["single_gpt5_correct"])
    multi_correct = _as_bool_series(merged["multi_correct"])
    merged["outcome_category"] = np.select(
        [
            single_correct & multi_correct,
            ~single_correct & ~multi_correct,
            ~single_correct & multi_correct,
            single_correct & ~multi_correct,
        ],
        [
            "both correct",
            "both wrong",
            "multi-agent only correct",
            "single GPT-5 only correct",
        ],
        default="unclassified",
    )
    return merged.sort_values(["outcome_category", "fixed_case_id"]).reset_index(drop=True)


def validate_comparison_inputs(systems: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Reject mixed datasets, labels, splits, or duplicate case rows."""
    checks: dict[str, Any] = {}
    for name, frame in systems.items():
        if frame.empty:
            raise ValueError(f"{name} prediction file is empty")
        if frame["fixed_case_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate fixed_case_id values")
        checks[f"{name}_n"] = len(frame)
    dataset_fingerprints = {
        str(value)
        for frame in systems.values()
        for value in frame["dataset_fingerprint"].dropna().unique()
    }
    label_fingerprints = {
        str(value)
        for frame in systems.values()
        for value in frame["label_fingerprint"].dropna().unique()
    }
    splits = {
        str(value)
        for frame in systems.values()
        for value in frame["split"].dropna().unique()
    }
    if len(dataset_fingerprints) != 1:
        raise ValueError("System files have different dataset fingerprints")
    if len(label_fingerprints) != 1:
        raise ValueError("System files have different label fingerprints")
    if len(splits) != 1:
        raise ValueError("System files have different splits")
    checks.update(
        {
            "dataset_fingerprint": next(iter(dataset_fingerprints)),
            "label_fingerprint": next(iter(label_fingerprints)),
            "split": next(iter(splits)),
        }
    )
    return checks
