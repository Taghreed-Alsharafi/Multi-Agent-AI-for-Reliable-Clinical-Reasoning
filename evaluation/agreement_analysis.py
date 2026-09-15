"""Offline agreement and paired-outcome analysis for clinical MCQ runs.

This module does not alter production consensus or trigger logic. It analyzes
saved specialist opinions and final predictions without making API calls.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import itertools
import json
import math
import statistics
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from orchestrator.consensus import compute_consensus


VALID_LETTERS = tuple("ABCDEFGH")


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def normalize_specialist_opinions(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate saved opinions and return clean records plus exclusion reasons."""
    raw = _json_value(value, [])
    if not isinstance(raw, list):
        return [], ["specialist_opinions_not_list"]
    clean: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            issues.append(f"opinion_{index}_not_object")
            continue
        opinion = dict(item)
        letter = str(opinion.get("answer_letter", opinion.get("answer", ""))).strip().upper()
        abstain = bool(opinion.get("abstain", False)) or not letter
        if letter and letter not in VALID_LETTERS:
            issues.append(f"opinion_{index}_invalid_letter")
            abstain = True
            letter = ""
        confidence = pd.to_numeric(pd.Series([opinion.get("confidence")]), errors="coerce").iloc[0]
        if pd.notna(confidence) and not 0 <= float(confidence) <= 1:
            issues.append(f"opinion_{index}_invalid_confidence")
            confidence = np.nan
        opinion.update(
            answer_letter=letter,
            confidence=float(confidence) if pd.notna(confidence) else np.nan,
            abstain=abstain,
            specialty=str(opinion.get("specialty", opinion.get("role", f"specialist_{index + 1}"))),
        )
        clean.append(opinion)
    return clean, issues


def compute_legacy_agreement(opinions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce orchestrator.consensus confidence consistency exactly."""
    report = compute_consensus([dict(item) for item in opinions])
    return {
        "legacy_agreement_score": report.agreement_score,
        "legacy_agreement_level": report.level,
        "mean_specialist_confidence": report.mean_confidence,
        "confidence_spread": report.confidence_spread,
        "number_participating": report.participating,
        "number_abstained": report.abstained,
        "legacy_outliers": report.outliers,
    }


def case_vote_features(opinions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute categorical per-case agreement; abstentions are not categories."""
    letters = [
        str(item.get("answer_letter", "")).upper()
        for item in opinions
        if not bool(item.get("abstain", False))
        and str(item.get("answer_letter", "")).upper() in VALID_LETTERS
    ]
    abstained = sum(bool(item.get("abstain", False)) or not item.get("answer_letter") for item in opinions)
    if not letters:
        return {
            "modal_answer": "", "modal_vote_fraction": np.nan, "number_unique_answers": 0,
            "vote_entropy": np.nan, "unanimous": False, "majority_agreement": False,
            "number_answered": 0, "number_abstained_categorical": abstained,
        }
    counts = Counter(letters)
    modal, modal_n = counts.most_common(1)[0]
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(letters)
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    normalized_entropy = entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0
    fraction = modal_n / len(letters)
    return {
        "modal_answer": modal, "modal_vote_fraction": fraction,
        "number_unique_answers": len(counts), "vote_entropy": normalized_entropy,
        "unanimous": len(counts) == 1, "majority_agreement": fraction > 0.5,
        "number_answered": len(letters), "number_abstained_categorical": abstained,
    }
def krippendorff_alpha_nominal(units: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Nominal Krippendorff alpha using coincidence-matrix weighting.

    Each unit is one case. Missing values and explicit abstentions (None/empty)
    are omitted rather than treated as answer categories.
    """
    coincidence: Counter[tuple[str, str]] = Counter()
    used = ratings = 0
    missing = 0
    total_slots = 0
    for unit in units:
        values = [str(v).strip().upper() if v is not None else "" for v in unit]
        total_slots += len(values)
        valid = [v for v in values if v in VALID_LETTERS]
        missing += len(values) - len(valid)
        if len(valid) < 2:
            continue
        used += 1
        ratings += len(valid)
        denominator = len(valid) - 1
        for left_index, left in enumerate(valid):
            for right_index, right in enumerate(valid):
                if left_index != right_index:
                    coincidence[(left, right)] += 1 / denominator
    total = sum(coincidence.values())
    if total <= 1:
        alpha = np.nan
    else:
        observed = sum(value for (left, right), value in coincidence.items() if left != right) / total
        marginals: Counter[str] = Counter()
        for (left, _), value in coincidence.items():
            marginals[left] += value
        expected = 1 - sum(count * (count - 1) for count in marginals.values()) / (total * (total - 1))
        alpha = 1 - observed / expected if expected > 0 else (1.0 if observed == 0 else np.nan)
    return {
        "krippendorff_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "number_cases_used": used, "number_ratings": ratings,
        "missing_abstention_rate": missing / total_slots if total_slots else np.nan,
    }


def bootstrap_alpha(units: Sequence[Sequence[Any]], n_bootstrap: int = 2000, seed: int = 2026) -> tuple[float, float]:
    if not units:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = [units[index] for index in rng.integers(0, len(units), len(units))]
        value = krippendorff_alpha_nominal(sample)["krippendorff_alpha"]
        if np.isfinite(value):
            values.append(value)
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def kendalls_w(rankings: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Kendall W for complete agent-by-option rankings; ties use average ranks."""
    matrix = np.asarray(rankings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        return {"kendall_w": np.nan, "p_value": np.nan, "number_agents": 0, "number_ranked_options": 0}
    ranked = np.vstack([stats.rankdata(row, method="average") for row in matrix])
    m, n = ranked.shape
    rank_sums = ranked.sum(axis=0)
    s_value = float(((rank_sums - rank_sums.mean()) ** 2).sum())
    tie_correction = sum(sum(count**3 - count for count in Counter(row).values()) for row in ranked)
    denominator = m * m * (n**3 - n) - m * tie_correction
    w_value = 12 * s_value / denominator if denominator > 0 else np.nan
    chi_square = m * (n - 1) * w_value if np.isfinite(w_value) else np.nan
    return {
        "kendall_w": float(w_value), "p_value": float(stats.chi2.sf(chi_square, n - 1)),
        "number_agents": m, "number_ranked_options": n,
    }


def normalize_probability_distribution(values: Mapping[str, Any], letters: Sequence[str]) -> np.ndarray:
    array = np.asarray([values.get(letter, np.nan) for letter in letters], dtype=float)
    if not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
        raise ValueError("Option probabilities must be finite, nonnegative, and have positive mass")
    return array / array.sum()


def approximate_distribution(answer: str, confidence: float, letters: Sequence[str]) -> np.ndarray:
    """Exploratory only: spread residual selected-answer confidence uniformly."""
    if answer not in letters or not np.isfinite(confidence) or not 0 <= confidence <= 1 or len(letters) < 2:
        raise ValueError("A valid answer, confidence in [0,1], and at least two options are required")
    result = np.full(len(letters), (1 - confidence) / (len(letters) - 1), dtype=float)
    result[list(letters).index(answer)] = confidence
    return result


def pairwise_jsd(distributions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Pairwise Jensen-Shannon divergence in base 2, normalized to [0, 1]."""
    normalized = []
    for values in distributions:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
            raise ValueError("Each distribution must be finite, nonnegative, and have positive mass")
        normalized.append(array / array.sum())
    if len(normalized) < 2 or len({len(item) for item in normalized}) != 1:
        return {key: np.nan for key in ("mean_pairwise_jsd", "max_pairwise_jsd", "minimum_pairwise_jsd", "jsd_variance")} | {"jsd_number_agents": len(normalized)}
    values = [float(stats.entropy((p + q) / 2, base=2) - (stats.entropy(p, base=2) + stats.entropy(q, base=2)) / 2) for p, q in itertools.combinations(normalized, 2)]
    return {
        "mean_pairwise_jsd": float(np.mean(values)), "max_pairwise_jsd": float(np.max(values)),
        "minimum_pairwise_jsd": float(np.min(values)), "jsd_variance": float(np.var(values)),
        "jsd_number_agents": len(normalized),
    }


def compute_case_agreement_features(row: Mapping[str, Any], allow_approximated_jsd: bool = False) -> dict[str, Any]:
    opinions, issues = normalize_specialist_opinions(row.get("specialist_opinions", []))
    output = compute_legacy_agreement(opinions) | case_vote_features(opinions)
    output["agreement_validation_issues"] = issues
    distributions = []
    source = "unavailable"
    for opinion in opinions:
        option_scores = opinion.get("option_scores")
        if isinstance(option_scores, dict):
            letters = sorted(letter for letter in option_scores if letter in VALID_LETTERS)
            if len(letters) >= 2:
                distributions.append(normalize_probability_distribution(option_scores, letters))
                source = "reported_option_scores"
        elif allow_approximated_jsd and not opinion.get("abstain") and np.isfinite(opinion.get("confidence", np.nan)):
            letters = list(str(row.get("option_letters", "ABCD")))
            distributions.append(approximate_distribution(opinion["answer_letter"], opinion["confidence"], letters))
            source = "approximated_from_selected_confidence"
    output.update(pairwise_jsd(distributions))
    output["jsd_probability_source"] = source if len(distributions) >= 2 else "unavailable"
    return output


def compute_maca_score(
    opinions: Sequence[Mapping[str, Any]],
    *,
    mean_pairwise_jsd: float = np.nan,
    probability_source: str = "unavailable",
) -> dict[str, Any]:
    """Compute the experimental Multi-Agent Clinical Agreement (MACA) score.

    MACA never uses the reference answer. Its components are answer concordance,
    probability similarity, confidence consistency, participation, and a mild
    severity penalty for multiple distinct answers. A geometric mean prevents a
    strong component from fully compensating for a weak one.
    """
    vote = case_vote_features(opinions)
    assigned = len(opinions)
    participating = int(vote["number_answered"])
    participation = participating / assigned if assigned else 0.0
    answer_agreement = vote["modal_vote_fraction"]
    if not np.isfinite(answer_agreement):
        answer_agreement = 0.0

    confidences = np.asarray(
        [
            item.get("confidence", np.nan)
            for item in opinions
            if not item.get("abstain", False)
            and np.isfinite(item.get("confidence", np.nan))
        ],
        dtype=float,
    )
    confidence_sd = float(np.std(confidences)) if len(confidences) else 1.0
    # 0.5 is the maximum population SD for bounded values in [0, 1].
    confidence_agreement = float(np.clip(1 - confidence_sd / 0.5, 0, 1))
    probability_available = np.isfinite(mean_pairwise_jsd)
    probability_agreement = float(np.clip(1 - mean_pairwise_jsd, 0, 1)) if probability_available else np.nan

    components = [answer_agreement, confidence_agreement]
    if probability_available:
        components.append(probability_agreement)
    geometric = float(np.prod(np.clip(components, 0, 1)) ** (1 / len(components)))
    unique_answers = int(vote["number_unique_answers"])
    # Severity is zero for unanimity and rises smoothly toward one as distinct
    # answers approach the number of participating agents.
    severity = (
        (unique_answers - 1) / max(participating - 1, 1)
        if participating > 1
        else 0.0
    )
    severity_penalty = 1 - 0.25 * float(np.clip(severity, 0, 1))
    maca = float(np.clip(geometric * participation * severity_penalty, 0, 1))
    return {
        "maca_score": maca,
        "maca_answer_agreement": float(answer_agreement),
        "maca_probability_agreement": probability_agreement,
        "maca_confidence_agreement": confidence_agreement,
        "maca_participation": float(participation),
        "maca_disagreement_severity": float(severity),
        "maca_probability_source": probability_source if probability_available else "unavailable",
    }


def robustness_to_missing_agents(
    frame: pd.DataFrame,
    scorer: Callable[[list[dict[str, Any]], Mapping[str, Any]], float],
    *,
    n_repeats: int = 200,
    seed: int = 2026,
) -> dict[str, float]:
    """Estimate score stability after randomly removing one participating agent."""
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    usable = 0
    records = frame.to_dict("records")
    candidates = []
    for record in records:
        opinions, _ = normalize_specialist_opinions(record.get("specialist_opinions", []))
        active = [item for item in opinions if not item.get("abstain", False)]
        if len(active) >= 2:
            candidates.append((record, active))
    if not candidates:
        return {"robustness_mae": np.nan, "stability": np.nan, "robustness_trials": 0}
    for _ in range(n_repeats):
        record, active = candidates[int(rng.integers(0, len(candidates)))]
        full = scorer(active, record)
        reduced = list(active)
        reduced.pop(int(rng.integers(0, len(reduced))))
        perturbed = scorer(reduced, record)
        if np.isfinite(full) and np.isfinite(perturbed):
            deltas.append(abs(float(full) - float(perturbed)))
            usable += 1
    mae = float(np.mean(deltas)) if deltas else np.nan
    return {
        "robustness_mae": mae,
        "stability": float(np.clip(1 - mae, 0, 1)) if np.isfinite(mae) else np.nan,
        "robustness_trials": usable,
    }


def build_case_analysis(multi: pd.DataFrame, allow_approximated_jsd: bool = False) -> pd.DataFrame:
    if multi["fixed_case_id"].duplicated().any():
        raise ValueError("Duplicate multi-agent fixed_case_id values")
    rows = []
    for record in multi.to_dict("records"):
        features = compute_case_agreement_features(record, allow_approximated_jsd)
        rows.append(dict(record) | features)
    return pd.DataFrame(rows)


def calibration_table(confidence: Iterable[Any], correct: Iterable[Any], bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"confidence": pd.to_numeric(pd.Series(confidence), errors="coerce"), "correct": pd.Series(correct).astype(float)}).dropna()
    frame = frame[frame.confidence.between(0, 1)]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = frame.confidence.ge(low) & (frame.confidence.le(high) if index == bins - 1 else frame.confidence.lt(high))
        subset = frame[mask]
        rows.append({"bin": f"{low:.1f}-{high:.1f}", "n": len(subset), "mean_confidence": subset.confidence.mean(), "observed_accuracy": subset.correct.mean(), "calibration_gap": subset.confidence.mean() - subset.correct.mean()})
    return pd.DataFrame(rows)


def calibration_summary(confidence: Iterable[Any], correct: Iterable[Any]) -> dict[str, float]:
    frame = calibration_table(confidence, correct)
    raw = pd.DataFrame({"confidence": pd.to_numeric(pd.Series(confidence), errors="coerce"), "correct": pd.Series(correct).astype(float)}).dropna()
    raw = raw[raw.confidence.between(0, 1)]
    return {
        "n": len(raw), "brier_score": float(np.mean((raw.confidence - raw.correct) ** 2)) if len(raw) else np.nan,
        "mean_confidence": raw.confidence.mean(), "accuracy": raw.correct.mean(),
        "confidence_accuracy_gap": raw.confidence.mean() - raw.correct.mean(),
        "ece": float((frame.n * frame.calibration_gap.abs()).sum() / frame.n.sum()) if frame.n.sum() else np.nan,
    }


def bootstrap_statistic(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], n_bootstrap: int = 2000, seed: int = 2026) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        try:
            value = statistic(sample)
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            values.append(value)
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def paired_single_multi(single: pd.DataFrame, multi: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 2026) -> tuple[pd.DataFrame, dict[str, Any]]:
    for name, frame in (("single", single), ("multi", multi)):
        if frame.fixed_case_id.duplicated().any():
            raise ValueError(f"Duplicate {name} fixed_case_id values")
    required = ["dataset_fingerprint", "label_fingerprint", "split"]
    for column in required:
        if column in single and column in multi and set(single[column]) != set(multi[column]):
            raise ValueError(f"Single/multi {column} mismatch")
    left = single[["fixed_case_id", "reference_letter", "predicted_letter", "correct", "confidence"]].rename(columns={"predicted_letter": "single_answer", "correct": "single_correct", "confidence": "single_confidence"})
    right = multi[["fixed_case_id", "reference_letter", "predicted_letter", "correct", "confidence"]].rename(columns={"predicted_letter": "multi_answer", "correct": "multi_correct", "confidence": "multi_confidence"})
    paired = left.merge(right, on=["fixed_case_id", "reference_letter"], how="inner", validate="one_to_one")
    paired[["single_correct", "multi_correct"]] = paired[["single_correct", "multi_correct"]].astype(bool)
    b = int((~paired.single_correct & paired.multi_correct).sum())
    c = int((paired.single_correct & ~paired.multi_correct).sum())
    discordant = b + c
    exact = discordant < 25
    if discordant == 0:
        statistic, p_value = 0.0, 1.0
    elif exact:
        result = stats.binomtest(min(b, c), discordant, 0.5, alternative="two-sided")
        statistic, p_value = float(min(b, c)), float(result.pvalue)
    else:
        statistic = (abs(b - c) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, 1))
    difference = float(paired.multi_correct.mean() - paired.single_correct.mean()) if len(paired) else np.nan
    ci = bootstrap_statistic(paired, lambda x: float(x.multi_correct.mean() - x.single_correct.mean()), n_bootstrap, seed)
    summary = {
        "n_matched": len(paired), "single_accuracy": paired.single_correct.mean(), "multi_accuracy": paired.multi_correct.mean(),
        "absolute_accuracy_difference": difference,
        "relative_improvement": difference / paired.single_correct.mean() if len(paired) and paired.single_correct.mean() else np.nan,
        "difference_ci_low": ci[0], "difference_ci_high": ci[1],
        "single_wrong_multi_correct": b, "single_correct_multi_wrong": c,
        "both_correct": int((paired.single_correct & paired.multi_correct).sum()), "both_wrong": int((~paired.single_correct & ~paired.multi_correct).sum()),
        "mcnemar_method": "exact_binomial" if exact else "continuity_corrected_chi_square",
        "mcnemar_statistic": statistic, "mcnemar_p_value": p_value,
    }
    return paired, summary


def evaluate_agreement_predictor(frame: pd.DataFrame, metric: str, direction: str, n_bootstrap: int = 1000, seed: int = 2026) -> dict[str, Any]:
    subset = frame[[metric, "correct"]].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna()
    y = subset.correct.astype(int).to_numpy()
    score = subset[metric].to_numpy(float)
    if direction == "lower_is_better":
        score = -score
    if len(subset) < 2 or len(np.unique(y)) < 2:
        return {"metric": metric, "direction": direction, "n": len(subset), "auroc": np.nan, "auprc": np.nan, "point_biserial_r": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))
    correlation = float(stats.pointbiserialr(y, score).statistic)
    work = pd.DataFrame({"y": y, "score": score})
    ci = bootstrap_statistic(work, lambda x: float(roc_auc_score(x.y, x.score)) if x.y.nunique() == 2 else np.nan, n_bootstrap, seed)
    return {"metric": metric, "direction": direction, "n": len(subset), "auroc": auroc, "auprc": auprc, "point_biserial_r": correlation, "ci_low": ci[0], "ci_high": ci[1]}


def create_disagreement_trigger(features: Mapping[str, Any], config: Mapping[str, float]) -> dict[str, Any]:
    """Proposed development-tunable trigger; never tune this on locked test."""
    reasons = []
    modal = features.get("modal_vote_fraction", np.nan)
    jsd = features.get("mean_pairwise_jsd", np.nan)
    confidence = features.get("mean_specialist_confidence", np.nan)
    router = features.get("router_confidence", np.nan)
    if np.isfinite(modal) and modal < config["min_modal_vote_fraction"]:
        reasons.append("categorical_disagreement")
    if np.isfinite(jsd) and jsd >= config["high_jsd_threshold"]:
        reasons.append("high_jsd")
    if np.isfinite(confidence) and confidence < config["low_confidence_threshold"]:
        reasons.append("low_specialist_confidence")
    if np.isfinite(router) and router < config["router_confidence_threshold"]:
        reasons.append("low_router_confidence")
    if features.get("number_abstained", 0) > 0:
        reasons.append("specialist_abstention")
    if features.get("difficulty") in {"complex", "very_complex", "difficult"}:
        reasons.append("high_case_difficulty")
    return {"trigger_critic": bool(reasons), "reasons": reasons}
