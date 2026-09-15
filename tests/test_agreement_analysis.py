from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.agreement_analysis import (
    approximate_distribution,
    calibration_summary,
    case_vote_features,
    compute_legacy_agreement,
    compute_maca_score,
    create_disagreement_trigger,
    evaluate_agreement_predictor,
    kendalls_w,
    krippendorff_alpha_nominal,
    paired_single_multi,
    pairwise_jsd,
    robustness_to_missing_agents,
)


def test_legacy_agreement_exactly_matches_confidence_consensus():
    result = compute_legacy_agreement([
        {"specialty": "one", "confidence": 0.8},
        {"specialty": "two", "confidence": 0.6},
        {"specialty": "three", "confidence": 0.0},
    ])
    assert result["legacy_agreement_score"] == 0.56
    assert result["number_participating"] == 2
    assert result["number_abstained"] == 1


def test_case_votes_separate_abstention_from_answer_categories():
    result = case_vote_features([
        {"answer_letter": "A", "abstain": False},
        {"answer_letter": "A", "abstain": False},
        {"answer_letter": "B", "abstain": True},
        {"answer_letter": "", "abstain": True},
    ])
    assert result["modal_vote_fraction"] == 1
    assert result["number_unique_answers"] == 1
    assert result["number_abstained_categorical"] == 2


def test_nominal_alpha_perfect_and_missing_ratings():
    result = krippendorff_alpha_nominal([["A", "A", None], ["B", "B", ""], ["A", "A", "A"]])
    assert result["krippendorff_alpha"] == pytest.approx(1.0)
    assert result["number_cases_used"] == 3
    assert result["number_ratings"] == 7
    assert result["missing_abstention_rate"] == pytest.approx(2 / 9)


def test_kendalls_w_detects_identical_rankings():
    result = kendalls_w([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    assert result["kendall_w"] == pytest.approx(1.0)
    assert result["number_agents"] == 3


def test_jsd_is_normalized_and_probability_validation_is_strict():
    result = pairwise_jsd([[1, 0], [0, 1]])
    assert result["mean_pairwise_jsd"] == pytest.approx(1.0)
    assert pairwise_jsd([[0.5, 0.5], [0.5, 0.5]])["mean_pairwise_jsd"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pairwise_jsd([[1, -1], [0.5, 0.5]])
    assert approximate_distribution("B", 0.7, list("ABCD")).sum() == pytest.approx(1.0)


def _prediction(correct_values, answers, system):
    return pd.DataFrame({
        "fixed_case_id": [f"c{i}" for i in range(len(correct_values))],
        "reference_letter": ["A"] * len(correct_values),
        "predicted_letter": answers,
        "correct": correct_values,
        "confidence": [0.8] * len(correct_values),
        "dataset_fingerprint": ["d"] * len(correct_values),
        "label_fingerprint": ["l"] * len(correct_values),
        "split": ["development"] * len(correct_values),
        "system": [system] * len(correct_values),
    })


def test_paired_comparison_reports_exact_mcnemar_transitions():
    single = _prediction([True, False, False, True], ["A", "B", "B", "A"], "single")
    multi = _prediction([True, True, False, False], ["A", "A", "B", "B"], "multi")
    paired, summary = paired_single_multi(single, multi, n_bootstrap=100)
    assert len(paired) == 4
    assert summary["single_wrong_multi_correct"] == 1
    assert summary["single_correct_multi_wrong"] == 1
    assert summary["mcnemar_method"] == "exact_binomial"
    assert summary["mcnemar_p_value"] == 1


def test_calibration_predictor_and_configured_trigger():
    calibration = calibration_summary([0.9, 0.8, 0.2], [1, 1, 0])
    assert calibration["brier_score"] == pytest.approx(0.03)
    frame = pd.DataFrame({"metric": [0.9, 0.8, 0.2, 0.1], "correct": [1, 1, 0, 0]})
    predictor = evaluate_agreement_predictor(frame, "metric", "higher_is_better", n_bootstrap=100)
    assert predictor["auroc"] == 1
    trigger = create_disagreement_trigger(
        {"modal_vote_fraction": 0.5, "mean_pairwise_jsd": 0.4, "mean_specialist_confidence": 0.9, "router_confidence": np.nan, "number_abstained": 0, "difficulty": "moderate"},
        {"min_modal_vote_fraction": 0.75, "high_jsd_threshold": 0.3, "low_confidence_threshold": 0.6, "router_confidence_threshold": 0.5},
    )
    assert trigger["trigger_critic"]
    assert trigger["reasons"] == ["categorical_disagreement", "high_jsd"]


def test_maca_is_bounded_gold_independent_and_penalizes_disagreement():
    unanimous = [
        {"answer_letter": "A", "confidence": 0.8, "abstain": False},
        {"answer_letter": "A", "confidence": 0.8, "abstain": False},
    ]
    split = [
        {"answer_letter": "A", "confidence": 0.8, "abstain": False},
        {"answer_letter": "B", "confidence": 0.8, "abstain": False},
    ]
    high = compute_maca_score(unanimous, mean_pairwise_jsd=0.0, probability_source="reported")
    low = compute_maca_score(split, mean_pairwise_jsd=1.0, probability_source="reported")
    assert 0 <= low["maca_score"] < high["maca_score"] <= 1
    assert "gold" not in high


def test_maca_penalizes_abstention_and_missing_agent_robustness_runs():
    opinions = [
        {"answer_letter": "A", "confidence": 0.9, "abstain": False},
        {"answer_letter": "", "confidence": 0.0, "abstain": True},
    ]
    result = compute_maca_score(opinions)
    assert result["maca_participation"] == 0.5
    assert result["maca_score"] <= 0.5
    frame = pd.DataFrame({"specialist_opinions": [[
        {"answer_letter": "A", "confidence": 0.8},
        {"answer_letter": "A", "confidence": 0.7},
    ]]})
    robust = robustness_to_missing_agents(
        frame, lambda items, _: compute_maca_score(items)["maca_score"], n_repeats=10
    )
    assert robust["robustness_trials"] == 10
    assert 0 <= robust["stability"] <= 1
