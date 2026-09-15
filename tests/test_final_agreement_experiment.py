from __future__ import annotations

import asyncio
import math

import pytest

from evaluation.final_agreement_experiment import (
    FINAL_SAFETY_MODEL,
    FINAL_SAFETY_MAX_OUTPUT_TOKENS,
    FINAL_SAFETY_RETRY_MAX_OUTPUT_TOKENS,
    FINAL_SAFETY_REASONING_EFFORT,
    FinalAgreementExperiment,
    COMPLEX_REASONING_MODEL,
    VALID_METHODS,
    care_case_consensus,
    categorical_case_disagreement,
    classification_summary,
    clinical_guideline_override,
    fit_care_configuration,
    dynamic_model_plan,
    independent_solver_support,
    kendall_w,
    krippendorff_alpha_nominal,
    mean_pairwise_jsd,
    normalize_specialist_output,
    run_ua_validation_tests,
    specialist_calibration_examples,
    tune_disagreement_threshold,
    ua_case_disagreement,
    ua_distance,
    uncertainty_aware_krippendorff_alpha,
    vote_entropy,
)
from evaluation.mcq_hybrid_v2 import CallResult


def opinion(answer, probabilities, ranking=None):
    ranking = ranking or sorted(probabilities, key=probabilities.get, reverse=True)
    return {
        "answer": answer,
        "probabilities": probabilities,
        "ranking": ranking,
        "abstained": False,
    }


def test_normalization_repairs_probabilities_and_complete_ranking():
    normalized = normalize_specialist_output(
        {"answer": "A", "probabilities": {"A": 7, "B": 2, "C": 1, "D": 0}, "ranking": ["A", "B"]},
        list("ABCD"),
    )
    assert normalized["answer"] == "A"
    assert math.isclose(sum(normalized["probabilities"].values()), 1.0)
    assert set(normalized["ranking"]) == set("ABCD")
    assert len(normalized["ranking"]) == 4


def test_vote_entropy_and_categorical_disagreement():
    unanimous = [opinion("A", {"A": 1, "B": 0}) for _ in range(3)]
    split = [
        opinion("A", {"A": 0.9, "B": 0.1}),
        opinion("B", {"A": 0.1, "B": 0.9}),
        opinion("A", {"A": 0.6, "B": 0.4}),
    ]
    assert vote_entropy(unanimous, 2) == pytest.approx(0.0)
    assert vote_entropy(split, 2) > 0.9
    assert categorical_case_disagreement(unanimous) == pytest.approx(0.0)
    assert categorical_case_disagreement(split) == pytest.approx(2 / 3)


def test_jsd_uses_full_probability_distributions():
    close = [
        opinion("A", {"A": 0.51, "B": 0.49}),
        opinion("B", {"A": 0.49, "B": 0.51}),
    ]
    far = [
        opinion("A", {"A": 0.99, "B": 0.01}),
        opinion("B", {"A": 0.01, "B": 0.99}),
    ]
    assert 0 <= mean_pairwise_jsd(close, list("AB")) < mean_pairwise_jsd(far, list("AB")) <= 1


def test_kendall_w_complete_rankings_and_missing_agents():
    identical = [
        opinion("A", {"A": 0.7, "B": 0.2, "C": 0.1}, ["A", "B", "C"]),
        opinion("A", {"A": 0.6, "B": 0.3, "C": 0.1}, ["A", "B", "C"]),
    ]
    assert kendall_w(identical, list("ABC")) == pytest.approx(1.0)
    assert math.isnan(kendall_w(identical[:1], list("ABC")))


def test_nominal_and_uncertainty_aware_alpha_are_bounded_for_example():
    cases = [
        [opinion("A", {"A": 0.8, "B": 0.2}), opinion("A", {"A": 0.7, "B": 0.3})],
        [opinion("A", {"A": 0.6, "B": 0.4}), opinion("B", {"A": 0.3, "B": 0.7})],
    ]
    nominal = krippendorff_alpha_nominal(cases)
    uncertainty = uncertainty_aware_krippendorff_alpha(cases)
    assert math.isfinite(nominal["alpha"])
    assert math.isfinite(uncertainty["ua_kalpha"])
    assert uncertainty["n_cases"] == 2
    assert uncertainty["n_ratings"] == 4


def test_ua_validation_and_lambda_one_nominal_reduction():
    values = run_ua_validation_tests()
    assert values["perfect"] == pytest.approx(0.0)
    assert values["weak_boundary"] < values["strong"]
    first = opinion("A", {"A": 0.51, "B": 0.49})
    second = opinion("B", {"A": 0.49, "B": 0.51})
    assert ua_distance(first, second, lambda_answer=1.0) == pytest.approx(1.0)
    assert ua_case_disagreement([first, second], lambda_answer=1.0) == pytest.approx(1.0)


def test_threshold_tuning_uses_high_scores_for_review():
    tuned = tune_disagreement_threshold([0.0, 0.1, 0.8, 0.9], [False, False, True, True])
    assert tuned["detection_f1"] == pytest.approx(1.0)
    assert 0.1 <= tuned["threshold"] < 0.8
    assert tuned["review_rate"] == pytest.approx(0.5)


def test_macro_metrics_and_safety_rate():
    result = classification_summary(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "B"],
        [False, True, False, False],
    )
    assert result["Accuracy"] == pytest.approx(0.75)
    assert result["Safety Violation Rate"] == pytest.approx(0.25)
    assert 0 <= result["Precision"] <= 1
    assert 0 <= result["Recall"] <= 1
    assert 0 <= result["F1"] <= 1


def test_care_is_bounded_and_rewards_coherent_consensus():
    unanimous = [opinion("A", {"A": 0.85, "B": 0.10, "C": 0.05}) for _ in range(3)]
    split = [
        opinion("A", {"A": 0.80, "B": 0.10, "C": 0.10}),
        opinion("B", {"A": 0.10, "B": 0.80, "C": 0.10}),
        opinion("C", {"A": 0.10, "B": 0.10, "C": 0.80}),
    ]
    config = {"slot_weights": [1, 1, 1], "probability_weight": 0.75, "certainty_exponent": 0.5}
    common = {"letters": list("ABC"), "gold_answer": "A", "split": "development"}
    coherent = care_case_consensus({**common, "specialists": unanimous}, config)
    conflicted = care_case_consensus({**common, "specialists": split}, config)
    assert 0 <= coherent["agreement"] <= 1
    assert 0 <= conflicted["agreement"] <= 1
    assert coherent["agreement"] > conflicted["agreement"]
    assert coherent["risk"] < conflicted["risk"]


def test_care_fit_is_development_only_and_prompt_examples_are_option_safe():
    specialists = [opinion("A", {"A": 0.8, "B": 0.2}) for _ in range(3)]
    development = [
        {"letters": list("AB"), "specialists": specialists, "gold_answer": "A", "split": "development"},
        {"letters": list("AB"), "specialists": specialists, "gold_answer": "A", "split": "development"},
    ]
    fitted = fit_care_configuration(development)
    assert fitted["fitted_split"] == "development"
    assert len(fitted["slot_weights"]) == 3
    with pytest.raises(ValueError, match="development"):
        fit_care_configuration([{**development[0], "split": "test"}])
    examples = specialist_calibration_examples(list("ABCDE"))
    assert "CALIBRATION EXAMPLES" in examples
    assert '"probabilities"' in examples


def test_care_v2_fits_regularized_development_risk_model():
    specialists = [opinion("A", {"A": 0.8, "B": 0.2}) for _ in range(3)]
    records = [
        {
            "letters": list("AB"),
            "specialists": specialists,
            "single_answer": "A" if index % 2 == 0 else "B",
            "gold_answer": "A" if index < 6 else "B",
            "split": "development",
        }
        for index in range(10)
    ]
    fitted = fit_care_configuration(records)
    assert fitted["risk_model"]["version"] == "care_regularized_oof_v2"
    assert len(fitted["risk_model"]["coefficients"]) == 8
    result = care_case_consensus(records[0], fitted)
    assert 0.0 <= result["risk"] <= 1.0
    assert set(result["risk_features"]) == set(fitted["risk_model"]["feature_names"])


def test_independent_solver_support_preserves_credible_minority_evidence():
    record = {
        "single_answer": "G",
        "specialists": [
            opinion("A", {"A": 0.60, "B": 0.05, "C": 0.04, "D": 0.03, "E": 0.03, "F": 0.05, "G": 0.20}, list("AGBFEDC")),
            opinion("A", {"A": 0.65, "B": 0.05, "C": 0.03, "D": 0.02, "E": 0.02, "F": 0.03, "G": 0.20}, list("AGBCFED")),
            opinion("A", {"A": 0.65, "B": 0.10, "C": 0.04, "D": 0.03, "E": 0.03, "F": 0.03, "G": 0.12}, list("ABGCFED")),
        ],
    }
    support = independent_solver_support(record)
    assert support["answer"] == "G"
    assert support["specialist_ranks"] == [2, 2, 3]
    assert support["mean_probability"] > 0.12
    assert support["credible_minority"] is True


def test_cdc_guideline_guard_finds_therapy_by_text_not_letter():
    row = {
        "question": "A pregnant patient has an organism identified on darkfield microscopy.",
        "option_A": "Azithromycin and ceftriaxone",
        "option_B": "Observation",
        "option_C": "Penicillin G",
    }
    override = clinical_guideline_override(row)
    assert override is not None
    assert override["answer"] == "C"
    assert override["rule_id"] == "CDC_PREGNANCY_SYPHILIS_PENICILLIN"
    assert "cdc.gov" in override["source"]
    assert clinical_guideline_override({**row, "question": "A nonpregnant patient has anemia."}) is None


def test_final_safety_uses_one_gpt5_call_and_dedicated_cache(tmp_path):
    class FakeSafetyClient:
        def __init__(self):
            self.calls = []

        async def call_json(self, **kwargs):
            self.calls.append(kwargs)
            return CallResult(
                role=kwargs["role"],
                model=kwargs["model"],
                parsed={"safety_violation": False, "reason": "No final-answer safety issue."},
                raw_text="{}",
                latency_seconds=0.01,
                attempts=1,
            )

    fake = FakeSafetyClient()
    experiment = FinalAgreementExperiment(output_dir=tmp_path, safety_client=fake)
    row = experiment.cases.iloc[0]
    case_id = str(row.fixed_case_id)
    answer = str(row.reference_letter)
    first = asyncio.run(experiment._external_safety_check(case_id, answer))
    second = asyncio.run(experiment._external_safety_check(case_id, answer))
    assert first == second
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["model"] == FINAL_SAFETY_MODEL == "gpt-5-mini"
    assert call["reasoning_effort"] == FINAL_SAFETY_REASONING_EFFORT == "medium"
    assert call["max_output_tokens"] == FINAL_SAFETY_MAX_OUTPUT_TOKENS == 2000
    assert call["allow_retry"] is False
    assert first["attempts"] == 1
    assert (tmp_path / "gpt5_final_safety_outputs.jsonl").exists()


def test_final_safety_retries_once_when_output_budget_is_incomplete(tmp_path):
    class BudgetRetrySafetyClient:
        def __init__(self):
            self.calls = []

        async def call_json(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return CallResult(
                    role=kwargs["role"],
                    model=kwargs["model"],
                    parsed=None,
                    raw_text="",
                    latency_seconds=0.01,
                    status="incomplete",
                    incomplete_reason="max_output_tokens",
                    error="Expecting value: line 1 column 1 (char 0)",
                )
            return CallResult(
                role=kwargs["role"],
                model=kwargs["model"],
                parsed={"safety_violation": False, "reason": "No safety issue."},
                raw_text="{}",
                latency_seconds=0.01,
            )

    fake = BudgetRetrySafetyClient()
    experiment = FinalAgreementExperiment(output_dir=tmp_path, safety_client=fake)
    row = experiment.cases.iloc[0]
    result = asyncio.run(
        experiment._external_safety_check(
            str(row.fixed_case_id),
            str(row.reference_letter),
        )
    )
    assert len(fake.calls) == 2
    assert fake.calls[0]["max_output_tokens"] == FINAL_SAFETY_MAX_OUTPUT_TOKENS
    assert fake.calls[1]["max_output_tokens"] == FINAL_SAFETY_RETRY_MAX_OUTPUT_TOKENS
    assert fake.calls[1]["role"] == "final_safety_budget_retry"
    assert result["attempts"] == result["api_call_count"] == 2


def test_final_safety_retries_transient_network_timeout(tmp_path):
    class TimeoutThenSuccessClient:
        def __init__(self):
            self.calls = []

        async def call_json(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return CallResult(
                    role=kwargs["role"],
                    model=kwargs["model"],
                    parsed=None,
                    raw_text="",
                    latency_seconds=0.01,
                    status="error",
                    error_type="api_error",
                    error="APITimeoutError: Request timed out; ConnectTimeout",
                )
            return CallResult(
                role=kwargs["role"],
                model=kwargs["model"],
                parsed={"safety_violation": False, "reason": "No safety issue."},
                raw_text="{}",
                latency_seconds=0.01,
            )

    fake = TimeoutThenSuccessClient()
    experiment = FinalAgreementExperiment(output_dir=tmp_path, safety_client=fake)
    row = experiment.cases.iloc[0]
    result = asyncio.run(
        experiment._external_safety_check(
            str(row.fixed_case_id),
            str(row.reference_letter),
        )
    )
    assert len(fake.calls) == 2
    assert fake.calls[1]["role"] == "final_safety_network_retry"
    assert result["api_call_count"] == 2


def test_dynamic_model_plan_escalates_only_complex_or_important_roles():
    routine = dynamic_model_plan({"difficulty": "moderate", "clinical_importance": "routine"})
    complex_case = dynamic_model_plan({"difficulty": "complex", "clinical_importance": "high"})
    critical = dynamic_model_plan({"difficulty": "very_complex", "clinical_importance": "critical"})
    assert routine["specialist_models"] == ["gpt-4.1"] * 3
    assert routine["judge_model"] == "gpt-4.1"
    assert complex_case["specialist_models"].count(COMPLEX_REASONING_MODEL) == 1
    assert complex_case["judge_model"] == COMPLEX_REASONING_MODEL == "gpt-5-mini"
    assert critical["specialist_models"].count(COMPLEX_REASONING_MODEL) == 2
    assert critical["judge_reasoning_effort"] == "high"
    assert critical["final_safety_model"] == "gpt-5-mini"


def test_active_comparison_excludes_removed_methods():
    assert VALID_METHODS == (
        "jsd",
        "kendall_w",
        "krippendorff_alpha",
        "care_consensus",
    )


def test_fresh_api_mode_rejects_existing_outputs(tmp_path):
    (tmp_path / "shared_agent_outputs.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Fresh API mode"):
        FinalAgreementExperiment(
            output_dir=tmp_path,
            reuse_existing_outputs=False,
        )
