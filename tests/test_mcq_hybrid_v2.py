from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation import mcq_hybrid_v2 as v2
from evaluation.mcq_gold_verification import BenchmarkSource, verify_gold_against_originals
from evaluation.mcq_hybrid import parse_options
from evaluation.mcq_v2_analysis import (
    paired_accuracy_comparison,
    performance_metrics,
    validate_comparison_inputs,
)


def _question(index: int = 0) -> str:
    return f"Case {index}: choose the best answer.\nA. Alpha {index}\nB. Beta {index}\nC. Gamma {index}\nD. Delta {index}"


def _write_dataset(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _clean_rows(count: int = 4) -> list[dict]:
    return [
        {
            "instruction": _question(index),
            "input": "",
            "output": "A",
            "gold_letter": "A",
            "gold_answer_text": f"Alpha {index}",
            "gold_status": "SOURCE_DERIVED_UNVERIFIED",
            "include_for_evaluation": 1,
            "row_id": index,
        }
        for index in range(count)
    ]


class FakeCaller:
    def __init__(self, specialist_letters: list[str] | None = None, difficulty: str = "moderate"):
        self.calls: list[dict] = []
        self.specialist_letters = list(specialist_letters or ["A", "A"])
        self.difficulty = difficulty
        self.specialist_index = 0

    async def call_json(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        if role == "router":
            parsed = {
                "difficulty": self.difficulty,
                "difficulty_reason": "Requires two complementary domains.",
                "lead_specialty": "Pediatric Dermatology",
                "panel_size": 2,
                "specialists": [
                    {
                        "specialty": "Pediatric Dermatology",
                        "role": "lead",
                        "focus": "lesion morphology",
                        "why_needed": "age-specific skin expertise",
                    },
                    {
                        "specialty": "Pediatric Infectious Disease",
                        "role": "supporting",
                        "focus": "transmission",
                        "why_needed": "distinct infection expertise",
                    },
                ],
                "expected_disagreement": "low",
            }
        elif role == "specialist":
            letter = self.specialist_letters[self.specialist_index]
            self.specialist_index += 1
            parsed = {
                "answer_letter": letter,
                "confidence": 0.85,
                "decisive_evidence": ["specific stem finding"],
                "domain_reasoning": "concise domain reasoning",
                "strongest_alternative": "B" if letter == "A" else "A",
                "why_alternative_is_weaker": "less consistent with the finding",
                "remaining_uncertainty": "none",
            }
        elif role == "critic":
            parsed = {
                "conflict_summary": "Specialists interpreted one clue differently.",
                "decisive_clues": ["specific stem finding"],
                "issues": [],
                "resolution_guidance": "Prioritize the discriminating finding.",
            }
        else:
            parsed = {
                "answer_letter": "A",
                "confidence": 0.9,
                "decisive_evidence": ["specific stem finding"],
                "explanation": "Alpha is best supported.",
                "disagreement_resolution": "none",
            }
        return v2.CallResult(
            role=role,
            model=kwargs["model"],
            parsed=parsed,
            raw_text=json.dumps(parsed),
            latency_seconds=0.01,
            input_tokens=10,
            output_tokens=5,
            attempts=1,
        )


def test_option_parser_supports_common_formats():
    for text in (
        "Question\nA. One\nB. Two",
        "Question\nA) One\nB) Two",
        "Question\nA: One\nB: Two",
        "Question\n(A) One\n(B) Two",
        "Question A. One B. Two",
    ):
        assert parse_options(text) == {"A": "One", "B": "Two"}


def test_direct_gold_letter_precedes_output_and_legacy_parser(tmp_path, monkeypatch):
    calls = []

    def fake_legacy(question, output):
        calls.append((question, output))
        return "C"

    monkeypatch.setattr(v2, "reference_letter", fake_legacy)
    rows = _clean_rows(3)
    rows[0]["gold_letter"] = "B"
    rows[0]["output"] = "Long generated reasoning with Final Answer: C"
    rows[0]["gold_answer_text"] = "Beta 0"
    rows[1]["gold_letter"] = ""
    rows[1]["output"] = "A"
    rows[2]["gold_letter"] = ""
    rows[2]["output"] = "verbose legacy response"
    rows[2]["gold_answer_text"] = "Gamma 2"
    result = v2.load_evaluation_dataset(_write_dataset(tmp_path / "data.csv", rows))
    assert list(result.data["reference_letter"]) == ["B", "A", "C"]
    assert list(result.data["reference_method"]) == [
        "direct_gold_letter",
        "direct_output_letter",
        "legacy_verbose_parser",
    ]
    assert len(calls) == 1


def test_case_and_dataset_identity_ignore_gold_but_label_fingerprint_changes():
    first = v2.assign_case_ids(pd.DataFrame(_clean_rows(3)))
    changed_rows = _clean_rows(3)
    changed_rows[0]["output"] = "B"
    changed_rows[0]["gold_letter"] = "B"
    second = v2.assign_case_ids(pd.DataFrame(changed_rows))
    assert list(first.fixed_case_id) == list(second.fixed_case_id)
    first["reference_letter"] = ["A", "A", "A"]
    second["reference_letter"] = ["B", "A", "A"]
    assert v2.dataset_fingerprint(first) == v2.dataset_fingerprint(second)
    assert v2.label_fingerprint(first) != v2.label_fingerprint(second)


def test_invalid_gold_and_source_exclusions_never_become_eligible(tmp_path):
    rows = _clean_rows(3)
    rows[0]["gold_letter"] = "Z"
    rows[1]["include_for_evaluation"] = 0
    rows[1]["qc_reasons"] = "MANUAL_EXCLUSION"
    result = v2.load_evaluation_dataset(_write_dataset(tmp_path / "data.csv", rows))
    assert len(result.eligible) == 1
    assert len(result.excluded) == 2
    assert not result.audit.loc[0, "valid_gold"]
    assert "source_excluded" in result.audit.loc[1, "exclusion_reason"]


def test_require_original_gold_filters_unverified_rows(tmp_path):
    rows = _clean_rows(2)
    rows[1]["gold_status"] = "ORIGINAL_BENCHMARK_VERIFIED"
    result = v2.load_evaluation_dataset(
        _write_dataset(tmp_path / "data.csv", rows), require_original_gold=True
    )
    assert len(result.eligible) == 1
    assert result.eligible.iloc[0]["gold_status"] == "ORIGINAL_BENCHMARK_VERIFIED"


def test_split_is_reproducible_and_has_no_overlap(tmp_path):
    source = _write_dataset(tmp_path / "data.csv", _clean_rows(10))
    qc = v2.load_evaluation_dataset(source)
    first, manifest_one, path_one = v2.ensure_fixed_split_v2(
        qc, tmp_path, dataset_name="sample", seed=2026
    )
    second, manifest_two, path_two = v2.ensure_fixed_split_v2(
        qc, tmp_path, dataset_name="sample", seed=2026
    )
    assert path_one == path_two
    assert manifest_one["cases"] == manifest_two["cases"]
    assert set(first["development"].fixed_case_id) == set(second["development"].fixed_case_id)
    assert set(first["development"].fixed_case_id).isdisjoint(
        set(first["test"].fixed_case_id)
    )
    assert len(first["development"]) == 2
    assert len(first["test"]) == 8


def test_changed_question_set_gets_new_manifest_without_overwriting_old(tmp_path):
    first_qc = v2.load_evaluation_dataset(
        _write_dataset(tmp_path / "first.csv", _clean_rows(5))
    )
    _, first_manifest, first_path = v2.ensure_fixed_split_v2(
        first_qc, tmp_path, dataset_name="friendly_name", seed=2026
    )
    first_manifest_text = first_path.read_text(encoding="utf-8")

    changed_rows = _clean_rows(5)
    changed_rows[0]["instruction"] = "A genuinely changed question.\nA. One\nB. Two"
    changed_rows[0]["gold_answer_text"] = "One"
    changed_qc = v2.load_evaluation_dataset(
        _write_dataset(tmp_path / "changed.csv", changed_rows)
    )
    changed_splits, changed_manifest, changed_path = v2.ensure_fixed_split_v2(
        changed_qc, tmp_path, dataset_name="friendly_name", seed=2026
    )
    assert changed_path != first_path
    assert changed_qc.dataset_fingerprint[:12] in changed_path.parent.name
    assert first_path.read_text(encoding="utf-8") == first_manifest_text
    assert changed_manifest["redirected_from_manifest"] == str(first_path)
    assert len(changed_splits["development"]) + len(changed_splits["test"]) == 5


def test_mismatched_qc_master_is_reported_not_used(tmp_path):
    evaluation_path = _write_dataset(tmp_path / "evaluation.csv", _clean_rows(2))
    master_rows = _clean_rows(3)
    master = v2.assign_case_ids(pd.DataFrame(master_rows))
    master["include_for_evaluation"] = [1, 0, 1]
    master["qc_reasons"] = ["", "OLD_EXCLUSION", ""]
    master_path = tmp_path / "master.csv"
    master.to_csv(master_path, index=False)
    qc = v2.load_evaluation_dataset(evaluation_path, qc_master_path=master_path)
    assert qc.summary["qc_master_matches_evaluation"] is False
    assert qc.summary["qc_master_excluded_rows"] is None
    assert qc.summary["qc_master_reported_excluded_rows"] == 1


@pytest.mark.asyncio
async def test_excluded_rows_are_rejected_before_model_execution(tmp_path):
    caller = FakeCaller()
    frame = v2.assign_case_ids(pd.DataFrame(_clean_rows(1)))
    frame["reference_letter"] = "A"
    frame["eligible_for_evaluation"] = False
    with pytest.raises(ValueError, match="Excluded QC rows"):
        await v2.run_single_agent_v2(
            frame,
            v2.SingleAgentConfig("single_test", "gpt-4o-mini", max_output_tokens=800),
            "development",
            tmp_path,
            caller=caller,
        )
    assert caller.calls == []


@pytest.mark.asyncio
async def test_safe_resume_rejects_config_mismatch(tmp_path):
    qc = v2.load_evaluation_dataset(_write_dataset(tmp_path / "data.csv", _clean_rows(1)))
    frame = qc.eligible
    caller = FakeCaller()
    first = v2.SingleAgentConfig("resume_test", "gpt-4o-mini", max_output_tokens=800)
    await v2.run_single_agent_v2(
        frame, first, "development", tmp_path, caller=caller, resume=False
    )
    changed = v2.SingleAgentConfig("resume_test", "gpt-4o-mini", max_output_tokens=900)
    with pytest.raises(RuntimeError, match="Resume rejected"):
        await v2.run_single_agent_v2(
            frame, changed, "development", tmp_path, caller=caller, resume=True
        )


def test_router_sanitation_deduplicates_and_uses_single_fallback():
    route = {
        "difficulty": "complex",
        "difficulty_reason": "cross-domain",
        "lead_specialty": "Clinical Dermatology",
        "panel_size": 3,
        "specialists": [
            {"specialty": "Clinical Dermatology", "role": "lead", "focus": "skin", "why_needed": "skin"},
            {"specialty": "General Dermatology", "role": "supporting", "focus": "same", "why_needed": "same"},
            {"specialty": "Clinical Reasoner", "role": "supporting", "focus": "generic", "why_needed": "generic"},
        ],
        "expected_disagreement": "high",
    }
    sanitized, fallback, issues = v2.sanitize_route(route, max_specialists=3)
    assert not fallback
    assert sanitized["panel_size"] == 1
    assert sanitized["specialists"][0]["specialty"] == "Dermatology"
    assert "duplicate_specialty_removed" in issues

    fallback_route, did_fallback, _ = v2.sanitize_route({}, max_specialists=3)
    assert did_fallback
    assert fallback_route["panel_size"] == 1
    assert fallback_route["lead_specialty"] == "Internal Medicine"


def test_critic_is_conditional_not_confidence_threshold_driven():
    route = {"difficulty": "moderate", "expected_disagreement": "low"}
    consensus = [
        {"answer_letter": "A", "confidence": 0.2, "valid_output": True, "remaining_uncertainty": "none"},
        {"answer_letter": "A", "confidence": 0.3, "valid_output": True, "remaining_uncertainty": "none"},
    ]
    assert v2.critic_triggers(route, consensus) == []
    disagreement = [dict(consensus[0]), {**consensus[1], "answer_letter": "B"}]
    assert "specialist_answer_disagreement" in v2.critic_triggers(route, disagreement)
    very_complex = {"difficulty": "very_complex", "expected_disagreement": "low"}
    assert "very_complex_case" in v2.critic_triggers(very_complex, consensus)


@pytest.mark.asyncio
async def test_multi_agent_uses_gpt5_exactly_once_and_has_no_auditor(tmp_path):
    qc = v2.load_evaluation_dataset(_write_dataset(tmp_path / "data.csv", _clean_rows(1)))
    caller = FakeCaller(["A", "A"])
    config = v2.MultiAgentConfig()
    result = await v2.run_multi_agent_v2(
        qc.eligible,
        config,
        "development",
        tmp_path,
        caller=caller,
        resume=False,
    )
    assert result.loc[0, "gpt5_call_count"] == 1
    assert result.loc[0, "correct"]
    assert [call["role"] for call in caller.calls].count("judge") == 1
    assert sum(call["model"].startswith("gpt-5") for call in caller.calls) == 1
    assert "critic" not in [call["role"] for call in caller.calls]
    assert "audit_rejected" not in result.columns
    assert all(call["role"] != "auditor" for call in caller.calls)
    v2.validate_v2_multi_results(result, qc.eligible, config)


@pytest.mark.asyncio
async def test_disagreement_calls_critic_but_still_one_gpt5(tmp_path):
    qc = v2.load_evaluation_dataset(_write_dataset(tmp_path / "data.csv", _clean_rows(1)))
    caller = FakeCaller(["A", "B"])
    result = await v2.run_multi_agent_v2(
        qc.eligible,
        v2.MultiAgentConfig(),
        "development",
        tmp_path,
        caller=caller,
    )
    roles = [call["role"] for call in caller.calls]
    assert roles.count("critic") == 1
    assert roles.count("judge") == 1
    assert result.loc[0, "gpt5_call_count"] == 1


@pytest.mark.asyncio
async def test_very_complex_case_uses_critic_and_still_one_gpt5(tmp_path):
    qc = v2.load_evaluation_dataset(_write_dataset(tmp_path / "data.csv", _clean_rows(1)))
    caller = FakeCaller(["A", "A"], difficulty="very_complex")
    result = await v2.run_multi_agent_v2(
        qc.eligible,
        v2.MultiAgentConfig(),
        "development",
        tmp_path,
        caller=caller,
    )
    roles = [call["role"] for call in caller.calls]
    assert roles.count("critic") == 1
    assert roles.count("judge") == 1
    assert result.loc[0, "gpt5_call_count"] == 1


def test_gpt5_is_forbidden_outside_final_judge():
    with pytest.raises(ValueError, match="restricted"):
        v2.MultiAgentConfig(router_model="gpt-5").validate()


def test_openai_client_configures_platform_trust_store(monkeypatch):
    configured = []
    monkeypatch.setattr(v2, "_configure_platform_trust_store", lambda: configured.append(True))
    v2.OpenAIJSONClient(api_key="test-key")
    assert configured == [True]


def test_exception_chain_message_includes_ssl_root_cause():
    try:
        try:
            raise OSError("certificate verify failed")
        except OSError as exc:
            raise RuntimeError("connection error") from exc
    except RuntimeError as exc:
        message = v2._exception_chain_message(exc)
    assert "RuntimeError: connection error" in message
    assert "OSError: certificate verify failed" in message


def test_high_confidence_error_denominator_and_calibration_exclude_invalid():
    frame = pd.DataFrame(
        [
            {"reference_letter": "A", "predicted_letter": "A", "confidence": 0.9, "correct": True, "invalid_answer": False, "api_error": False, "abstain": False},
            {"reference_letter": "B", "predicted_letter": "A", "confidence": 0.8, "correct": False, "invalid_answer": False, "api_error": False, "abstain": False},
            {"reference_letter": "C", "predicted_letter": "", "confidence": 0.99, "correct": False, "invalid_answer": True, "api_error": False, "abstain": False},
        ]
    )
    metrics = performance_metrics(frame)
    assert metrics["high_confidence_error_numerator"] == 1
    assert metrics["high_confidence_error_denominator"] == 2
    assert metrics["high_confidence_error_rate"] == 0.5
    assert metrics["coverage"] == pytest.approx(2 / 3)


def _prediction_frame(fingerprint: str = "dataset", label_fingerprint: str = "labels") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixed_case_id": "x",
                "reference_letter": "A",
                "predicted_letter": "A",
                "invalid_answer": False,
                "api_error": False,
                "correct": True,
                "dataset_fingerprint": fingerprint,
                "label_fingerprint": label_fingerprint,
                "split": "development",
            }
        ]
    )


def test_comparison_requires_matching_fingerprints_and_pairs_by_id():
    with pytest.raises(ValueError, match="dataset fingerprints"):
        validate_comparison_inputs(
            {"one": _prediction_frame("one"), "two": _prediction_frame("two")}
        )
    stats, paired = paired_accuracy_comparison(
        _prediction_frame(), _prediction_frame(), "candidate", "baseline", n_bootstrap=100
    )
    assert stats["n_paired"] == 1
    assert stats["paired_accuracy_difference"] == 0
    assert list(paired.fixed_case_id) == ["x"]


def test_exact_original_benchmark_match_updates_status_without_guessing(tmp_path):
    evaluation = pd.DataFrame(_clean_rows(1))
    evaluation = v2.assign_case_ids(evaluation)
    evaluation["question"] = "Exact source question"
    source = pd.DataFrame(
        [{"id": "source-1", "question": "Exact source question", "answer": "B"}]
    )
    source_path = tmp_path / "source.csv"
    source.to_csv(source_path, index=False)
    verified, audit = verify_gold_against_originals(
        evaluation,
        [BenchmarkSource(source_path, "MedQA", "test")],
    )
    assert verified.loc[0, "gold_letter"] == "B"
    assert verified.loc[0, "gold_status"] == "ORIGINAL_BENCHMARK_VERIFIED"
    assert bool(audit.loc[0, "verified"])
