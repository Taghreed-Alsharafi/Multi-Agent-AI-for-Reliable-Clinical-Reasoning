"""Tests for confidence-based agreement scoring."""

from __future__ import annotations

from orchestrator.consensus import compute_consensus


def _opinion(specialty: str, confidence):
    return {"specialty": specialty, "confidence": confidence}


def test_aligned_panel_scores_strong_agreement():
    """Close, high confidences mean the panel agrees."""
    report = compute_consensus(
        [_opinion("Cardiology", 0.9), _opinion("Nephrology", 0.85)]
    )

    assert report.participating == 2
    assert report.mean_confidence == 0.875
    assert report.level == "strong"
    assert report.agreement_score > 0.8


def test_split_panel_is_penalised_despite_decent_mean():
    """A 0.1 / 0.9 split averages 0.5 but is not consensus."""
    report = compute_consensus(
        [_opinion("Cardiology", 0.1), _opinion("Neurology", 0.9)]
    )

    assert report.mean_confidence == 0.5
    # stdev 0.4 of a possible 0.5 discounts 80% of the mean away.
    assert report.confidence_spread == 0.4
    assert report.agreement_score == 0.1
    assert report.level == "none"

    # ...whereas the same mean from an aligned panel scores its full value.
    aligned = compute_consensus(
        [_opinion("Cardiology", 0.5), _opinion("Neurology", 0.5)]
    )
    assert aligned.mean_confidence == 0.5
    assert aligned.agreement_score == 0.5
    assert aligned.level == "moderate"


def test_single_specialist_scores_its_own_confidence():
    """With no one to disagree with, agreement is just the confidence."""
    report = compute_consensus([_opinion("Cardiology", 0.7)])

    assert report.confidence_spread == 0.0
    assert report.agreement_score == 0.7
    assert report.level == "moderate"


def test_zero_confidence_counts_as_abstention():
    """A specialist with nothing relevant to say should not drag the mean."""
    report = compute_consensus(
        [_opinion("Cardiology", 0.8), _opinion("Dermatology", 0.0)]
    )

    assert report.participating == 1
    assert report.abstained == 1
    assert report.mean_confidence == 0.8
    assert report.per_specialty["Dermatology"] == 0.0


def test_all_abstaining_yields_no_agreement():
    report = compute_consensus(
        [_opinion("Cardiology", 0.0), _opinion("Dermatology", 0.0)]
    )

    assert report.participating == 0
    assert report.abstained == 2
    assert report.agreement_score == 0.0
    assert report.level == "none"


def test_missing_or_malformed_confidence_is_treated_as_zero():
    """Agents sometimes return junk – scoring must not crash."""
    report = compute_consensus(
        [
            {"specialty": "Cardiology"},
            _opinion("Neurology", "not a number"),
            _opinion("Nephrology", 0.6),
        ]
    )

    assert report.abstained == 2
    assert report.mean_confidence == 0.6


def test_confidence_is_clamped_to_unit_range():
    """An agent claiming 1.5 confidence gets clamped, not trusted."""
    report = compute_consensus(
        [_opinion("Cardiology", 1.5), _opinion("Nephrology", 1.0)]
    )

    assert report.per_specialty["Cardiology"] == 1.0
    assert report.mean_confidence == 1.0
    assert report.agreement_score == 1.0


def test_outlier_specialist_is_surfaced():
    """The specialist standing apart from the panel is named."""
    report = compute_consensus(
        [
            _opinion("Cardiology", 0.9),
            _opinion("Nephrology", 0.85),
            _opinion("Neurology", 0.2),
        ]
    )

    assert "Neurology" in report.outliers
    assert "Cardiology" not in report.outliers


def test_tightly_clustered_panel_has_no_outliers():
    """A few points off an 88% mean is not disagreement — don't flag it."""
    report = compute_consensus(
        [
            _opinion("Nephrology", 0.95),
            _opinion("Pharmacy", 0.85),
            _opinion("Endocrinology", 0.85),
        ]
    )

    assert report.level == "strong"
    assert report.outliers == []


def test_empty_panel_does_not_crash():
    report = compute_consensus([])

    assert report.participating == 0
    assert report.agreement_score == 0.0
    assert report.per_specialty == {}
