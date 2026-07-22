"""Inter-agent agreement scoring.

Each specialist reports a ``confidence`` in 0-1.  The swarm's agreement is
derived from those numbers: a high *average* confidence only counts as
consensus when the specialists are also *close to each other*, so the mean is
discounted by how far the individual scores spread apart.

    agreement = mean_confidence * (1 - dispersion)
    dispersion = stdev(confidences) / 0.5

0.5 is the largest standard deviation reachable by values bounded to 0-1
(half the panel at 0, half at 1), so ``dispersion`` lands in 0-1 and a fully
split panel scores 0 no matter how confident its members are.
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel

#: Largest stdev possible for values in [0, 1] — used to normalise dispersion.
MAX_STDEV = 0.5

#: A specialist must sit at least this far from the mean to count as an outlier.
#: Without a floor, a tightly-clustered panel (spread ±0.05) flags anyone a few
#: points off the mean, which reads as disagreement where there is none.
MIN_OUTLIER_DELTA = 0.15

#: Lower bound of each agreement label, highest first.
AGREEMENT_LEVELS = [
    (0.75, "strong"),
    (0.50, "moderate"),
    (0.25, "weak"),
    (0.00, "none"),
]


class ConsensusReport(BaseModel):
    """Agreement across the specialist swarm."""

    agreement_score: float  # 0-1, mean confidence discounted by spread
    level: str  # strong | moderate | weak | none
    mean_confidence: float  # 0-1, average across participating specialists
    confidence_spread: float  # stdev of participating confidences
    participating: int  # specialists that gave an opinion
    abstained: int  # specialists with nothing relevant to add
    per_specialty: dict[str, float]  # specialty -> reported confidence
    outliers: list[str]  # specialties furthest from the mean


def _confidence(opinion: dict[str, Any]) -> float:
    """Read a specialist's confidence, clamped to 0-1."""
    try:
        value = float(opinion.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _label(score: float) -> str:
    """Map an agreement score onto a human-readable level."""
    for threshold, level in AGREEMENT_LEVELS:
        if score >= threshold:
            return level
    return "none"


def compute_consensus(opinions: list[dict[str, Any]]) -> ConsensusReport:
    """Score how much the specialist swarm agrees.

    Specialists that reported zero confidence are treated as abstentions and
    excluded from the average — they found nothing relevant in their domain,
    which is not the same as disagreeing.
    """
    scored = [
        (opinion.get("specialty", "Unknown"), _confidence(opinion))
        for opinion in opinions
    ]
    per_specialty = dict(scored)
    participating = [(name, c) for name, c in scored if c > 0.0]

    if not participating:
        return ConsensusReport(
            agreement_score=0.0,
            level="none",
            mean_confidence=0.0,
            confidence_spread=0.0,
            participating=0,
            abstained=len(scored),
            per_specialty=per_specialty,
            outliers=[],
        )

    confidences = [c for _, c in participating]
    mean = statistics.fmean(confidences)
    # pstdev over the panel we actually have — not a sample of a larger one.
    spread = statistics.pstdev(confidences) if len(confidences) > 1 else 0.0
    dispersion = min(spread / MAX_STDEV, 1.0)
    agreement = mean * (1.0 - dispersion)

    # Anyone more than one stdev from the mean — but never closer than
    # MIN_OUTLIER_DELTA — is worth surfacing to the user.
    cutoff = max(spread, MIN_OUTLIER_DELTA)
    outliers = [name for name, c in participating if abs(c - mean) > cutoff]

    return ConsensusReport(
        agreement_score=round(agreement, 3),
        level=_label(agreement),
        mean_confidence=round(mean, 3),
        confidence_spread=round(spread, 3),
        participating=len(participating),
        abstained=len(scored) - len(participating),
        per_specialty=per_specialty,
        outliers=outliers,
    )
