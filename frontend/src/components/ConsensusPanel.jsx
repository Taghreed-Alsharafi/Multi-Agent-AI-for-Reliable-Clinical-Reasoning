import React from 'react';

const LEVEL_COLORS = {
    strong: 'var(--safety-color)',
    moderate: 'var(--judge-color)',
    weak: 'var(--judge-color)',
    none: 'var(--error-color)',
};

const LEVEL_LABELS = {
    strong: 'Strong agreement',
    moderate: 'Moderate agreement',
    weak: 'Weak agreement',
    none: 'No agreement',
};

const pct = (n) => `${Math.round((n || 0) * 100)}%`;

export default function ConsensusPanel({ consensus }) {
    if (!consensus) return null;

    const {
        agreement_score = 0,
        level = 'none',
        mean_confidence = 0,
        confidence_spread = 0,
        participating = 0,
        abstained = 0,
        per_specialty = {},
        outliers = [],
    } = consensus;

    const color = LEVEL_COLORS[level] || 'var(--text-secondary)';
    const ranked = Object.entries(per_specialty).sort((a, b) => b[1] - a[1]);

    return (
        <div className="consensus-panel">
            <div className="consensus-head">
                <h3>🤝 Panel Agreement</h3>
                <span className="consensus-badge" style={{ color, borderColor: color }}>
                    {LEVEL_LABELS[level] || level}
                </span>
            </div>

            <div className="consensus-score" style={{ color }}>
                {pct(agreement_score)}
            </div>
            <div className="bar-track">
                <div
                    className="bar-fill"
                    style={{ width: pct(agreement_score), background: color }}
                />
            </div>

            <div className="consensus-stats">
                <div className="consensus-stat">
                    <span className="stat-value">{pct(mean_confidence)}</span>
                    <span className="stat-label">Mean confidence</span>
                </div>
                <div className="consensus-stat">
                    <span className="stat-value">±{pct(confidence_spread)}</span>
                    <span className="stat-label">Spread</span>
                </div>
                <div className="consensus-stat">
                    <span className="stat-value">{participating}</span>
                    <span className="stat-label">Contributing</span>
                </div>
                <div className="consensus-stat">
                    <span className="stat-value">{abstained}</span>
                    <span className="stat-label">Abstained</span>
                </div>
            </div>

            <div className="consensus-breakdown">
                {ranked.map(([specialty, confidence]) => {
                    const isOutlier = outliers.includes(specialty);
                    const abstaining = confidence === 0;
                    return (
                        <div key={specialty} className="consensus-row">
                            <span className="consensus-specialty">
                                {specialty}
                                {isOutlier && <span className="outlier-tag">outlier</span>}
                                {abstaining && <span className="abstain-tag">abstained</span>}
                            </span>
                            <div className="bar-track slim">
                                <div
                                    className="bar-fill"
                                    style={{
                                        width: pct(confidence),
                                        background: isOutlier ? 'var(--error-color)' : color,
                                        opacity: abstaining ? 0.25 : 1,
                                    }}
                                />
                            </div>
                            <span className="consensus-value">{pct(confidence)}</span>
                        </div>
                    );
                })}
            </div>

            <p className="consensus-note">
                Agreement is the mean specialist confidence, discounted by how far
                the individual scores spread apart. Abstaining specialists are excluded.
            </p>
        </div>
    );
}
