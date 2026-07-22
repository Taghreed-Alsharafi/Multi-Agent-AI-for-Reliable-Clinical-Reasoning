import React from 'react';

export default function FinalReport({ safetyReport }) {
    if (!safetyReport) return null;

    const {
        verified_findings = [],
        flagged_issues = [],
        final_summary = '',
        overall_confidence = 0,
        is_safe = false,
    } = safetyReport;

    return (
        <div className="final-report">
            <h3>
                🛡️ Safety-Verified Final Report
                <span className={`report-badge ${is_safe ? 'safe' : 'unsafe'}`}>
                    {is_safe ? '✓ Verified Safe' : '⚠ Issues Found'}
                </span>
            </h3>

            <div className="report-summary">{final_summary}</div>

            {verified_findings.length > 0 && (
                <div className="verified-findings">
                    <h4>✅ Verified Findings</h4>
                    {verified_findings.map((finding, i) => (
                        <div key={i} className="verified-item">
                            <span className="check-icon">✓</span>
                            <span>{finding}</span>
                        </div>
                    ))}
                </div>
            )}

            {flagged_issues.length > 0 && (
                <div className="flagged-issues">
                    <h4>⚠️ Flagged Issues</h4>
                    {flagged_issues.map((issue, i) => (
                        <div key={i} className="flagged-item">
                            <div className="flag-specialist">{issue.specialist}</div>
                            <div className="flag-claim">Claim: "{issue.claim}"</div>
                            <div className="flag-issue">Issue: {issue.issue}</div>
                        </div>
                    ))}
                </div>
            )}

            <div className="confidence-bar" style={{ marginTop: 20 }}>
                <div className="bar-label" style={{ marginBottom: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Overall Verification Confidence
                </div>
                <div className="bar-track">
                    <div
                        className="bar-fill"
                        style={{
                            width: `${overall_confidence * 100}%`,
                            background: is_safe ? 'var(--safety-color)' : 'var(--error-color)'
                        }}
                    />
                </div>
                <div className="bar-label" style={{ color: is_safe ? 'var(--safety-color)' : 'var(--error-color)', fontWeight: 700 }}>
                    {(overall_confidence * 100).toFixed(0)}%
                </div>
            </div>
        </div>
    );
}
