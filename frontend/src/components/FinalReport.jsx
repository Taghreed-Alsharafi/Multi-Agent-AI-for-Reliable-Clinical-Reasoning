import React from 'react';
import { asArray, boundedPercent, displayText } from '../utils/format';

export default function FinalReport({ safetyReport }) {
    if (!safetyReport) return null;

    const {
        verified_findings = [],
        flagged_issues = [],
        final_summary = '',
        overall_confidence = 0,
        is_safe = false,
    } = safetyReport;
    const confidence = boundedPercent(overall_confidence);

    return (
        <div className="final-report">
            <h3>
                Safety-Verified Final Report
                <span className={`report-badge ${is_safe ? 'safe' : 'unsafe'}`}>
                    {is_safe ? 'Verified Safe' : 'Issues Found'}
                </span>
            </h3>

            <div className="report-summary">{displayText(final_summary, 'No final summary was returned.')}</div>

            {asArray(verified_findings).length > 0 && (
                <div className="verified-findings">
                    <h4>Verified Findings</h4>
                    {asArray(verified_findings).map((finding, i) => (
                        <div key={i} className="verified-item">
                            <span className="check-icon">✓</span>
                            <span>{displayText(finding, 'Verified finding')}</span>
                        </div>
                    ))}
                </div>
            )}

            {asArray(flagged_issues).length > 0 && (
                <div className="flagged-issues">
                    <h4>Flagged Issues</h4>
                    {asArray(flagged_issues).map((issue, i) => (
                        <div key={i} className="flagged-item">
                            <div className="flag-specialist">{displayText(issue?.specialist, 'Safety review')}</div>
                            <div className="flag-claim">Claim: "{displayText(issue?.claim || issue)}"</div>
                            <div className="flag-issue">Issue: {displayText(issue?.issue, 'Needs review')}</div>
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
                            width: `${confidence}%`,
                            background: is_safe ? 'var(--safety-color)' : 'var(--error-color)',
                        }}
                    />
                </div>
                <div className="bar-label" style={{ color: is_safe ? 'var(--safety-color)' : 'var(--error-color)', fontWeight: 700 }}>
                    {confidence.toFixed(0)}%
                </div>
            </div>
        </div>
    );
}
