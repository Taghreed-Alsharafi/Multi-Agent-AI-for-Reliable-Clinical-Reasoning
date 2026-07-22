import React from 'react';
import { asArray, boundedPercent, displayText } from '../utils/format';

const AGENT_LABELS = {
    triage: 'Supervisor',
    specialist: 'Specialist',
    safety: 'Safety',
    discussion: 'Discussion',
    judge: 'Judge',
    orchestrator: 'Orchestrator',
};

const SPECIALTY_ICONS = {};

function getIcon(type) {
    return AGENT_LABELS[type]?.slice(0, 1) || 'A';
}

export default function AgentCard({ type, name, specialty, status, summary, data, isLead, draft }) {
    const icon = getIcon(type);
    const statusClass = status === 'thinking' ? 'thinking' : status === 'done' ? 'done' : '';

    return (
        <div className={`agent-card ${type} ${statusClass} ${isLead ? 'lead-agent' : ''}`}>
            {isLead && <div className="lead-badge">Lead Specialist</div>}
            <div className="agent-card-header">
                <div className="agent-avatar">{icon}</div>
                <div className="agent-info">
                    <div className="agent-name">{displayText(name, 'Agent')}</div>
                    <div className="agent-role">{displayText(specialty || type, 'Agent')}</div>
                </div>
                <div className={`agent-status-badge ${statusClass}`}>
                    {status === 'thinking' ? 'Analyzing' : status === 'done' ? 'Complete' : 'Idle'}
                </div>
            </div>

            <div className="agent-card-body">
                {status === 'thinking' && (
                    <div className="thinking-indicator">
                        <div className="ekg-loader">
                            <svg viewBox="0 0 100 30" preserveAspectRatio="none">
                                <polyline points="0,15 20,15 30,5 45,25 55,15 100,15" />
                            </svg>
                        </div>
                        <span>{displayText(summary, 'Processing...')}</span>

                        {draft && (
                            <div className="agent-draft-stream">
                                <div className="draft-label">Drafting Clinical Report:</div>
                                <div className="draft-text">{displayText(draft)}</div>
                            </div>
                        )}
                    </div>
                )}

                {status === 'done' && data && (
                    <div className="agent-summary">
                        {type === 'triage' && data.specialties && (
                            <>
                                <p><strong>Specialist Team:</strong></p>
                                <div className="evidence-list">
                                    {asArray(data.specialties).map((s, i) => (
                                        <div key={i} className={`evidence-quote ${s?.name === data.lead_specialist ? 'highlight-lead' : ''}`}>
                                            <strong>{displayText(s?.name, 'Specialist')}</strong>
                                            {s?.name === data.lead_specialist && <span className="lead-tag"> (Lead)</span>}
                                            <div className="reason-text">{displayText(s?.reason)}</div>
                                        </div>
                                    ))}
                                </div>
                                {data.rationale && <p style={{ marginTop: 10 }}>{displayText(data.rationale)}</p>}
                            </>
                        )}

                        {type === 'specialist' && (
                            <>
                                {data.findings && <p><strong>Findings:</strong> {displayText(data.findings)}</p>}
                                {data.recommendation && <p style={{ marginTop: 8 }}><strong>Recommendation:</strong> {displayText(data.recommendation)}</p>}
                                {asArray(data.evidence_quotes).length > 0 && (
                                    <div className="evidence-list">
                                        {asArray(data.evidence_quotes).map((q, i) => (
                                            <div key={i} className="evidence-quote">"{displayText(q)}"</div>
                                        ))}
                                    </div>
                                )}
                                {data.confidence !== undefined && (
                                    <div className="confidence-bar">
                                        <div className="bar-track">
                                            <div className="bar-fill" style={{ width: `${boundedPercent(data.confidence)}%` }} />
                                        </div>
                                        <div className="bar-label">{boundedPercent(data.confidence).toFixed(0)}%</div>
                                    </div>
                                )}
                            </>
                        )}

                        {type === 'judge' && (
                            <>
                                {data.final_summary && <p><strong>Summary:</strong> {displayText(data.final_summary)}</p>}
                                {asArray(data.consolidated_recommendations).length > 0 && (
                                    <div className="recommendations-list">
                                        <strong>Recommendations:</strong>
                                        <ul>
                                            {asArray(data.consolidated_recommendations).map((r, i) => (
                                                <li key={i}>{displayText(r)}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </>
                        )}

                        {type === 'discussion' && (
                            <>
                                <p>{displayText(data.message)}</p>
                                {asArray(data.opinions_preview).length > 0 && (
                                    <div className="discussion-opinions">
                                        {asArray(data.opinions_preview).map((op, i) => (
                                            <div key={i} className="opinion-chip">
                                                {SPECIALTY_ICONS[op?.specialty] || 'Specialist'} {displayText(op?.specialty, 'Specialist')}
                                                <span className="chip-confidence">{boundedPercent(op?.confidence).toFixed(0)}%</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </>
                        )}

                        {type === 'safety' && (
                            <p>{displayText(data.message, 'Verifying claims against documents...')}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
