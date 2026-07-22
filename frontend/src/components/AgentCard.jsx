import React from 'react';

const AGENT_ICONS = {
    triage: '🔬',
    specialist: '🧠',
    safety: '🛡️',
    discussion: '💬',
    orchestrator: '⚙️',
};

const SPECIALTY_ICONS = {
    'Cardiology': '❤️',
    'Endocrinology': '🧬',
    'Nephrology': '🫘',
    'Neurology': '🧠',
    'Pharmacology': '💊',
    'Radiology': '📡',
    'Internal Medicine': '🩺',
    'Pulmonology': '🫁',
    'Oncology': '🎗️',
    'Gastroenterology': '🔬',
    'Hematology': '🩸',
    'Rheumatology': '🦴',
    'Dermatology': '🧴',
    'Psychiatry': '🧠',
    'Ophthalmology': '👁️',
};

function getIcon(type, specialty) {
    if (type === 'specialist' && specialty) {
        return SPECIALTY_ICONS[specialty] || '🧠';
    }
    return AGENT_ICONS[type] || '🤖';
}

export default function AgentCard({ type, name, specialty, status, summary, data, isLead, draft }) {
    const icon = getIcon(type, specialty);
    const statusClass = status === 'thinking' ? 'thinking' : status === 'done' ? 'done' : '';

    return (
        <div className={`agent-card ${type} ${statusClass} ${isLead ? 'lead-agent' : ''}`}>
            {isLead && <div className="lead-badge">Lead Specialist</div>}
            <div className="agent-card-header">
                <div className="agent-avatar">{icon}</div>
                <div className="agent-info">
                    <div className="agent-name">{name}</div>
                    <div className="agent-role">{specialty || type}</div>
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
                        <span>{summary || 'Processing...'}</span>

                        {draft && (
                            <div className="agent-draft-stream">
                                <div className="draft-label">Drafting Clinical Report:</div>
                                <div className="draft-text">{draft}</div>
                            </div>
                        )}
                    </div>
                )}

                {status === 'done' && data && (
                    <div className="agent-summary">
                        {/* Supervisor (Triage) done */}
                        {type === 'triage' && data.specialties && (
                            <>
                                <p><strong>Specialist Team:</strong></p>
                                <div className="evidence-list">
                                    {data.specialties.map((s, i) => (
                                        <div key={i} className={`evidence-quote ${s.name === data.lead_specialist ? 'highlight-lead' : ''}`}>
                                            {SPECIALTY_ICONS[s.name] || '🧠'} <strong>{s.name}</strong>
                                            {s.name === data.lead_specialist && <span className="lead-tag"> (Lead)</span>}
                                            <div className="reason-text">{s.reason}</div>
                                        </div>
                                    ))}
                                </div>
                                {data.rationale && <p style={{ marginTop: 10 }}>{data.rationale}</p>}
                            </>
                        )}

                        {/* Specialist done */}
                        {type === 'specialist' && (
                            <>
                                {data.findings && <p><strong>Findings:</strong> {data.findings}</p>}
                                {data.recommendation && <p style={{ marginTop: 8 }}><strong>Recommendation:</strong> {data.recommendation}</p>}
                                {data.evidence_quotes && data.evidence_quotes.length > 0 && (
                                    <div className="evidence-list">
                                        {data.evidence_quotes.map((q, i) => (
                                            <div key={i} className="evidence-quote">"{q}"</div>
                                        ))}
                                    </div>
                                )}
                                {typeof data.confidence === 'number' && (
                                    <div className="confidence-bar">
                                        <div className="bar-track">
                                            <div className="bar-fill" style={{ width: `${data.confidence * 100}%` }} />
                                        </div>
                                        <div className="bar-label">{(data.confidence * 100).toFixed(0)}%</div>
                                    </div>
                                )}
                            </>
                        )}

                        {/* Judge done */}
                        {type === 'judge' && (
                            <>
                                {data.final_summary && <p><strong>Summary:</strong> {data.final_summary}</p>}
                                {data.consolidated_recommendations && (
                                    <div className="recommendations-list">
                                        <strong>Recommendations:</strong>
                                        <ul>
                                            {data.consolidated_recommendations.map((r, i) => (
                                                <li key={i}>{r}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </>
                        )}

                        {/* Discussion summary (transitional) */}
                        {type === 'discussion' && (
                            <>
                                <p>{data.message}</p>
                                {data.opinions_preview && (
                                    <div className="discussion-opinions">
                                        {data.opinions_preview.map((op, i) => (
                                            <div key={i} className="opinion-chip">
                                                {SPECIALTY_ICONS[op.specialty] || '🧠'} {op.specialty}
                                                <span className="chip-confidence">{(op.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </>
                        )}

                        {/* Safety done */}
                        {type === 'safety' && (
                            <p>{data.message || 'Verifying claims against documents...'}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
