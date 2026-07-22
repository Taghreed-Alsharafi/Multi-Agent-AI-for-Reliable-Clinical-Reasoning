import React from 'react';
import AgentCard from './AgentCard';
import ConnectionLine from './ConnectionLine';
import ConsensusPanel from './ConsensusPanel';
import FinalReport from './FinalReport';

export default function PipelineFlow({ stages, agentDrafts = {} }) {
    const {
        triage,
        specialists,
        specialtiesSpawned,
        leadSpecialist,
        consensus,
        discussion,
        judge,
        safety,
        safetyReport,
        pipelineComplete,
    } = stages;

    // Nothing to show yet
    if (!triage) return null;

    return (
        <div className="pipeline-flow">
            <div className="medical-grid-bg" />

            {/* ── Stage 1: Supervisor (Triage) ──────────────────── */}
            <div className="stage-section">
                <div className={`stage-label ${triage ? 'active' : ''}`}>
                    <span className="stage-number">1</span>
                    Stage 1 — Clinical Supervisor
                </div>
                <AgentCard
                    type="triage"
                    name="Supervisor Agent"
                    specialty="Case Coordination"
                    status={triage.status}
                    summary={triage.summary}
                    data={triage.data}
                    draft={agentDrafts["supervisor_agent"]}
                />
            </div>

            {/* Connection: Triage → Specialists */}
            {triage.status === 'done' && (
                <ConnectionLine
                    active={!!specialtiesSpawned}
                    fromColor="var(--triage-color)"
                    toColor="var(--specialist-color)"
                />
            )}

            {/* ── Stage 2: Specialist Swarm ───────────────────── */}
            {specialtiesSpawned && (
                <div className="stage-section">
                    <div className="stage-label active">
                        <span className="stage-number" style={{ borderColor: 'var(--specialist-color)', color: 'var(--specialist-color)' }}>2</span>
                        Stage 2 — Specialist Swarm ({Object.keys(specialists).length} agents)
                    </div>
                    <div className="specialist-grid">
                        {Object.entries(specialists).map(([agentName, agent]) => (
                            <AgentCard
                                key={agentName}
                                type="specialist"
                                name={agent.specialty}
                                specialty={agent.specialty}
                                status={agent.status}
                                summary={agent.summary}
                                data={agent.data}
                                isLead={agent.specialty === leadSpecialist}
                                draft={agentDrafts[agentName]}
                            />
                        ))}
                    </div>
                </div>
            )}

            {/* ── Agreement across the swarm ──────────────────── */}
            {consensus && (
                <>
                    <ConnectionLine
                        active={true}
                        fromColor="var(--specialist-color)"
                        toColor="var(--judge-color)"
                    />
                    <ConsensusPanel consensus={consensus} />
                </>
            )}

            {/* Connection: Specialists → Discussion */}
            {discussion && (
                <ConnectionLine
                    active={true}
                    fromColor="var(--specialist-color)"
                    toColor="var(--discussion-color)"
                />
            )}

            {/* ── Stage 3: Consolidation (Judge) ────────────────── */}
            {discussion && (
                <div className="stage-section">
                    <div className="stage-label active">
                        <span className="stage-number" style={{ borderColor: 'var(--discussion-color)', color: 'var(--discussion-color)' }}>3</span>
                        Stage 3 — Medical Judge (Consolidation)
                    </div>
                    <div className="discussion-card">
                        <AgentCard
                            type="judge"
                            name="Medical Judge"
                            specialty="Expert Consensus"
                            status={judge?.status || 'idle'}
                            summary={judge?.summary}
                            data={judge?.data}
                            isLead={false}
                            draft={agentDrafts["judge_agent"]}
                        />
                    </div>
                </div>
            )}

            {/* Connection: Judge → Safety */}
            {safety && (
                <ConnectionLine
                    active={true}
                    fromColor="var(--discussion-color)"
                    toColor="var(--safety-color)"
                />
            )}

            {/* ── Stage 4: Safety Verification ─────────────────── */}
            {safety && (
                <div className="stage-section">
                    <div className="stage-label active">
                        <span className="stage-number" style={{ borderColor: 'var(--safety-color)', color: 'var(--safety-color)' }}>4</span>
                        Stage 4 — Safety Verification
                    </div>
                    <AgentCard
                        type="safety"
                        name="Safety Agent"
                        specialty="Document Grounding"
                        status={safety.status}
                        summary={safety.summary}
                        data={safety.data}
                        draft={agentDrafts["safety_agent"]}
                    />
                </div>
            )}

            {/* ── Final Report ────────────────────────────────── */}
            {pipelineComplete && safetyReport && (
                <>
                    <ConnectionLine
                        active={true}
                        fromColor="var(--safety-color)"
                        toColor="var(--safety-color)"
                    />
                    <FinalReport safetyReport={safetyReport} />
                </>
            )}
        </div>
    );
}
