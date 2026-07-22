import React, { useState, useCallback, useRef } from 'react';
import InputPanel from './components/InputPanel';
import PipelineFlow from './components/PipelineFlow';
import './App.css';

function resolveWebSocketUrl() {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }

  if (typeof window === 'undefined') {
    return 'ws://localhost:8000/ws/assess';
  }

  const isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
  if (isLocalHost) {
    return 'ws://localhost:8000/ws/assess';
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/ws/assess`;
}

const WS_URL = resolveWebSocketUrl();

function App() {
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState({
    triage: null,
    specialists: {},
    specialtiesSpawned: false,
    leadSpecialist: '',
    consensus: null,
    discussion: null,
    judge: null,
    safety: null,
    safetyReport: null,
    pipelineComplete: false,
    error: null,
  });

  const [agentDrafts, setAgentDrafts] = useState({}); // New state for streaming tokens

  const wsRef = useRef(null);

  const resetStages = () => ({
    triage: null,
    specialists: {},
    specialtiesSpawned: false,
    leadSpecialist: '',
    consensus: null,
    discussion: null,
    judge: null,
    safety: null,
    safetyReport: null,
    pipelineComplete: false,
    error: null,
  });

  const handleSubmit = useCallback(({ question, documents }) => {
    setRunning(true);
    setStages(resetStages());
    setAgentDrafts({}); // Clear previous drafts

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ question, documents }));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const { type, agent_name, data } = msg;

      setStages((prev) => {
        const next = { ...prev };

        switch (type) {
          case 'triage_thinking':
            next.triage = {
              status: 'thinking',
              summary: data?.message || 'Analyzing...',
              data: null,
            };
            break;

          case 'triage_done':
            next.triage = {
              status: 'done',
              summary: '',
              data: data,
            };
            next.leadSpecialist = data.lead_specialist;
            break;

          case 'specialists_spawned': {
            next.specialtiesSpawned = true;
            next.leadSpecialist = data.lead_specialist || next.leadSpecialist;
            // Initialize specialist cards
            const newSpecs = {};
            (data?.specialties || []).forEach((s) => {
              const key = `specialist_${s.toLowerCase().replace(/\s/g, '_')}`;
              newSpecs[key] = {
                specialty: s,
                status: 'idle',
                summary: '',
                data: null,
              };
            });
            next.specialists = newSpecs;
            break;
          }

          case 'specialist_thinking':
            if (agent_name && next.specialists[agent_name]) {
              next.specialists = {
                ...next.specialists,
                [agent_name]: {
                  ...next.specialists[agent_name],
                  status: 'thinking',
                  summary: data?.message || 'Analyzing...',
                },
              };
            }
            break;

          case 'specialist_done':
            if (agent_name && next.specialists[agent_name]) {
              next.specialists = {
                ...next.specialists,
                [agent_name]: {
                  ...next.specialists[agent_name],
                  status: 'done',
                  data: data,
                },
              };
            }
            break;

          case 'consensus_done':
            next.consensus = data;
            break;

          case 'discussion_summary':
            next.discussion = data;
            break;

          case 'judge_thinking':
            next.judge = {
              status: 'thinking',
              summary: data?.message || 'Synthesizing...',
              data: null,
            };
            break;

          case 'judge_done':
            next.judge = {
              status: 'done',
              summary: '',
              data: data,
            };
            break;

          case 'safety_thinking':
            next.safety = {
              status: 'thinking',
              summary: data?.message || 'Verifying...',
              data: null,
            };
            break;

          case 'safety_done':
            next.safety = {
              status: 'done',
              summary: '',
              data: data,
            };
            next.safetyReport = data;
            break;

          case 'pipeline_complete':
            next.pipelineComplete = true;
            if (data?.safety_report) {
              next.safetyReport = data.safety_report;
            }
            if (data?.consensus) {
              next.consensus = data.consensus;
            }
            break;

          case 'error':
            console.error('Pipeline error:', data);
            next.error =
              data?.message || 'The pipeline stopped unexpectedly.';
            break;

          default:
            break;
        }

        return next;
      });

      if (type === 'agent_stream') {
        setAgentDrafts((prev) => ({
          ...prev,
          [agent_name]: (prev[agent_name] || '') + data.token,
        }));
      }
    };

    ws.onclose = () => {
      setRunning(false);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      // Fires when the backend is unreachable, which is by far the most
      // common cause - say so rather than leaving a silent dead pipeline.
      setStages((prev) => ({
        ...prev,
        error:
          prev.error ||
          `Could not reach the backend at ${WS_URL}. Start it with ` +
            `start.bat (or "uvicorn api.main:app --reload") and try again.`,
      }));
      setRunning(false);
    };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Multi-Agent Medical Assessment</h1>
        <p>Dynamic AI specialist swarm with real-time safety verification</p>
      </header>

      <InputPanel onSubmit={handleSubmit} disabled={running} />

      {stages.error && (
        <div className="error-banner" role="alert">
          <span className="error-icon">⚠️</span>
          <div>
            <div className="error-title">Assessment stopped</div>
            <div className="error-message">{stages.error}</div>
          </div>
        </div>
      )}

      {!stages.triage && !running && !stages.error && (
        <div className="empty-state">
          <div className="empty-icon">🧬</div>
          <p>Submit a clinical question to activate the agent pipeline</p>
        </div>
      )}

      <PipelineFlow
        stages={stages}
        agentDrafts={agentDrafts}
      />
    </div>
  );
}

export default App;
