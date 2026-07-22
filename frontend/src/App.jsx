import React, { useState, useCallback, useRef } from 'react';
import InputPanel from './components/InputPanel';
import PipelineFlow from './components/PipelineFlow';
import './App.css';

const emptyStages = () => ({
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

function resolveApiUrl() {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  if (typeof window === 'undefined') {
    return 'http://localhost:8000/assess';
  }

  const isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
  if (isLocalHost) {
    return 'http://localhost:8000/assess';
  }

  return '/api/assess';
}

const API_URL = resolveApiUrl();

const specialistKey = (specialty) =>
  `specialist_${String(specialty || 'general').toLowerCase().replace(/\s/g, '_')}`;

function stagesFromRestResult(result) {
  const specialists = {};
  (result.specialist_opinions || []).forEach((opinion) => {
    const specialty = opinion.specialty || opinion.name || 'Specialist';
    specialists[specialistKey(specialty)] = {
      specialty,
      status: 'done',
      summary: '',
      data: opinion,
    };
  });

  return {
    triage: {
      status: 'done',
      summary: '',
      data: result.supervisor,
    },
    specialists,
    specialtiesSpawned: Object.keys(specialists).length > 0,
    leadSpecialist: result.supervisor?.lead_specialist || '',
    consensus: result.consensus || null,
    discussion: {
      message: 'Specialists completed their reviews.',
      opinions_preview: result.specialist_opinions || [],
    },
    judge: {
      status: 'done',
      summary: '',
      data: result.judge_report,
    },
    safety: {
      status: 'done',
      summary: '',
      data: result.safety_report,
    },
    safetyReport: result.safety_report,
    pipelineComplete: true,
    error: null,
  };
}

function App() {
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState(emptyStages);

  const [agentDrafts, setAgentDrafts] = useState({});

  const wsRef = useRef(null);
  const fallbackStartedRef = useRef(false);
  const messageCountRef = useRef(0);
  const completeRef = useRef(false);
  const timersRef = useRef([]);

  const clearTimers = () => {
    timersRef.current.forEach((timer) => clearTimeout(timer));
    timersRef.current = [];
  };

  const handleSubmit = useCallback(({ question, documents }) => {
    setRunning(true);
    fallbackStartedRef.current = false;
    messageCountRef.current = 0;
    completeRef.current = false;
    clearTimers();
    setStages({
      ...emptyStages(),
      triage: {
        status: 'thinking',
        summary: 'Starting secure assessment...',
        data: null,
      },
    });
    setAgentDrafts({});

    const runRestAssessment = async (message) => {
      if (fallbackStartedRef.current) return;
      fallbackStartedRef.current = true;
      clearTimers();

      if (wsRef.current && wsRef.current.readyState < WebSocket.CLOSING) {
        wsRef.current.close();
      }

      setStages((prev) => ({
        ...prev,
        triage: prev.triage || {
          status: 'thinking',
          summary: 'Running complete assessment...',
          data: null,
        },
        error: null,
      }));

      try {
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, documents }),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Request failed with ${response.status}`);
        }

        const result = await response.json();
        setStages(stagesFromRestResult(result));
      } catch (error) {
        console.error(message, error);
        setStages((prev) => ({
          ...prev,
          error:
            error?.message ||
            'The assessment could not complete. Please try again.',
        }));
      } finally {
        setRunning(false);
      }
    };

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    timersRef.current.push(
      setTimeout(() => {
        runRestAssessment('Streaming connection timed out; using REST fallback.');
      }, 5000),
    );

    ws.onopen = () => {
      clearTimers();
      timersRef.current.push(
        setTimeout(() => {
          if (messageCountRef.current === 0) {
            runRestAssessment('Streaming response stalled; using REST fallback.');
          }
        }, 12000),
      );
      ws.send(JSON.stringify({ question, documents }));
    };

    ws.onmessage = (event) => {
      if (fallbackStartedRef.current) return;

      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (error) {
        console.error('Could not parse stream event:', error);
        return;
      }

      messageCountRef.current += 1;
      clearTimers();
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
            completeRef.current = true;
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
        const token = data?.token || '';
        if (!token || !agent_name) return;

        setAgentDrafts((prev) => ({
          ...prev,
          [agent_name]: (prev[agent_name] || '') + token,
        }));
      }
    };

    ws.onclose = () => {
      clearTimers();
      if (!fallbackStartedRef.current && !completeRef.current) {
        runRestAssessment('Streaming closed before completion; using REST fallback.');
        return;
      }
      setRunning(false);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      runRestAssessment('Streaming failed; using REST fallback.');
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
