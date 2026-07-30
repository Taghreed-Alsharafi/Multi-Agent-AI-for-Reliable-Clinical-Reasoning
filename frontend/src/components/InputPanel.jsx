import React, { useState } from 'react';

export default function InputPanel({ onSubmit, disabled }) {
    const [question, setQuestion] = useState('');
    const [documents, setDocuments] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!question.trim() || !documents.trim()) return;

        // Split documents by double newline or treat as single document
        const docs = documents
            .split(/\n{2,}/)
            .map(d => d.trim())
            .filter(Boolean);

        onSubmit({ question: question.trim(), documents: docs });
    };

    return (
        <form className="input-panel" onSubmit={handleSubmit}>
            <h2>Start a research assessment</h2>

            <div className="input-group">
                <label htmlFor="question">Question for the agent team</label>
                <input
                    id="question"
                    type="text"
                    placeholder="Example: Which findings require the closest follow-up?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={disabled}
                />
            </div>

            <div className="input-group">
                <label htmlFor="documents">De-identified case information</label>
                <textarea
                    id="documents"
                    placeholder={"Paste fictional or de-identified notes here.\n\nSeparate multiple documents with a blank line."}
                    value={documents}
                    onChange={(e) => setDocuments(e.target.value)}
                    disabled={disabled}
                    rows={6}
                />
            </div>

            <button type="submit" className="submit-btn" disabled={disabled || !question.trim() || !documents.trim()}>
                {disabled ? 'Agents are reviewing the case...' : 'Run agent review'}
            </button>
        </form>
    );
}
