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
            <h2>📋 Medical Assessment Query</h2>

            <div className="input-group">
                <label htmlFor="question">Clinical Question</label>
                <input
                    id="question"
                    type="text"
                    placeholder="e.g. What medications should be adjusted given the latest labs?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={disabled}
                />
            </div>

            <div className="input-group">
                <label htmlFor="documents">Patient Documents</label>
                <textarea
                    id="documents"
                    placeholder={"Paste patient documents here (lab reports, clinical notes, etc.)\n\nSeparate multiple documents with a blank line."}
                    value={documents}
                    onChange={(e) => setDocuments(e.target.value)}
                    disabled={disabled}
                    rows={6}
                />
            </div>

            <button type="submit" className="submit-btn" disabled={disabled || !question.trim() || !documents.trim()}>
                {disabled ? '⏳ Pipeline Running...' : '🚀 Start Multi-Agent Assessment'}
            </button>
        </form>
    );
}
