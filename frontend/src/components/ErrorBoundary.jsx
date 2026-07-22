import React from 'react';

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { error: null };
    }

    static getDerivedStateFromError(error) {
        return { error };
    }

    componentDidCatch(error, info) {
        console.error('UI render failed:', error, info);
    }

    render() {
        if (!this.state.error) {
            return this.props.children;
        }

        return (
            <div className="app">
                <header className="app-header">
                    <h1>Multi-Agent Medical Assessment</h1>
                    <p>Dynamic AI specialist swarm with real-time safety verification</p>
                </header>
                <div className="error-banner" role="alert">
                    <span className="error-icon">!</span>
                    <div>
                        <div className="error-title">The result could not be displayed</div>
                        <div className="error-message">
                            The assessment finished, but one response field had an unexpected format.
                            Refresh the page and try again.
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}
