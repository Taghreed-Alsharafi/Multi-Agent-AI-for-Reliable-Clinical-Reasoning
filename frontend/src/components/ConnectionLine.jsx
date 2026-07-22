import React from 'react';

export default function ConnectionLine({ active, fromColor, toColor }) {
    return (
        <div className={`connection-line ${active ? 'active' : ''}`}
            style={{ '--from-color': fromColor, '--to-color': toColor }}>
            <div className="line-track">
                <div className="line-fill" />
            </div>
            <div className="pulse-dot" />
        </div>
    );
}
