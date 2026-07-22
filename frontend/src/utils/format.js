export function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value === null || value === undefined || value === '') return [];
    return [value];
}

export function displayText(value, fallback = '') {
    if (value === null || value === undefined) return fallback;
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) {
        return value.map((item) => displayText(item)).filter(Boolean).join(', ');
    }
    if (typeof value === 'object') {
        return (
            value.message ||
            value.summary ||
            value.finding ||
            value.claim ||
            value.issue ||
            JSON.stringify(value)
        );
    }
    return fallback;
}

export function boundedPercent(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.min(100, Math.max(0, n * 100));
}
