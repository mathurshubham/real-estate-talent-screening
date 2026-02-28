const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const assessmentApi = {
    // AI Generation
    generateQuestion: async (context) => {
        const response = await fetch(`${API_BASE_URL}/generate?context=${encodeURIComponent(context)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        if (!response.ok) throw new Error('Failed to generate question');
        return response.json();
    },

    // Questions
    getQuestions: async (source = 'standard') => {
        const response = await fetch(`${API_BASE_URL}/questions?source=${source}`);
        if (!response.ok) throw new Error('Failed to fetch questions');
        return response.json();
    },

    // Sessions
    saveSession: async (sessionId, state) => {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state),
        });
        if (!response.ok) throw new Error('Failed to save session');
        return response.json();
    },

    getSession: async (sessionId) => {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch session');
        return response.json();
    }
};
