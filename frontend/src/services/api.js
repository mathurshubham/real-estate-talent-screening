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
    },

    // Evaluation
    evaluateAnswer: async (questionContext, transcript) => {
        const response = await fetch(`${API_BASE_URL}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question_context: questionContext,
                candidate_transcript: transcript
            }),
        });
        if (!response.ok) throw new Error('Failed to evaluate answer');
        return response.json();
    },

    // Candidate Portal
    getCandidateAssessment: async (accessKey) => {
        const response = await fetch(`${API_BASE_URL}/candidate/assessment/${accessKey}`);
        if (!response.ok) throw new Error('Failed to fetch candidate assessment');
        return response.json();
    },

    submitCandidateAssessment: async (accessKey, answers) => {
        const response = await fetch(`${API_BASE_URL}/candidate/assessment/${accessKey}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ answers }),
        });
        if (!response.ok) throw new Error('Failed to submit assessment');
        return response.json();
    },

    getCompletedAssessments: async () => {
        const response = await fetch(`${API_BASE_URL}/candidate/assessments/completed`);
        if (!response.ok) throw new Error('Failed to fetch completed assessments');
        return response.json();
    },

    saveChart: async (sessionId, imageData) => {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/chart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: imageData }),
        });
        if (!response.ok) throw new Error('Failed to save chart');
        return response.json();
    },

    downloadPdf: async (sessionId) => {
        window.open(`${API_BASE_URL}/sessions/${sessionId}/pdf`, '_blank');
    }
};
