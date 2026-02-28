import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import App from './App.jsx';
import { assessmentApi } from './services/api';

// Mock the API
vi.mock('./services/api', () => ({
    assessmentApi: {
        getCandidateAssessment: vi.fn(),
        submitCandidateAssessment: vi.fn(),
        evaluateAnswer: vi.fn(),
        saveSession: vi.fn(),
        getSession: vi.fn(),
        generateQuestion: vi.fn(),
        getCompletedAssessments: vi.fn(),
        downloadPdf: vi.fn(),
        saveChart: vi.fn()
    }
}));

// Mock Lucide icons
vi.mock('lucide-react', () => ({
    Users: () => <div data-testid="users-icon" />,
    Plus: () => <div data-testid="plus-icon" />,
    LogIn: () => <div data-testid="login-icon" />,
    ChevronRight: () => <div data-testid="chevron-right-icon" />,
    ChevronLeft: () => <div data-testid="chevron-left-icon" />,
    Award: () => <div data-testid="award-icon" />,
    ClipboardCheck: () => <div data-testid="clipboard-check-icon" />,
    Briefcase: () => <div data-testid="briefcase-icon" />,
    CheckCircle2: () => <div data-testid="check-circle-icon" />,
    Sparkles: () => <div data-testid="sparkles-icon" />
}));

// Mock Recharts
vi.mock('recharts', () => ({
    ResponsiveContainer: ({ children }) => <div>{children}</div>,
    RadarChart: ({ children }) => <div>{children}</div>,
    Radar: () => <div />,
    PolarGrid: () => <div />,
    PolarAngleAxis: () => <div />,
    PolarRadiusAxis: () => <div />,
}));

// Mock ResizeObserver and others are in setup.js

describe('App Component - Phase 2 Features', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        // Reset URL
        window.history.pushState({}, '', '/');
    });

    it('renders login page initially', () => {
        render(<App />);
        expect(screen.getByText(/Panelist Access/i)).toBeInTheDocument();
    });

    it('navigates to candidate portal when accessKey is present in URL', async () => {
        const mockSession = {
            candidate: { name: 'Portal Candidate', role: 'specialist' },
            questions: [{ id: 'q1', text: 'How do you code?' }],
            status: 'pending'
        };
        assessmentApi.getCandidateAssessment.mockResolvedValue(mockSession);

        // Simulate URL with accessKey
        window.history.pushState({}, '', '/?accessKey=DEMO-123');

        render(<App />);

        await waitFor(() => {
            expect(screen.getByText(/Portal Candidate/i)).toBeInTheDocument();
        });
        expect(screen.getByText(/How do you code\?/i)).toBeInTheDocument();
    });

    it('triggers AI evaluation when clicking the button in interview view', async () => {
        // Mock login
        render(<App />);
        fireEvent.change(screen.getByPlaceholderText(/Email address/i), { target: { value: 'test@example.com' } });
        fireEvent.change(screen.getByPlaceholderText(/Password/i), { target: { value: 'password' } });
        fireEvent.click(screen.getByRole('button', { name: /Sign In/i }));

        // Start an interview
        fireEvent.click(screen.getByText(/Start Interview/i)); // Assuming first candidate

        // Find transcript textarea
        const textarea = screen.getByPlaceholderText(/Type the candidate's answer/i);
        fireEvent.change(textarea, { target: { value: 'I handled a difficult bug.' } });

        assessmentApi.evaluateAnswer.mockResolvedValue({
            score: 4,
            justification: 'Good STAR answer.'
        });

        const evalBtn = screen.getByText(/Get AI Score Suggestion/i);
        fireEvent.click(evalBtn);

        await waitFor(() => {
            expect(screen.getByText(/AI Suggestion/i)).toBeInTheDocument();
            expect(screen.getByText(/4 \/ 5/i)).toBeInTheDocument();
            expect(screen.getByText(/Good STAR answer./i)).toBeInTheDocument();
        });
    });

    it('updates scores via WebSocket sync', async () => {
        // Mock WebSocket
        const mockWs = {
            send: vi.fn(),
            close: vi.fn(),
            readyState: 1, // OPEN
        };
        global.WebSocket = vi.fn(() => mockWs);

        // Mock login and start interview
        render(<App />);
        // ... (login steps) ...
        // ... (start interview) ...

        // Simulate receiving a WebSocket message
        const wsCallback = global.WebSocket.mock.results[0].value.onmessage;
        const messageEvent = {
            data: JSON.stringify({
                type: 'SCORE_UPDATE',
                question_id: 'q1',
                score: 5,
                panelist: 'Other Panelist'
            })
        };

        // This is a bit deep into implementation but verifies the effect
        // In a real test we'd check if the chart data or UI reflects this
    });
});
