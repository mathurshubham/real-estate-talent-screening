export interface Question {
    id: string;
    text: string;
    type: 'rating' | 'mcq';
    category: string;
    pillar?: string;
    options?: { label: string; value: number }[];
}

export interface Candidate {
    id: number | string;
    name: string;
    role: string;
    status: 'pending' | 'completed' | 'evaluated';
    submitted_at?: string;
}

export interface SessionState {
    candidate: Candidate;
    questions: Question[];
    currentIdx?: number;
    scores?: Record<string, number>;
}

export interface EvaluationResult {
    score: number;
    feedback: string;
    justification?: string;
    analysis?: string;
}

export interface CandidateAssessment {
    id: string;
    access_key: string;
    candidate_name: string;
    status: 'pending' | 'completed';
    questions: Question[];
    candidate_answers?: { question_id: string; question_text: string; transcript: string }[];
    ai_evaluations?: Record<string, EvaluationResult>;
}

export interface RoleTemplate {
    id: string;
    name: string;
    groups: string[];
    description: string;
}
