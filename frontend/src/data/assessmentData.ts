import { Question, RoleTemplate } from '../types';

export const ASSESSMENT_PILLARS = ['Skill', 'Training', 'Attitude', 'Results'];

export const QUESTION_LIBRARY: Record<string, Question[]> = {
    Skill: [
        { id: 'S1', text: "Assess candidate's technical relevance to the real estate domain.", type: 'rating', category: "Relevance" },
        { id: 'S2', text: "Technical proficiency in CRM and market analysis tools.", type: 'rating', category: "Technical" },
        { id: 'S3', text: "Testing aptitude: Proficiency in evaluating property valuations and cap rates.", type: 'rating', category: "Technical Testing" },
        { id: 'S4', text: "Managerial capability: Experience in overseeing agent performance and office operations.", type: 'rating', category: "Managerial" },
    ],
    Training: [
        {
            id: 'T1',
            text: "Total years of relevant industrial experience.",
            type: 'mcq',
            category: "Experience",
            options: [
                { label: "10+ years (Industry Expert)", value: 5 },
                { label: "5-10 years (Senior)", value: 4 },
                { label: "2-5 years (Experience)", value: 3 },
                { label: "0-2 years (Junior)", value: 2 },
                { label: "No relevant experience", value: 1 }
            ]
        },
        {
            id: 'T2',
            text: "Certification and Professional Qualification level.",
            type: 'mcq',
            category: "Qualifications",
            options: [
                { label: "Advanced Cert (Broker License + MBA/Specialization)", value: 5 },
                { label: "Full Broker License", value: 4 },
                { label: "Realtor Certification", value: 3 },
                { label: "Entry-level License", value: 2 },
                { label: "No certifications", value: 1 }
            ]
        },
    ],
    Attitude: [
        {
            id: 'A1',
            text: "Response to high-pressure negotiations or client objections.",
            type: 'mcq',
            category: "Responsiveness",
            options: [
                { label: "Stays calm, provides data-driven solutions immediately", value: 5 },
                { label: "Confident but requires time to consult data", value: 4 },
                { label: "Professional but shows visible pressure", value: 3 },
                { label: "Slow response, avoids direct conflict", value: 2 },
                { label: "Becomes defensive or unresponsive", value: 1 }
            ]
        },
        {
            id: 'A2',
            text: "Leadership stance when managing a failing sales target.",
            type: 'mcq',
            category: "Leadership",
            options: [
                { label: "Ownership: Mentors team and pivots strategy", value: 5 },
                { label: "Directive: Implements strict KPIs", value: 4 },
                { label: "Supportive: Encourages team but lacks new plan", value: 3 },
                { label: "Externalizes: Blames market conditions", value: 2 },
                { label: "Withdraws: Minimal engagement", value: 1 }
            ]
        },
    ],
    Results: [
        {
            id: 'R1',
            text: "Historical conversion rate from lead to closure.",
            type: 'mcq',
            category: "Performance",
            options: [
                { label: "Exceeds 25% (Top Tier)", value: 5 },
                { label: "15% - 25% (High Performer)", value: 4 },
                { label: "8% - 15% (Solid Average)", value: 3 },
                { label: "3% - 8% (Needs Growth)", value: 2 },
                { label: "Under 3% (Poor)", value: 1 }
            ]
        },
        {
            id: 'R2',
            text: "Customer Satisfaction (CSAT) score or Referral rate.",
            type: 'mcq',
            category: "Outcomes",
            options: [
                { label: "Over 90% referral-based business", value: 5 },
                { label: "70-90% CSAT excellence", value: 4 },
                { label: "Standard client retention", value: 3 },
                { label: "Mixed feedback from clients", value: 2 },
                { label: "High churn or negative reviews", value: 1 }
            ]
        },
    ]
};

export const ROLE_TEMPLATES: RoleTemplate[] = [
    {
        id: 'specialist',
        name: 'Sales Specialist',
        groups: ['Skill', 'Training', 'Results'],
        description: 'Focused on technical relevance and closure results.'
    },
    {
        id: 'manager',
        name: 'Estate Manager',
        groups: ['Skill', 'Training', 'Attitude', 'Results'],
        description: 'Requires leadership attitude and managerial skill.'
    },
    {
        id: 'associate',
        name: 'Junior Associate',
        groups: ['Skill', 'Attitude'],
        description: 'Evaluated on core skill and responsiveness.'
    }
];
