import React, { useState, useMemo, useRef, useEffect } from 'react';

import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { Profile } from './pages/Profile';
import { Admin } from './pages/Admin';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import { Skeleton } from "./components/ui/skeleton";

import { toPng } from 'html-to-image';
import {
    Users, Plus, LogIn, ChevronRight, ChevronLeft,
    Award, ClipboardCheck, Briefcase, CheckCircle2, Sparkles
} from 'lucide-react';
import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis,
    PolarRadiusAxis, ResponsiveContainer
} from 'recharts';
import { Modal } from './components/ui';
import { Button } from './components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { QUESTION_LIBRARY, ROLE_TEMPLATES, ASSESSMENT_PILLARS } from './data/assessmentData';
import KAGGLE_QUESTIONS from './data/kaggleQuestions.json';
import { cn } from './lib/utils';
import { assessmentApi, API_BASE_URL } from './services/api';
import { Candidate, Question, SessionState, EvaluationResult, CandidateAssessment } from './types';

function App() {
    const [user, setUser] = useState<{ name: string } | null>(null);
    const [view, setView] = useState<'login' | 'dashboard' | 'candidate' | 'candidate-thanks' | 'interview' | 'summary' | 'review'>('login');
    const [useKaggle, setUseKaggle] = useState(false);
    const [candidates, setCandidates] = useState<Candidate[]>([
        { id: 1, name: "Jordan Smith", role: "manager", status: "pending" },
        { id: 2, name: "Alex Rivera", role: "specialist", status: "pending" },
    ]);
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);
    const [newCandidate, setNewCandidate] = useState({ name: '', role: 'specialist' });

    const [activeSession, setActiveSession] = useState<SessionState | null>(null);
    const [currentQuestionIdx, setCurrentQuestionIdx] = useState(0);
    const [scores, setScores] = useState<Record<string, number>>({});
    const [isAiGenerating, setIsAiGenerating] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [aiEvaluation, setAiEvaluation] = useState<EvaluationResult | null>(null);
    const [isEvaluating, setIsEvaluating] = useState(false);
    const [candidateSession, setCandidateSession] = useState<CandidateAssessment | null>(null);
    const [candidateAnswers, setCandidateAnswers] = useState<Record<string, string>>({});
    const [accessKey, setAccessKey] = useState('');
    const [timeLeft, setTimeLeft] = useState<number | null>(null);
    const [completedAssessments, setCompletedAssessments] = useState<any[]>([]);
    const [reviewAssessment, setReviewAssessment] = useState<CandidateAssessment | null>(null);
    const [socket, setSocket] = useState<WebSocket | null>(null);
    const [remoteScores, setRemoteScores] = useState<Record<string, Record<string, number>>>({}); // Stores other panelists' scores: {questionId: {panelistName: score, ...}}
    const chartRef = useRef<HTMLDivElement>(null);
    const [isDownloading, setIsDownloading] = useState(false);

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        setUser({ name: 'Panelist Lead' });
        setView('dashboard');
    };

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const key = params.get('accessKey');
        if (key) {
            setAccessKey(key);
            fetchCandidateAssessment(key);
        }
    }, []);

    const fetchCandidateAssessment = async (key: string) => {
        try {
            const session = await assessmentApi.getCandidateAssessment(key);
            setCandidateSession(session);
            setView('candidate');
            if (session.questions.length > 0) {
                setTimeLeft(session.questions.length * 120); // 2 mins per question
            }
        } catch (error) {
            console.error("Failed to fetch assessment:", error);
        }
    };

    const fetchCompletedAssessmentList = async () => {
        try {
            const data = await assessmentApi.getCompletedAssessments();
            setCompletedAssessments(data);
        } catch (error) {
            console.error("Failed to fetch completed list:", error);
        }
    };

    useEffect(() => {
        if (view === 'dashboard') {
            fetchCompletedAssessmentList();
        }
    }, [view]);

    const handleReviewAssessment = async (sessionId: string | number) => {
        try {
            const session = await assessmentApi.getSession(sessionId);
            setReviewAssessment(session);
            setView('review');
        } catch (error) {
            console.error("Failed to fetch assessment for review:", error);
        }
    };

    useEffect(() => {
        if (view === 'candidate' && timeLeft !== null && timeLeft > 0) {
            const timer = setInterval(() => setTimeLeft(prev => (prev !== null ? prev - 1 : null)), 1000);
            return () => clearInterval(timer);
        } else if (timeLeft === 0 && view === 'candidate') {
            submitCandidateAssessment();
        }
    }, [view, timeLeft]);

    // WebSocket Connection for Panelists
    useEffect(() => {
        if ((view === 'interview' || view === 'summary') && activeSession && user) {
            const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss:' : 'ws:';
            const wsHost = API_BASE_URL.replace(/^https?:\/\//, '').split('/')[0];
            const wsUrl = `${wsProtocol}//${wsHost}/api/v1/ws/session/${activeSession.candidate.id}`;

            let ws: WebSocket;
            let reconnectTimer: any;

            const connect = () => {
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    console.log("Connected to session broadcast");
                    setSocket(ws);
                };

                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    if (message.type === 'SCORE_UPDATE' && message.panelist !== user.name) {
                        setRemoteScores(prev => ({
                            ...prev,
                            [message.question_id]: {
                                ...(prev[message.question_id] || {}),
                                [message.panelist]: message.score
                            }
                        }));
                    } else if (message.type === 'PING') {
                        ws.send(JSON.stringify({ type: 'PONG' }));
                    }
                };

                ws.onclose = () => {
                    setSocket(null);
                    reconnectTimer = setTimeout(connect, 3000);
                };
            };

            connect();
            return () => {
                if (ws) ws.close();
                clearTimeout(reconnectTimer);
            };
        }
    }, [view, activeSession, user]);

    const addCandidate = () => {
        if (!newCandidate.name) return;
        setCandidates([...candidates, { ...newCandidate, id: Date.now(), status: 'pending' }]);
        setIsAddModalOpen(false);
        setNewCandidate({ name: '', role: 'specialist' });
    };

    const startInterview = async (candidate: Candidate) => {
        let questions: Question[] = [];
        try {
            if (useKaggle) {
                questions = (await assessmentApi.getQuestions('kaggle')) as Question[];
                // Pick 10 random
                questions = questions.sort(() => 0.5 - Math.random()).slice(0, 10);
            } else {
                const roleData = ROLE_TEMPLATES.find(r => r.id === candidate.role);
                if (roleData) {
                    // We still use local QUESTION_LIBRARY for now, but in future this would be an API call
                    questions = roleData.groups.flatMap(group =>
                        (QUESTION_LIBRARY[group] || []).map(q => ({ ...q, pillar: group }))
                    );
                }
            }

            setActiveSession({ candidate, questions });
            setCurrentQuestionIdx(0);
            setScores({});
            setView('interview');
        } catch (err) {
            console.error("Failed to start interview:", err);
        }
    };

    const generateAiQuestion = async () => {
        if (!activeSession) return;
        setIsAiGenerating(true);
        try {
            const currentQuestion = activeSession.questions[currentQuestionIdx];
            const currentPillar = currentQuestion.pillar;
            const roleData = ROLE_TEMPLATES.find(r => r.id === activeSession.candidate.role);
            const candidateRole = roleData ? roleData.name : 'Unknown';
            const currentCategory = currentQuestion.category;
            const currentQuestionText = currentQuestion.text;

            const context = `Pillar: ${currentPillar}, Role: ${candidateRole}, Category: ${currentCategory}, Last: ${currentQuestionText}`;

            const { question } = await assessmentApi.generateQuestion(context);

            const aiQuestion: Question = {
                id: `AI-${Date.now()}`,
                text: `[Gemini Insight] ${question.trim()}`,
                type: 'rating',
                pillar: currentPillar,
                category: 'AI Generated'
            };

            const newQuestions = [...activeSession.questions];
            newQuestions.splice(currentQuestionIdx + 1, 0, aiQuestion);
            setActiveSession({ ...activeSession, questions: newQuestions });

            // Save session state to backend/redis
            await assessmentApi.saveSession(activeSession.candidate.id, {
                questions: newQuestions,
                currentIdx: currentQuestionIdx,
                scores
            });
        } catch (error) {
            console.error("AI Generation Error:", error);
        } finally {
            setIsAiGenerating(false);
        }
    };

    const handleDownloadPdf = async () => {
        if (!chartRef.current || !activeSession) return;
        setIsDownloading(true);
        try {
            // Capture the radar chart at 3x scale for crispness
            const dataUrl = await toPng(chartRef.current, { backgroundColor: '#ffffff', pixelRatio: 3 });

            // Save chart to backend
            await assessmentApi.saveChart(activeSession.candidate.id, dataUrl);

            // Trigger PDF download
            await assessmentApi.downloadPdf(activeSession.candidate.id);
        } catch (error) {
            console.error("PDF Download failed:", error);
        } finally {
            setIsDownloading(false);
        }
    };

    const handleAiEvaluate = async () => {
        if (!transcript || !activeSession) return;
        setIsEvaluating(true);
        try {
            const context = activeSession.questions[currentQuestionIdx].text;
            const result = await assessmentApi.evaluateAnswer(context, transcript);
            setAiEvaluation(result);
        } catch (error) {
            console.error("Evaluation Error:", error);
        } finally {
            setIsEvaluating(false);
        }
    };

    const handleScore = async (value: number) => {
        if (!activeSession) return;
        const currentQuestion = activeSession.questions[currentQuestionIdx];
        const newScores = { ...scores, [currentQuestion.id]: value };
        setScores(newScores);

        // Broadcast score update
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'SCORE_UPDATE',
                question_id: currentQuestion.id,
                score: value,
                panelist: user?.name,
                timestamp: Date.now()
            }));
        }

        // Save to backend
        await assessmentApi.saveSession(activeSession.candidate.id, {
            questions: activeSession.questions,
            currentIdx: currentQuestionIdx + 1,
            scores: newScores
        });

        // Clear AI evaluation for next question
        setTranscript('');
        setAiEvaluation(null);

        if (currentQuestionIdx < activeSession.questions.length - 1) {
            setCurrentQuestionIdx(currentQuestionIdx + 1);
        } else {
            setView('summary');
        }
    };

    const submitCandidateAssessment = async () => {
        if (!candidateSession) return;
        try {
            const submissions = candidateSession.questions.map(q => ({
                question_id: q.id,
                question_text: q.text,
                transcript: candidateAnswers[q.id] || ''
            }));
            await assessmentApi.submitCandidateAssessment(accessKey, submissions);
            setView('candidate-thanks');
        } catch (error) {
            console.error("Submission failed:", error);
        }
    };

    const chartData = useMemo(() => {
        if (!activeSession) return [];
        return ASSESSMENT_PILLARS.map(pillar => {
            const pillarQuestions = activeSession.questions.filter(q => q.pillar === pillar);

            // Aggregate local and remote scores
            const averages = pillarQuestions.map(q => {
                const localScore = scores[q.id] || 0;
                const remoteScoresForQ = Object.values(remoteScores[q.id] || {});
                const allScores = localScore > 0 ? [localScore, ...remoteScoresForQ] : remoteScoresForQ;
                return allScores.length > 0 ? allScores.reduce((a, b) => a + b, 0) / allScores.length : 0;
            });

            const average = averages.length > 0
                ? averages.reduce((a, b) => a + b, 0) / averages.length
                : 0;
            return { subject: pillar, A: average, fullMark: 5 };
        });
    }, [activeSession, scores, remoteScores]);


    const DashboardContent = () => (
        <div className="animate-in fade-in duration-500">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Welcome back, {user?.name.split(' ')[0] || 'User'}</h2>
                    <p className="text-muted-foreground mt-1">Here is what's happening today.</p>
                </div>
                <div className="flex items-center gap-4">
                    <Button onClick={() => setIsAddModalOpen(true)} className="gap-2 bg-primary">
                        <Plus className="h-4 w-4" /> New Assessment
                    </Button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Active Interviews</CardTitle>
                        <Briefcase className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{candidates.filter(c => c.status === 'pending').length}</div>
                        <p className="text-xs text-muted-foreground">+2 this week</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Candidates</CardTitle>
                        <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{candidates.length}</div>
                        <p className="text-xs text-muted-foreground">+12 this month</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Evaluations Today</CardTitle>
                        <ClipboardCheck className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{completedAssessments.length}</div>
                        <p className="text-xs text-muted-foreground">Stable trend</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Avg. Score</CardTitle>
                        <Award className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">4.2</div>
                        <p className="text-xs text-muted-foreground">+0.1 from last month</p>
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Recent Activity</CardTitle>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Name</TableHead>
                                <TableHead>Role</TableHead>
                                <TableHead>Date</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>Actions</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {candidates.map(candidate => (
                                <TableRow key={candidate.id}>
                                    <TableCell className="font-medium">
                                        <div className="flex items-center gap-3">
                                            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold text-xs">
                                                {String(candidate.name).charAt(0)}
                                            </div>
                                            {candidate.name}
                                        </div>
                                    </TableCell>
                                    <TableCell>{ROLE_TEMPLATES.find(r => r.id === candidate.role)?.name || candidate.role}</TableCell>
                                    <TableCell>{new Date().toLocaleDateString()}</TableCell>
                                    <TableCell>
                                        <Badge variant={candidate.status === 'evaluated' ? 'default' : 'secondary'}>
                                            {candidate.status}
                                        </Badge>
                                    </TableCell>
                                    <TableCell>
                                        <Button
                                            variant="ghost" size="sm"
                                            onClick={() => startInterview(candidate)}
                                        >
                                            Start <ChevronRight className="h-4 w-4 ml-1" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {completedAssessments.map(item => (
                                <TableRow key={"comp-"+item.id}>
                                    <TableCell className="font-medium">
                                        <div className="flex items-center gap-3">
                                            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold text-xs">
                                                {String(item.candidate_name).charAt(0)}
                                            </div>
                                            {item.candidate_name}
                                        </div>
                                    </TableCell>
                                    <TableCell>Completed Role</TableCell>
                                    <TableCell>{new Date(item.submitted_at).toLocaleDateString()}</TableCell>
                                    <TableCell>
                                        <Badge variant="outline" className="border-green-500 text-green-600">Evaluated</Badge>
                                    </TableCell>
                                    <TableCell>
                                        <Button variant="ghost" size="sm" onClick={() => handleReviewAssessment(item.id)}>
                                            Review
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Modal isOpen={isAddModalOpen} onClose={() => setIsAddModalOpen(false)} title="Register Candidate">
                <div className="space-y-4">
                    <div className="space-y-2">
                        <label className="text-sm font-semibold">Candidate Name</label>
                        <Input
                            placeholder="e.g. Robert Vance"
                            value={newCandidate.name}
                            onChange={(e) => setNewCandidate({ ...newCandidate, name: e.target.value })}
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-semibold">Assessment Role</label>
                        <select
                            className="w-full h-10 px-3 py-2 rounded-md border border-input text-sm bg-background text-foreground"
                            value={newCandidate.role}
                            onChange={(e) => setNewCandidate({ ...newCandidate, role: e.target.value })}
                        >
                            {ROLE_TEMPLATES.map(role => (
                                <option key={role.id} value={role.id}>{role.name}</option>
                            ))}
                        </select>
                    </div>
                    <div className="pt-4 flex gap-3">
                        <Button variant="outline" className="flex-1" onClick={() => setIsAddModalOpen(false)}>Cancel</Button>
                        <Button className="flex-1" onClick={addCandidate}>Add to List</Button>
                    </div>
                </div>
            </Modal>
        </div>
    );

    const InterviewContent = () => {
        if (!activeSession) return null;
        return (
            <div className="flex flex-col lg:flex-row gap-6 animate-in fade-in duration-500">
                <div className="w-full lg:w-64 shrink-0 space-y-6">
                    <Card className="sticky top-6">
                        <CardHeader className="pb-3 border-b">
                            <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Candidate Info</CardTitle>
                        </CardHeader>
                        <CardContent className="pt-6 space-y-4">
                            <div>
                                <h3 className="font-bold text-lg">{activeSession.candidate.name}</h3>
                                <p className="text-sm text-muted-foreground">{ROLE_TEMPLATES.find(r => r.id === activeSession.candidate.role)?.name}</p>
                            </div>
                            <div className="space-y-3 pt-4 border-t text-sm">
                                <div>
                                    <span className="text-muted-foreground block text-xs">Email</span>
                                    <span>{activeSession.candidate.name.replace(' ', '.').toLowerCase()}@gmail.com</span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground block text-xs">Assessment ID</span>
                                    <span className="font-mono text-xs">#{activeSession.candidate.id}-2024</span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <div className="flex-1 min-w-0">
                    <Card className="shadow-lg min-h-[500px] flex flex-col">
                        <CardHeader className="flex flex-row items-center justify-between border-b bg-muted/30 pb-4">
                            <div>
                                <CardTitle className="text-lg">Active Interview</CardTitle>
                                <div className="text-sm text-muted-foreground mt-1 flex items-center gap-2">
                                    <Badge variant="outline">{activeSession.questions[currentQuestionIdx].pillar}</Badge>
                                    <span>Step {currentQuestionIdx + 1} of {activeSession.questions.length}</span>
                                </div>
                            </div>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={generateAiQuestion}
                                disabled={isAiGenerating}
                                className="gap-2"
                            >
                                {isAiGenerating ? <Sparkles className="h-4 w-4 animate-spin text-primary" /> : <Sparkles className="h-4 w-4 text-primary" />}
                                Smart Generate
                            </Button>
                        </CardHeader>

                        <CardContent className="flex-1 pt-6 flex flex-col gap-6">
                            <div className="space-y-4 flex-1">
                                {isAiGenerating ? (
                                    <div className="space-y-2">
                                        <Skeleton className="h-6 w-3/4" />
                                        <Skeleton className="h-6 w-full" />
                                        <Skeleton className="h-6 w-5/6" />
                                    </div>
                                ) : (
                                    <h2 className="text-xl sm:text-2xl font-semibold leading-tight flex items-start gap-3">
                                        <div className="bg-primary/10 p-2 rounded-md shrink-0 mt-1">
                                            <CheckCircle2 className="h-5 w-5 text-primary" />
                                        </div>
                                        {activeSession.questions[currentQuestionIdx].text}
                                    </h2>
                                )}

                                <div className="pt-6 flex-1 flex flex-col">
                                    <label className="text-sm font-medium text-muted-foreground mb-2">Response Transcription</label>
                                    <textarea
                                        className="flex-1 w-full min-h-[160px] p-4 rounded-md border bg-background text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 resize-none"
                                        placeholder="Write your response here..."
                                        value={transcript}
                                        onChange={(e) => setTranscript(e.target.value)}
                                    />
                                    <div className="mt-4 flex justify-end gap-3">
                                        <Button
                                            variant="secondary"
                                            onClick={handleAiEvaluate}
                                            disabled={isEvaluating || !transcript}
                                            className="gap-2"
                                        >
                                            {isEvaluating ? 'Evaluating...' : 'Get AI Score Suggestion'}
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <div className="w-full lg:w-72 shrink-0">
                    <Card className="h-full bg-muted/20 border">
                        <CardHeader>
                            <CardTitle className="text-lg flex items-center gap-2">
                                <Sparkles className="h-5 w-5" /> AI Evaluation
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            {isEvaluating ? (
                                <div className="space-y-3">
                                    <Skeleton className="h-24 w-full" />
                                    <Skeleton className="h-32 w-full" />
                                </div>
                            ) : aiEvaluation ? (
                                <div className="space-y-4 animate-in fade-in">
                                    <div className="bg-card border rounded-lg p-6 text-center shadow-sm">
                                        <p className="text-sm text-muted-foreground mb-2">Suggested score:</p>
                                        <div className="text-5xl font-bold tracking-tighter">{aiEvaluation.score}<span className="text-xl text-muted-foreground">/5</span></div>
                                    </div>
                                    <p className="text-sm text-muted-foreground leading-relaxed">
                                        {aiEvaluation.justification || aiEvaluation.feedback}
                                    </p>
                                    
                                    <div className="space-y-3 pt-4 border-t">
                                        <p className="text-sm font-medium">Final Rating</p>
                                        <div className="flex justify-between">
                                            {[1, 2, 3, 4, 5].map(val => (
                                                <Button
                                                    key={val}
                                                    variant={aiEvaluation.score === val ? "default" : "outline"}
                                                    size="icon"
                                                    className="h-10 w-10 font-bold"
                                                    onClick={() => handleScore(val)}
                                                >
                                                    {val}
                                                </Button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <p className="text-sm text-muted-foreground text-center py-8">
                                    AI evaluation feedback loop will generate suggested STAR feedback here.
                                </p>
                            )}

                             {activeSession.questions[currentQuestionIdx].type !== 'rating' && !aiEvaluation && (
                                <div className="space-y-3 mt-8 pt-6 border-t">
                                    <p className="text-sm font-medium mb-3">Or manually select best match:</p>
                                    {activeSession.questions[currentQuestionIdx].options?.map((opt, i) => (
                                        <Button
                                            key={i}
                                            variant="outline"
                                            className="w-full justify-start text-left h-auto py-3 px-4 font-normal"
                                            onClick={() => handleScore(opt.value)}
                                        >
                                            {opt.label}
                                        </Button>
                                    ))}
                                </div>
                            )}

                        </CardContent>
                    </Card>
                </div>
            </div>
        );
    };

    if (!user) {
        return (
            <div className="min-h-screen bg-background flex flex-col">
              <div className="flex-1 flex items-center justify-center p-4">
                <Card className="w-full max-w-md shadow-lg border">
                    <CardHeader className="text-center pb-2">
                        <div className="bg-primary/10 p-3 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                            <ClipboardCheck className="h-8 w-8 text-primary" />
                        </div>
                        <CardTitle className="text-2xl font-bold">EstateAssess</CardTitle>
                        <p className="text-muted-foreground text-sm mt-2">Sign in to the Panelist Portal</p>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleLogin} className="space-y-4 pt-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Email Address</label>
                                <Input type="email" placeholder="panelist@estateassess.com" defaultValue="demo@estateassess.com" required className="bg-muted/50" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Password</label>
                                <Input type="password" placeholder="••••••••" defaultValue="password" required className="bg-muted/50" />
                            </div>
                            <Button type="submit" className="w-full mt-2" size="lg">Sign In</Button>
                        </form>
                    </CardContent>
                </Card>
              </div>
            </div>
        );
    }

    return (
        <BrowserRouter>
            <DashboardLayout>
                <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<DashboardContent />} />
                    <Route path="/profile" element={<Profile />} />
                    <Route path="/admin" element={<Admin />} />
                    <Route path="/interview" element={view === 'interview' ? <InterviewContent /> : <Navigate to="/dashboard" replace />} />
                    <Route path="/summary" element={
                        <div className="p-8 bg-card border rounded-xl text-center shadow-sm">
                            <CheckCircle2 className="h-16 w-16 text-green-500 mx-auto mb-4" />
                            <h2 className="text-3xl font-bold mb-2">Assessment Complete</h2>
                            <p className="text-muted-foreground mb-8">The assessment data has been recorded successfully.</p>
                            <Button onClick={() => setView('dashboard')}>Return to Dashboard</Button>
                            <div className="hidden">
                                <div ref={chartRef}></div>
                            </div>
                        </div>
                    } />
                    <Route path="*" element={<DashboardContent />} />
                </Routes>
                
                <StateRouter view={view} setView={setView} />
            </DashboardLayout>
        </BrowserRouter>
    );
}

const StateRouter = ({ view, setView }: { view: string, setView: any }) => {
    const navigate = useNavigate();
    const location = useLocation();

    useEffect(() => {
        if (view === 'interview' && location.pathname !== '/interview') {
            navigate('/interview');
        } else if (view === 'summary' && location.pathname !== '/summary') {
            navigate('/summary');
        } else if (view === 'dashboard' && location.pathname !== '/dashboard' && location.pathname !== '/profile' && location.pathname !== '/admin') {
            navigate('/dashboard');
        }
    }, [view]);

    useEffect(() => {
        if (location.pathname === '/profile' || location.pathname === '/admin') {
            setView(location.pathname.substring(1));
        }
    }, [location.pathname]);

    return null;
}
export default App;
