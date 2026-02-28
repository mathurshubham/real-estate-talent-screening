import React, { useState, useMemo, useRef } from 'react';
import { toPng } from 'html-to-image';
import {
  Users, Plus, LogIn, ChevronRight, ChevronLeft,
  Award, ClipboardCheck, Briefcase, CheckCircle2, Sparkles
} from 'lucide-react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer
} from 'recharts';
import {
  Button, Card, CardHeader, CardTitle, CardContent, Input, Modal, Badge
} from './components/ui';
import { QUESTION_LIBRARY, ROLE_TEMPLATES, ASSESSMENT_PILLARS } from './data/assessmentData';
import KAGGLE_QUESTIONS from './data/kaggleQuestions.json';
import { cn } from './lib/utils';
import { assessmentApi } from './services/api';

function App() {
  const [user, setUser] = useState(null);
  const [view, setView] = useState('login');
  const [useKaggle, setUseKaggle] = useState(false);
  const [candidates, setCandidates] = useState([
    { id: 1, name: "Jordan Smith", role: "manager", status: "pending" },
    { id: 2, name: "Alex Rivera", role: "specialist", status: "pending" },
  ]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newCandidate, setNewCandidate] = useState({ name: '', role: 'specialist' });

  const [activeSession, setActiveSession] = useState(null);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(0);
  const [scores, setScores] = useState({});
  const [isAiGenerating, setIsAiGenerating] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [aiEvaluation, setAiEvaluation] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [candidateSession, setCandidateSession] = useState(null);
  const [candidateAnswers, setCandidateAnswers] = useState({});
  const [accessKey, setAccessKey] = useState('');
  const [timeLeft, setTimeLeft] = useState(null);
  const [completedAssessments, setCompletedAssessments] = useState([]);
  const [reviewAssessment, setReviewAssessment] = useState(null);
  const [socket, setSocket] = useState(null);
  const [remoteScores, setRemoteScores] = useState({}); // Stores other panelists' scores: {questionId: {panelistName: score, ...}}
  const chartRef = useRef(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    setUser({ name: 'Panelist Lead' });
    setView('dashboard');
  };

  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const key = params.get('accessKey');
    if (key) {
      setAccessKey(key);
      fetchCandidateAssessment(key);
    }
  }, []);

  const fetchCandidateAssessment = async (key) => {
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

  React.useEffect(() => {
    if (view === 'dashboard') {
      fetchCompletedAssessmentList();
    }
  }, [view]);

  const handleReviewAssessment = async (sessionId) => {
    try {
      const session = await assessmentApi.getSession(sessionId);
      setReviewAssessment(session);
      setView('review');
    } catch (error) {
      console.error("Failed to fetch assessment for review:", error);
    }
  };

  React.useEffect(() => {
    if (view === 'candidate' && timeLeft > 0) {
      const timer = setInterval(() => setTimeLeft(prev => prev - 1), 1000);
      return () => clearInterval(timer);
    } else if (timeLeft === 0 && view === 'candidate') {
      submitCandidateAssessment();
    }
  }, [view, timeLeft]);

  // WebSocket Connection for Panelists
  React.useEffect(() => {
    if ((view === 'interview' || view === 'summary') && activeSession && user) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host;
      const wsUrl = `${protocol}//${host}/api/v1/ws/session/${activeSession.candidate.id}`;

      let ws;
      let reconnectTimer;

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

  const startInterview = async (candidate) => {
    let questions = [];
    try {
      if (useKaggle) {
        questions = await assessmentApi.getQuestions('kaggle');
        // Pick 10 random
        questions = questions.sort(() => 0.5 - Math.random()).slice(0, 10);
      } else {
        const roleData = ROLE_TEMPLATES.find(r => r.id === candidate.role);
        // We still use local QUESTION_LIBRARY for now, but in future this would be an API call
        questions = roleData.groups.flatMap(group =>
          QUESTION_LIBRARY[group].map(q => ({ ...q, pillar: group }))
        );
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
    setIsAiGenerating(true);
    try {
      const currentPillar = activeSession.questions[currentQuestionIdx].pillar;
      const candidateRole = ROLE_TEMPLATES.find(r => r.id === activeSession.candidate.role).name;
      const currentCategory = activeSession.questions[currentQuestionIdx].category;
      const currentQuestionText = activeSession.questions[currentQuestionIdx].text;

      const context = `Pillar: ${currentPillar}, Role: ${candidateRole}, Category: ${currentCategory}, Last: ${currentQuestionText}`;

      const { question } = await assessmentApi.generateQuestion(context);

      const aiQuestion = {
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
    if (!chartRef.current) return;
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
    if (!transcript) return;
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

  const handleScore = async (value) => {
    const newScores = { ...scores, [activeSession.questions[currentQuestionIdx].id]: value };
    setScores(newScores);

    // Broadcast score update
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'SCORE_UPDATE',
        question_id: activeSession.questions[currentQuestionIdx].id,
        score: value,
        panelist: user.name,
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

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <header className="sticky top-0 z-40 w-full border-b bg-white/80 backdrop-blur-md">
        <div className="container flex h-16 items-center justify-between mx-auto px-4">
          <div className="flex items-center gap-2">
            <div className="bg-[#1A365D] p-1.5 rounded-lg">
              <ClipboardCheck className="text-white h-5 w-5" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-[#1A365D]">
              Estate<span className="text-[#D4AF37]">Assess</span>
              <span className="ml-2 text-[10px] font-normal text-slate-400 uppercase tracking-widest">v3.0 STAR</span>
            </h1>
          </div>
          {user && (
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-slate-600 hidden sm:inline">User: {user.name}</span>
              <Button variant="ghost" size="sm" onClick={() => setView('login')}>Logout</Button>
            </div>
          )}
        </div>
      </header>

      <main className="container py-8 mx-auto px-4">
        {view === 'login' && (
          <div className="flex h-[70vh] items-center justify-center">
            <Card className="w-full max-w-md p-8 shadow-2xl">
              <div className="text-center mb-8">
                < Award className="h-12 w-12 text-[#D4AF37] mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-[#1A365D]">STAR Recruitment Access</h2>
                <p className="text-slate-500 text-sm mt-2">Professional Panelist Authentication</p>
              </div>
              <form onSubmit={handleLogin} className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-semibold">Email</label>
                  <Input type="email" placeholder="panelist@realestate.com" required />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-semibold">Password</label>
                  <Input type="password" placeholder="••••••••" required />
                </div>
                <Button type="submit" className="w-full">Sign In</Button>
              </form>
              <div className="mt-6 pt-6 border-t text-center">
                <p className="text-xs text-slate-400 mb-2">Candidate participating in a remote assessment?</p>
                <Button variant="ghost" size="sm" className="text-xs text-[#1A365D] hover:underline" onClick={() => {
                  const demoKey = "DEMO-123";
                  // In a real app, this key would be generated per candidate
                  window.history.pushState({}, '', `?accessKey=${demoKey}`);
                  setAccessKey(demoKey);
                  fetchCandidateAssessment(demoKey);
                }}>Candidate Assessment Demo</Button>
              </div>
            </Card>
          </div>
        )}

        {view === 'candidate' && candidateSession && (
          <div className="max-w-3xl mx-auto animate-in fade-in zoom-in-95 duration-500">
            <div className="mb-8 flex justify-between items-center bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
              <div>
                <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Candidate Assessment</h2>
                <p className="text-xl font-bold text-[#1A365D]">{candidateSession.candidate_name}</p>
              </div>
              <div className="text-right">
                <span className="text-[10px] font-bold text-slate-400 uppercase block">Time Remaining</span>
                <span className={cn("text-2xl font-black tabular-nums", timeLeft < 60 ? "text-red-500 animate-pulse" : "text-[#1A365D]")}>
                  {Math.floor(timeLeft / 60)}:{String(timeLeft % 60).padStart(2, '0')}
                </span>
              </div>
            </div>

            <div className="space-y-6">
              {candidateSession.questions.map((q, idx) => (
                <Card key={q.id} className="p-8 border-l-4 border-l-[#D4AF37]">
                  <h3 className="text-sm font-bold text-[#D4AF37] mb-4 uppercase tracking-wider">Question {idx + 1}</h3>
                  <p className="text-lg font-bold text-slate-800 mb-6">{q.text}</p>
                  <textarea
                    className="w-full p-4 rounded-xl border-2 border-slate-100 focus:border-[#1A365D] focus:ring-0 transition-all text-sm min-h-[150px]"
                    placeholder="Write your STAR-based answer here..."
                    value={candidateAnswers[q.id] || ''}
                    onChange={(e) => setCandidateAnswers({ ...candidateAnswers, [q.id]: e.target.value })}
                  />
                </Card>
              ))}
              <div className="pt-8">
                <Button className="w-full py-6 text-lg font-bold shadow-xl flex items-center justify-center gap-2" size="lg" onClick={submitCandidateAssessment}>
                  Submit Assessment <CheckCircle2 className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </div>
        )}

        {view === 'candidate-thanks' && (
          <div className="flex h-[70vh] items-center justify-center text-center">
            <Card className="max-w-md p-12 shadow-2xl space-y-6">
              <div className="h-20 w-20 bg-[#48BB78]/10 text-[#48BB78] rounded-full flex items-center justify-center mx-auto">
                <CheckCircle2 className="h-10 w-10" />
              </div>
              <h2 className="text-3xl font-black text-[#1A365D]">Assessment Received</h2>
              <p className="text-slate-500">Thank you for completing your screening. Our team will review your responses and get in touch shortly.</p>
              <Button onClick={() => {
                window.history.pushState({}, '', window.location.pathname);
                setView('login');
              }} variant="ghost" className="text-slate-400">Back to Home</Button>
            </Card>
          </div>
        )}

        {view === 'dashboard' && (
          <div className="animate-in fade-in slide-in-from-right-8 duration-500">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
              <div>
                <h2 className="text-3xl font-bold text-[#1A365D]">Assessment Hub</h2>
                <p className="text-slate-500 mt-1">STAR Methodology: Skill, Training, Attitude, Results</p>
                <div className="flex items-center gap-2 mt-4">
                  <label className="text-xs font-bold text-slate-400 uppercase">Question Source:</label>
                  <button
                    onClick={() => setUseKaggle(false)}
                    className={cn("px-3 py-1 text-[10px] rounded-full font-bold transition-all", !useKaggle ? "bg-[#1A365D] text-white" : "bg-slate-200 text-slate-500")}
                  >Standard</button>
                  <button
                    onClick={() => setUseKaggle(true)}
                    className={cn("px-3 py-1 text-[10px] rounded-full font-bold transition-all", useKaggle ? "bg-[#D4AF37] text-white" : "bg-slate-200 text-slate-500")}
                  >Kaggle HR (50+)</button>
                </div>
              </div>
              <Button onClick={() => setIsAddModalOpen(true)} className="gap-2 shrink-0" variant="accent">
                <Plus className="h-4 w-4" /> Add New Candidate
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {candidates.map(candidate => (
                <Card key={candidate.id} className="group hover:border-[#1A365D] transition-all">
                  <CardContent className="p-6 pt-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center text-[#1A365D] font-bold text-lg group-hover:bg-[#1A365D] group-hover:text-white transition-colors">
                        {candidate.name.charAt(0)}
                      </div>
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded bg-slate-100 text-slate-500">
                        {candidate.role}
                      </span>
                    </div>
                    <CardTitle className="mb-1">{candidate.name}</CardTitle>
                    <p className="text-xs text-slate-400 mb-6 flex items-center gap-1">
                      <Briefcase className="h-3 w-3" />
                      {ROLE_TEMPLATES.find(r => r.id === candidate.role).name}
                    </p>
                    <Button
                      className="w-full group/btn"
                      variant="outline"
                      onClick={() => startInterview(candidate)}
                    >
                      Start Assessment <ChevronRight className="h-4 w-4 ml-2 group-hover/btn:translate-x-1 transition-transform" />
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>

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
                    className="w-full h-10 px-3 py-2 rounded-md border border-input text-sm"
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

            {completedAssessments.length > 0 && (
              <div className="mt-12 animate-in slide-in-from-bottom-4 duration-700">
                <h2 className="text-xl font-black text-[#1A365D] mb-6 flex items-center gap-2">
                  <ClipboardCheck className="h-6 w-6 text-[#D4AF37]" /> Completed Remote Assessments
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {completedAssessments.map(item => (
                    <Card key={item.id} className="p-6 border-slate-100 hover:shadow-xl transition-all group">
                      <div className="flex justify-between items-start mb-4">
                        <div className="h-10 w-10 bg-[#1A365D]/5 text-[#1A365D] rounded-lg flex items-center justify-center font-bold">
                          {item.candidate_name.charAt(0)}
                        </div>
                        <Badge variant="outline" className="bg-[#48BB78]/10 text-[#48BB78] border-[#48BB78]/20">
                          {item.status === 'evaluated' ? 'Ready to Review' : 'Processing AI...'}
                        </Badge>
                      </div>
                      <h3 className="font-bold text-slate-800">{item.candidate_name}</h3>
                      <p className="text-xs text-slate-400 mb-6">Completed: {new Date(item.submitted_at).toLocaleDateString()}</p>
                      <Button onClick={() => handleReviewAssessment(item.id)} className="w-full bg-[#1A365D]" size="sm">
                        View Responses
                      </Button>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {view === 'review' && reviewAssessment && (
          <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
            <div className="flex items-center justify-between">
              <div>
                <Button variant="ghost" className="pl-0 text-slate-400 hover:text-[#1A365D]" onClick={() => setView('dashboard')}>
                  <ChevronLeft className="mr-2 h-4 w-4" /> Assessment Hub
                </Button>
                <h1 className="text-3xl font-black text-[#1A365D] mt-4">Review: {reviewAssessment.candidate.name}</h1>
              </div>
              <Badge className="bg-[#D4AF37] px-4 py-2 text-md">Remote Assessment</Badge>
            </div>

            <div className="space-y-6">
              {reviewAssessment.candidate_answers && reviewAssessment.candidate_answers.map((ans, idx) => {
                const aiEval = reviewAssessment.ai_evaluations?.[ans.question_id];
                return (
                  <Card key={ans.question_id || idx} className="p-8 border-l-4 border-[#1A365D] overflow-hidden">
                    <div className="flex justify-between items-start mb-6">
                      <div className="space-y-1">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Question {idx + 1}</span>
                        <h3 className="text-xl font-bold text-slate-800">{ans.question_text}</h3>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8 pt-8 border-t border-slate-50">
                      <div className="bg-slate-50 p-6 rounded-2xl">
                        <h4 className="text-xs font-bold text-slate-400 uppercase mb-4">Candidate Answer</h4>
                        <p className="text-slate-700 italic leading-relaxed">"{ans.transcript}"</p>
                      </div>

                      <div className="bg-[#1A365D]/5 p-6 rounded-2xl border border-[#1A365D]/10">
                        <div className="flex justify-between items-center mb-4">
                          <h4 className="text-xs font-bold text-[#1A365D] uppercase flex items-center gap-1">
                            <Sparkles className="h-3 w-3" /> AI Evaluation
                          </h4>
                          {aiEval && (
                            <Badge className="bg-[#1A365D]">{aiEval.score} / 5</Badge>
                          )}
                        </div>
                        {aiEval ? (
                          <p className="text-slate-600 text-sm leading-relaxed">{aiEval.justification}</p>
                        ) : (
                          <div className="flex items-center gap-2 text-slate-400">
                            <div className="h-4 w-4 border-2 border-[#1A365D] border-t-transparent rounded-full animate-spin" />
                            Evaluating...
                          </div>
                        )}
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          </div>
        )}

        {view === 'interview' && (
          <div className="max-w-3xl mx-auto animate-in fade-in zoom-in-95 duration-500">
            <div className="mb-8 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-widest text-[#D4AF37]">
                    Assessing: {activeSession.candidate.name}
                  </h3>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[#1A365D] font-bold text-xs">PILLAR: {activeSession.questions[currentQuestionIdx].pillar}</span>
                  </div>
                </div>
                <div className="text-right flex items-center gap-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-[10px] font-bold text-[#D4AF37] hover:bg-[#D4AF37]/10"
                    onClick={generateAiQuestion}
                    disabled={isAiGenerating}
                  >
                    {isAiGenerating ? 'Generating...' : '✨ Smart Generate'}
                  </Button>
                  <span className="text-sm font-bold text-slate-400">
                    Step {currentQuestionIdx + 1} of {activeSession.questions.length}
                  </span>
                </div>
              </div>
              <div className="h-1.5 w-full bg-slate-200 rounded-full">
                <div
                  className="h-full bg-[#1A365D] transition-all duration-500"
                  style={{ width: `${((currentQuestionIdx + 1) / activeSession.questions.length) * 100}%` }}
                />
              </div>
            </div>

            <Card className="border-t-4 border-t-[#D4AF37] shadow-xl overflow-hidden">
              <div className="p-8 sm:p-12 space-y-8">
                <div className="flex items-center gap-2">
                  <span className="bg-[#1A365D]/10 text-[#1A365D] text-[10px] font-bold px-2 py-1 rounded">
                    {activeSession.questions[currentQuestionIdx].category}
                  </span>
                </div>
                <h2 className="text-2xl sm:text-3xl font-bold text-slate-800 leading-tight">
                  {activeSession.questions[currentQuestionIdx].text}
                </h2>

                <div className="pt-8 border-t space-y-6">
                  <div>
                    <p className="text-sm font-bold text-slate-500 uppercase tracking-wide mb-3">Candidate Answer Transcript</p>
                    <textarea
                      className="w-full p-4 rounded-xl border-2 border-slate-100 focus:border-[#1A365D] focus:ring-0 transition-all text-sm min-h-[120px]"
                      placeholder="Type the candidate's answer here for AI evaluation..."
                      value={transcript}
                      onChange={(e) => setTranscript(e.target.value)}
                    />
                    <div className="mt-3 flex justify-end">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleAiEvaluate}
                        disabled={isEvaluating || !transcript}
                        className="text-xs font-bold gap-2"
                      >
                        {isEvaluating ? 'Evaluating...' : '✨ Get AI Score Suggestion'}
                      </Button>
                    </div>
                  </div>

                  {aiEvaluation && (
                    <div className="bg-[#1A365D]/5 border border-[#1A365D]/10 rounded-xl p-6 animate-in fade-in slide-in-from-top-4">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="font-bold text-[#1A365D] text-sm flex items-center gap-2">
                          <Award className="h-4 w-4" /> AI Suggestion
                        </h4>
                        <div className="bg-[#1A365D] text-white px-3 py-1 rounded-full text-xs font-bold">
                          Score: {aiEvaluation.score} / 5
                        </div>
                      </div>
                      <p className="text-sm text-slate-600 leading-relaxed italic">
                        "{aiEvaluation.justification}"
                      </p>
                      <div className="mt-4 pt-4 border-t border-[#1A365D]/10 text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                        Based on STAR Methodology
                      </div>
                    </div>
                  )}

                  {activeSession.questions[currentQuestionIdx].type === 'rating' ? (
                    <div className="space-y-6">
                      <p className="text-sm font-bold text-slate-500 uppercase tracking-wide">Final Numeric Rating</p>
                      <div className="flex justify-between max-w-sm mx-auto">
                        {[1, 2, 3, 4, 5].map(val => (
                          <button
                            key={val}
                            onClick={() => handleScore(val)}
                            className={cn(
                              "group flex flex-col items-center gap-2",
                              aiEvaluation?.score === val && "scale-110"
                            )}
                          >
                            <div className={cn(
                              "h-12 w-12 sm:h-14 sm:w-14 rounded-full border-2 flex items-center justify-center font-bold text-lg hover:border-[#D4AF37] hover:bg-[#D4AF37]/5 transition-all",
                              aiEvaluation?.score === val ? "border-[#D4AF37] bg-[#D4AF37]/10" : "border-slate-200"
                            )}>
                              {val}
                            </div>
                            <span className={cn(
                              "text-[10px] uppercase font-bold group-hover:text-[#D4AF37]",
                              aiEvaluation?.score === val ? "text-[#D4AF37]" : "text-slate-400"
                            )}>
                              {val === 1 ? 'Poor' : val === 5 ? 'Elite' : ''}
                              {aiEvaluation?.score === val && ' (AI Rec)'}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <p className="text-sm font-bold text-slate-500 uppercase tracking-wide">Select the best description</p>
                      <div className="grid grid-cols-1 gap-3">
                        {activeSession.questions[currentQuestionIdx].options.map((opt, i) => (
                          <button
                            key={i}
                            onClick={() => handleScore(opt.value)}
                            className={cn(
                              "flex items-center justify-between p-4 rounded-lg border-2 hover:border-[#D4AF37] hover:bg-slate-50 transition-all text-left group",
                              aiEvaluation?.score === opt.value ? "border-[#D4AF37] bg-slate-50" : "border-slate-100"
                            )}
                          >
                            <span className="text-sm font-medium pr-4">{opt.label}</span>
                            <div className={cn(
                              "h-5 w-5 rounded-full border-2 group-hover:border-[#D4AF37] shrink-0",
                              aiEvaluation?.score === opt.value ? "border-[#D4AF37] bg-[#D4AF37]" : "border-slate-200"
                            )} />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>

            <div className="mt-8 flex justify-between items-center">
              <Button
                variant="ghost"
                onClick={() => setCurrentQuestionIdx(Math.max(0, currentQuestionIdx - 1))}
                disabled={currentQuestionIdx === 0}
              >
                <ChevronLeft className="mr-2 h-4 w-4" /> Go Back
              </Button>
              <p className="text-xs text-slate-400 italic">Scores are saved automatically on selection</p>
            </div>
          </div>
        )}

        {view === 'summary' && (
          <div className="max-w-5xl mx-auto animate-in fade-in slide-in-from-bottom-12 duration-700">
            <div className="bg-white rounded-3xl p-8 sm:p-12 shadow-2xl border border-slate-100">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 gap-6">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <CheckCircle2 className="text-[#48BB78] h-6 w-6" />
                    <span className="text-sm font-bold text-[#48BB78] uppercase tracking-widest">Assessment Complete</span>
                  </div>
                  <h2 className="text-4xl font-black text-[#1A365D] tracking-tight">{activeSession.candidate.name}</h2>
                  <p className="text-slate-400 font-medium italic mt-1">STAR Method Breakdown</p>
                </div>
                <div className="flex gap-4">
                  <Button
                    onClick={handleDownloadPdf}
                    variant="outline"
                    size="lg"
                    disabled={isDownloading}
                    className="border-[#1A365D] text-[#1A365D]"
                  >
                    {isDownloading ? 'Capturing Report...' : 'Download PDF Report'}
                  </Button>
                  <Button onClick={() => setView('dashboard')} variant="accent" size="lg">Return to Pipeline</Button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                <div className="space-y-8">
                  {chartData.map(data => (
                    <div key={data.subject} className="group">
                      <div className="flex justify-between items-end mb-2">
                        <div>
                          <span className="block text-[10px] font-bold text-[#D4AF37] uppercase tracking-widest">Pillar</span>
                          <span className="text-xl font-bold text-[#1A365D]">{data.subject}</span>
                        </div>
                        <span className="text-2xl font-black text-[#1A365D]">{data.A.toFixed(1)} <span className="text-xs text-slate-300">/ 5.0</span></span>
                      </div>
                      <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-[#1A365D] to-[#2A4365] rounded-full transition-all duration-1000"
                          style={{ width: `${(data.A / 5) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="bg-slate-50/50 rounded-3xl p-4 sm:p-8 flex flex-col items-center">
                  <h3 className="text-sm font-bold text-[#1A365D] uppercase tracking-widest mb-8">Competency Spider Map</h3>
                  <div className="w-full h-[300px] sm:h-[400px]" ref={chartRef}>
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                        <PolarGrid stroke="#cbd5e1" />
                        <PolarAngleAxis dataKey="subject" tick={{ fill: '#1A365D', fontSize: 12, fontWeight: 800 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 5]} tick={false} axisLine={false} />
                        <Radar
                          name="Candidate"
                          dataKey="A"
                          stroke="#D4AF37"
                          strokeWidth={4}
                          fill="#D4AF37"
                          fillOpacity={0.4}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="mt-12 pt-12 border-t grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-slate-50 rounded-2xl text-center">
                  <span className="block text-[10px] font-bold text-slate-400 uppercase">Overall Index</span>
                  <span className="text-2xl font-black text-[#1A365D]">
                    {(chartData.reduce((a, b) => a + b.A, 0) / chartData.length).toFixed(1)}
                  </span>
                </div>
                {/* Add more summary stats if needed */}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
