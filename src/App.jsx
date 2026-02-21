import React, { useState, useMemo } from 'react';
import { GoogleGenerativeAI } from "@google/generative-ai";
import {
  Users, Plus, LogIn, ChevronRight, ChevronLeft,
  Award, ClipboardCheck, Briefcase, CheckCircle2
} from 'lucide-react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer
} from 'recharts';
import {
  Button, Card, CardHeader, CardTitle, CardContent, Input, Modal
} from './components/ui';
import { QUESTION_LIBRARY, ROLE_TEMPLATES, ASSESSMENT_PILLARS } from './data/assessmentData';
import KAGGLE_QUESTIONS from './data/kaggleQuestions.json';
import { cn } from './lib/utils';

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

  const handleLogin = (e) => {
    e.preventDefault();
    setUser({ name: 'Panelist Lead' });
    setView('dashboard');
  };

  const addCandidate = () => {
    if (!newCandidate.name) return;
    setCandidates([...candidates, { ...newCandidate, id: Date.now(), status: 'pending' }]);
    setIsAddModalOpen(false);
    setNewCandidate({ name: '', role: 'specialist' });
  };

  const startInterview = (candidate) => {
    let questions = [];
    if (useKaggle) {
      // Pick 10 random questions from Kaggle library
      questions = [...KAGGLE_QUESTIONS].sort(() => 0.5 - Math.random()).slice(0, 10);
    } else {
      const roleData = ROLE_TEMPLATES.find(r => r.id === candidate.role);
      questions = roleData.groups.flatMap(group =>
        QUESTION_LIBRARY[group].map(q => ({ ...q, pillar: group }))
      );
    }

    setActiveSession({ candidate, questions });
    setCurrentQuestionIdx(0);
    setScores({});
    setView('interview');
  };

  const generateAiQuestion = async () => {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      console.error("Gemini API Key missing. Please ensure VITE_GEMINI_API_KEY is set in your .env file.");
      return;
    }

    setIsAiGenerating(true);
    try {
      const genAI = new GoogleGenerativeAI(apiKey);

      // Integrating Gemini 3.0 Flash Preview for advanced reasoning
      const model = genAI.getGenerativeModel({
        model: "gemini-3-flash-preview",
        // Note: Thinking level is handled by the model family by default
      });

      const currentPillar = activeSession.questions[currentQuestionIdx].pillar;
      const candidateRole = ROLE_TEMPLATES.find(r => r.id === activeSession.candidate.role).name;
      const currentCategory = activeSession.questions[currentQuestionIdx].category;
      const currentQuestionText = activeSession.questions[currentQuestionIdx].text;

      const prompt = `
        As a professional Real Estate Recruitment Specialist, generate ONE highly relevant, probing interview question for the "${currentPillar}" category.
        
        CONTEXT:
        - Candidate: ${activeSession.candidate.name}
        - Role: ${candidateRole}
        - Current Topic: ${currentCategory}
        - Last Question Asked: "${currentQuestionText}"
        
        INSTRUCTIONS:
        - The question should be sophisticated and specific to high-stakes real estate.
        - Return ONLY the question text. 
        - Do not include prefixes like "Question:" or any other commentary.
      `;

      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      if (!text) throw new Error("Empty response from AI");

      const aiQuestion = {
        id: `AI-${Date.now()}`,
        text: `[Gemini Insight] ${text.trim()}`,
        type: 'rating',
        pillar: currentPillar,
        category: 'AI Generated'
      };

      const newQuestions = [...activeSession.questions];
      newQuestions.splice(currentQuestionIdx + 1, 0, aiQuestion);
      setActiveSession({ ...activeSession, questions: newQuestions });
    } catch (error) {
      console.error("Gemini AI Integration Error:", error);
      // Fallback if the user's specific endpoint/key has issues with flash
      if (error.message?.includes("404")) {
        console.warn("Model not found. Ensure your API key has access to gemini-1.5-flash.");
      }
    } finally {
      setIsAiGenerating(false);
    }
  };

  const handleScore = (value) => {
    setScores({ ...scores, [activeSession.questions[currentQuestionIdx].id]: value });
    if (currentQuestionIdx < activeSession.questions.length - 1) {
      setCurrentQuestionIdx(currentQuestionIdx + 1);
    } else {
      setView('summary');
    }
  };

  const chartData = useMemo(() => {
    if (!activeSession) return [];
    return ASSESSMENT_PILLARS.map(pillar => {
      const pillarQuestions = activeSession.questions.filter(q => q.pillar === pillar);
      const pillarScores = pillarQuestions.map(q => scores[q.id] || 0);
      const average = pillarScores.length > 0
        ? pillarScores.reduce((a, b) => a + b, 0) / pillarScores.length
        : 0;
      return { subject: pillar, A: average, fullMark: 5 };
    });
  }, [activeSession, scores]);

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

                <div className="pt-8 border-t">
                  {activeSession.questions[currentQuestionIdx].type === 'rating' ? (
                    <div className="space-y-6">
                      <p className="text-sm font-bold text-slate-500 uppercase tracking-wide">Numeric Rating (Standard Scale)</p>
                      <div className="flex justify-between max-w-sm mx-auto">
                        {[1, 2, 3, 4, 5].map(val => (
                          <button
                            key={val}
                            onClick={() => handleScore(val)}
                            className="group flex flex-col items-center gap-2"
                          >
                            <div className="h-12 w-12 sm:h-14 sm:w-14 rounded-full border-2 border-slate-200 flex items-center justify-center font-bold text-lg hover:border-[#D4AF37] hover:bg-[#D4AF37]/5 transition-all">
                              {val}
                            </div>
                            <span className="text-[10px] uppercase text-slate-400 font-bold group-hover:text-[#D4AF37]">
                              {val === 1 ? 'Poor' : val === 5 ? 'Elite' : ''}
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
                            className="flex items-center justify-between p-4 rounded-lg border-2 border-slate-100 hover:border-[#D4AF37] hover:bg-slate-50 transition-all text-left group"
                          >
                            <span className="text-sm font-medium pr-4">{opt.label}</span>
                            <div className="h-5 w-5 rounded-full border-2 border-slate-200 group-hover:border-[#D4AF37] shrink-0" />
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
                <Button onClick={() => setView('dashboard')} variant="accent" size="lg">Return to Pipeline</Button>
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
                  <div className="w-full h-[300px] sm:h-[400px]">
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
