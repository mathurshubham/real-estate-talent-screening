# EstateAssess User Handbook

Welcome to the EstateAssess STAR Recruitment Handbook. This guide provides an overview of the platform's features and instructions on how to effectively use it for talent screening.

## 📋 Feature Overview

### 1. Panelist Assessment Hub
The central dashboard for recruiters. From here, you can:
- **Start New Assessments**: Initiate live sessions for candidates in the pipeline.
- **Resume In-Progress**: Pick up where you left off in an active interview.
- **View Remote Results**: Review assessments completed by candidates through the asynchronous portal.

### 2. AI-Driven STAR Evaluation
During a live assessment, panelists can:
- Enter a rough **Candidate Answer Transcript**.
- Click **"Get AI Score Suggestion"** to receive a score (1-5) and a justification based on the STAR methodology (Situation, Task, Action, Result).
- Use these suggestions as a baseline or override them with their own expert rating.

### 3. Asynchronous Candidate Portal
Automate the initial screening phase:
- Candidates receive a unique access link (e.g., `?accessKey=CANDIDATE-ID`).
- They complete the assessment at their own pace within a set time limit for each question.
- Responses are automatically sent to the backend for background AI evaluation.

### 4. Real-Time Multi-Panelist Sync
For panel interviews with multiple evaluators:
- All panelists joined to the same session see the radar chart update in real-time as scores are submitted.
- The platform automatically averages scores across all panelists to provide a fair, balanced final result.

### 5. High-Resolution PDF Reporting
Generate professional candidate profiles with one click:
- Includes the full STAR breakdown with pillar-specific scores.
- Features a high-resolution competency spider map.
- Provides a comprehensive overview for decision-making and archival purposes.

## 🚀 How to Use

### For Panelists (Live Interviews)
1. Navigate to the **Assessment Hub**.
2. Select a candidate and click **"Start Assessment"**.
3. For each question, type the candidate's core answer into the transcript box.
4. Click **"Get AI Score Suggestion"** to assist your rating.
5. Select a final score (1-5) to proceed.
6. Upon completion, review the **Competency Spider Map** and download the PDF report.

### For Candidates (Remote Screening)
1. Open the provided assessment link.
2. Read the instructions and click **"Start Assessment"**.
3. Answer each question clearly, following the STAR framework for better results.
4. Monitor the timer to ensure all questions are submitted before they expire.
5. Click **"Submit Assessment"** at the final step.

## 🛠️ Configuration & Deployment

### Environment Variables
The application relies on a `.env` file for configuration:
- `VITE_GEMINI_API_KEY`: Required for AI evaluations.
- `VITE_API_BASE_URL`: Defines the backend endpoint for the frontend.
- `DATABASE_URL` & `REDIS_URL`: Connection strings for infrastructure services.

### Port Mappings (Default)
- **Frontend**: 8200
- **Backend**: 8201
- **Postgres**: 8202
- **Redis**: 8203

---
*EstateAssess - Precision Recruitment for Real Estate.*
