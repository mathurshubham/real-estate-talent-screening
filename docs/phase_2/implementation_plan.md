# Phase 2 Feature Expansion Implementation Plan

This document outlines the detailed plan to implement the Phase 2 Feature Expansion, including restructuring the project into a Turborepo, adding AI answer evaluation, creating a candidate portal, adding multi-panelist websocket support, and enabling PDF generation.

## 1. Turborepo Migration & Directory Restructuring

**Goal**: Organize the codebase into a monorepo using Turborepo to ease workspace management and builds.

### Proposed Changes
- **Root Directory**:
  - `package.json`: Initialize a root `package.json` with npm workspaces (`"workspaces": ["frontend", "backend"]`) and add `turbo` as a dev dependency.
  - `turbo.json`: Add Turborepo configuration to define the pipeline for `dev`, `build`, and `lint`.
- **Frontend Move**:
  - Move the following files and folders from the root to a new `frontend/` directory: `src/`, `public/`, `index.html`, `vite.config.js`, `tailwind.config.js`, `postcss.config.js`, `eslint.config.js`, and `package.json`.
  - Update `frontend/package.json` name to something like `"@estateassess/frontend"`.
- **Backend Updates**:
  - Add a `package.json` to `backend/` with a script to wrap Python execution (e.g. `"dev": "uv run uvicorn main:app --reload"`), allowing `turbo run dev` to orchestrate both.
- **Docker Compose**:
  - Update `docker-compose.yml` frontend build context from `.` to `./frontend`.

---

## 2. AI-Driven Answer Evaluation

**Goal**: Allow panelists to input candidate answers (transcripts) and get an AI-evaluated score (1-5) and justification based on the STAR framework.

### Proposed Changes
- **Backend**:
  - `backend/api/routes/evaluation.py` [NEW]: Create a new router for evaluation endpoints.
  - `POST /api/evaluate/answer`: Accepts `question_context` and `candidate_transcript`.
  - Utilize `google-genai` to prompt Gemini. The prompt will enforce the STAR (Situation, Task, Action, Result) framework. We will use Structured Outputs (Pydantic models) to ensure the AI always returns `score` (int 1-5) and `justification` (string).
- **Frontend**:
  - Add a text area in the active interview screen for panelists to type the candidate's transcript.
  - Add a "Generate AI Evaluation" button that calls the `/api/evaluate/answer` endpoint and displays the suggested score and justification inline.

---

## 3. Asynchronous Candidate Portal

**Goal**: A timed portal where candidates can take assessments synchronously without a panelist.

### Proposed Changes
- **Frontend**:
  - `src/pages/CandidatePortal.jsx` [NEW]: A new route `/candidate/assessment/:sessionId`.
  - A staging screen to verify identity/instructions.
  - An active assessment screen that displays the pre-generated questions one by one with a countdown timer.
  - Textareas for written answers. Upon submission, it calls the backend assessment submission API.
- **Backend**:
  - Endpoints to fetch a session's questions for the candidate.
  - `POST /api/sessions/{session_id}/submit`: Accepts all written answers, triggers async background tasks (using FastAPI `BackgroundTasks`) to pre-evaluate each answer using the AI evaluation tool, and stores the results and transcripts in the DB.

---

## 4. Multi-Panelist Collaboration (WebSockets)

**Goal**: Allow multiple stakeholders to join the same assessment, score secretly, and average scores on the Radar chart.

### Proposed Changes
- **Backend**:
  - `backend/api/websockets.py` [NEW]: Implement a `ConnectionManager` class to handle active WebSocket connections locally.
  - **Robustness**: Include a heartbeat (ping/pong) mechanism on the server and use it to prune stale connections.
  - Since we have Redis, we will use Redis Pub/Sub to broadcast WebSocket messages across multiple FastAPI workers if needed.
  - `WebSocket /api/ws/session/{session_id}`: Route for panelists to join a session.
  - Event schemas: `SCORE_UPDATE`, `STATE_SYNC`.
- **Frontend**:
  - Use native WebSocket or a library like `react-use-websocket` in the Interview session page.
  - **Resilience**: Implement automatic reconnection logic to handle panelist dropout or tab switching.
  - **Race Condition Handling**: Include a `timestamp` or `version` flag with `SCORE_UPDATE` events. The frontend and backend will use these to ensure the latest update is always honored, even if events arrive out of order.
  - When Panelist A updates a score, broadcast the `SCORE_UPDATE`. The UI will obscure other panelists' scores during the interview, but the Dashboard/Summary page will receive the aggregated data.
  - Update the Radar chart logic (which uses `recharts`) to display the average of all panelist scores.

---

## 5. PDF Report Generation

**Goal**: Generate a branded, downloadable PDF report.

### Proposed Changes
- **Backend**:
  - Add `reportlab` to `backend/pyproject.toml`.
  - `GET /api/reports/{session_id}/pdf` [NEW]: Endpoint to generate PDF.
  - The PDF will contain: The candidate's info, average score, the spider chart, questions, transcripts, and panelist notes.
  - *Note on Chart*: Rendering the `recharts` radar chart in Python natively is difficult. To include the chart in the PDF, the frontend summary page will capture the `recharts` SVG or Canvas as a Base64 image, and POST it to the backend `POST /api/reports/{session_id}/save-chart`. The backend temporarily saves this image for the PDF generation.
- **Frontend**:
  - Add "Download PDF" button to Summary Screen. Use `html-to-image` to extract the Radar Chart as a Base64 image.
  - **Quality Optimization**: Configure `html-to-image` to scale the capture by 3x to ensure the chart looks crisp and professional in the final PDF report.
  - POST the scaled image to the backend and then initiate the download.

---

## Verification & Testing Plan

### Automated Tests
1. **Backend Tests** (`pytest`):
   - Unit tests for Gemini prompt logic and evaluation schema parsing.
   - Integration tests for candidate submission flow and async evaluation triggering.
   - Mocked WebSocket client tests to verify broadcast and heartbeat logic.
2. **Frontend Tests**:
   - Component tests for the new AI evaluation UI.
   - Integration tests for the candidate portal navigation and assessment timer.
   - Reconnection logic tests for WebSockets.

### Browser Agent Verification
The browser agent will be used to perform end-to-end validation of all new features:
1. **Refactor Verification**: Verify `turbo run dev` starts the full stack correctly.
2. **Evaluation Flow**: Type a transcript, trigger AI scoring, and verify the resulting UI update.
3. **Assessment Flow**: Complete a candidate assessment end-to-end and see results appear in the panelist dashboard.
4. **Collaboration Sync**: Use the agent to simulate multi-user behavior and verify WebSocket sync.
5. **PDF Check**: Trigger a PDF generation and verify the file content (where possible).

### Manual Verification
1. Click "Download PDF Report", verify the Spider chart, the average scores, and notes are cleanly formatted in the downloaded file.
