# Phase 2 Feature Expansion

## 1. Setup Turborepo & Refactor Directory Structure
- [x] Move frontend code (`src`, `public`, `vite.config.js`, `package.json`, etc.) into `frontend/` directory.
- [x] Initialize Turborepo at the root with a workspace configuration.
- [x] Update `docker-compose.yml` to reflect new `frontend` path.
- [x] Verify frontend and backend services can be started together with Turbo and Docker Compose.

## 2. AI-Driven Answer Evaluation
- [x] Add a new FastAPI endpoint (e.g., `POST /api/evaluate`) to accept candidate's answer and the question context.
- [x] Implement Gemini prompt using the STAR framework to score the answer (1-5) and generate a brief justification.
- [x] Update frontend panelist view to allow input of rough candidate transcripts.
- [x] Display AI evaluation result (score and justification) in the UI alongside the question.
- [x] Write backend unit tests for evaluation logic.
- [x] Write frontend unit/integration tests for evaluation display.
- [x] Verify functionality with Browser Agent.

## 3. Asynchronous Candidate Portal
- [x] Create simple authentication/login page for candidates (e.g., using shared room codes or candidate ID).
- [x] Build a timed candidate assessment interface displaying pre-generated questions.
- [x] Implement form submission for text answers (or recorded media if supported).
- [x] Add backend logic to asynchronously pre-evaluate submitted answers using the AI-evaluation endpoint.
- [x] Create a summary view for panelists to review completed async assessments.
- [x] Write backend unit tests for assessment submission and evaluation triggers.
- [x] Write frontend tests for the candidate assessment flow.
- [x] Verify portal flow with Browser Agent.

## 4. Multi-Panelist Collaboration
- [x] Incorporate WebSockets in FastAPI.
- [x] Create a WebSocket manager for real-time room/assessment sessions with heartbeat (ping/pong) mechanism and event versioning/timestamps.
- [x] Update frontend to connect to the session WebSocket with automatic reconnection logic and race condition handling.
- [x] Implement real-time secret scoring broadcasting to updating the shared session state.
- [x] Modify the comprehensive summary Radar chart to compute and display the average scores.
- [x] Write backend integration tests for WebSocket score synchronization.
- [x] Write frontend tests for real-time updates and reconnection.
- [x] Verify multi-user sync with Browser Agent.

## 5. PDF Report Generation
- [x] Add `reportlab` or `weasyprint` (or similar) to backend `pyproject.toml`.
- [x] Create a new FastAPI endpoint `GET /api/reports/{session_id}/pdf`.
- [x] Fetch session data (average scores, panelist notes, candidate answers).
- [x] Generate the PDF with the spider chart (captured at 3x scale for quality) and panelist notes.
- [x] Return the generated PDF file as a downloadable attachment.

## 6. Verification & Browser Testing
- [x] Verify Turborepo starts frontend/backend correctly.
- [x] Run all backend tests (`pytest`).
- [x] Run all frontend tests (`vitest` or similar).
- [x] Use Browser Agent to walk through:
    - [x] New evaluation flow.
    - [x] Candidate portal assessment.
    - [x] WebSocket sync between two "simulated" browser sessions.
    - [x] PDF generation and download.
