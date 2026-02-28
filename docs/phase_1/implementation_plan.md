# Phase 1: Fullstack Architecture & Security Fixes

This phase focuses on migrating the AI generation logic to a secure Python backend, implementing PostgreSQL for persistent storage, and using Redis to enhance performance and manage rate limits.

## User Review Required

> [!CAUTION]
> The AI logic is moving to the backend. The frontend will no longer use the Google Generative AI SDK directly.

> [!IMPORTANT]
> A PostgreSQL database and a Redis server will be required to run the application locally or in production. We will use `uv` for the Python package management and FastAPI for the backend framework. We will use SQLAlchemy for the ORM to interact with PostgreSQL.

## Proposed Changes

### Infrastructure & Dockerization

#### [NEW] devinfra/docker-compose.yml
Provision persistent local infrastructure:
- **PostgreSQL**: Internal port 5432, persistent volume for data.
- **Redis**: Internal port 6379, persistent volume for data.

#### [NEW] backend/Dockerfile
Multi-stage build using `python:3.12-slim` and `uv` to install dependencies and run the FastAPI server.

#### [NEW] Dockerfile (Frontend)
Multi-stage build:
1.  **Build**: `node:20-alpine` to build the Vite app.
2.  **Serve**: `nginx:alpine` to serve static files.

#### [NEW] docker-compose.yml (Root)
Orchestrate the entire stack:
- `backend` service (depends on `postgres` and `redis`).
- `frontend` service (depends on `backend`).
- Includes `/devinfra` services.

### Backend Setup (FastAPI & Python)

We will create a new `/backend` directory to house the FastApi application.

#### [NEW] backend/pyproject.toml
Use `uv` to initialize the project with dependencies: `fastapi`, `uvicorn`, `redis`, `sqlalchemy`, `asyncpg`, `google-generativeai`.

#### [NEW] backend/main.py
The entry point for the FastAPI application, configuring CORS to allow requests from the Vite frontend.

#### [NEW] backend/core/config.py
Setup environment variable parsing (using Pydantic `BaseSettings` or standard `os.environ`), including `GEMINI_API_KEY`, `DATABASE_URL`, and `REDIS_URL`.

#### [NEW] backend/api/routes/generation.py
Create the endpoint to handle the Gemini API requests securely from the backend, effectively replacing the frontend AI logic.

---

### Database Schema (PostgreSQL)

#### [NEW] backend/db/models.py
Define SQLAlchemy models for:
- `User` (Panelists, roles, orgs)
- `Candidate` (Details, status)
- `QuestionBank` (Standard library and Kaggle imports)
- `Assessment` (Link candidate to panelist with timestamp states)
- `Response` (Scores and AI-generated questions)

#### [NEW] backend/db/session.py
Configure the async SQLAlchemy engine and session maker connected to PostgreSQL.

---

### Redis Caching & Rate Limiting

#### [NEW] backend/core/redis.py
Setup the async Redis client.

#### [MODIFY] backend/api/routes/generation.py
Add rate-limiting logic (e.g., token bucket or simple fixed window) using Redis before calling the Gemini API.

#### [NEW] backend/api/routes/questions.py
Create endpoints to fetch the standard question bank and Kaggle questions, retrieving them from Redis cache if available, or reading/querying the database and populating the cache if missing.

#### [NEW] backend/api/routes/sessions.py
Endpoint to push/pull active interview states temporarily in Redis.

---

### Frontend Updates

#### [MODIFY] src/vite.config.js or .env
Remove `VITE_GEMINI_API_KEY`. Add `VITE_API_BASE_URL` pointing to the FastAPI backend (e.g., `http://localhost:8000`).

#### [MODIFY] src/services/api.js (or equivalent file)
Replace direct Gemini SDK calls with `fetch` or `axios` requests to the new FastAPI `/generate` endpoint.

#### [MODIFY] src/components/ui.jsx / Assessment flow
Update the Assessment components to send and retrieve state from the new backend API, replacing local state persistence where applicable.

## Verification Plan

### Automated Tests
- Test endpoints utilizing `pytest` and `httpx` to verify route handlers (Generation, DB operations, Caching).
- Ensure the rate limit acts correctly by writing a test that spams the endpoint and expects a 429 response.

### Manual Verification
- Start the backend server (`uvicorn`) and frontend (`vite`).
- Conduct a mock interview in the browser.
- Verify that AI Generation triggers the backend and returns a valid question without exposing the API key in network requests.
- Verify PostgreSQL contains the session data and the generated response.
- Restart the browser mid-session and verify Redis restores the interview state.
