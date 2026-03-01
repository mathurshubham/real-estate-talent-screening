# EstateAssess STAR 🛡️

A premium, AI-powered recruitment assessment platform designed specifically for high-stakes real estate candidate evaluation using the **STAR Methodology** (Skill, Training, Attitude, Results).

## 🌟 Key Features

- **STAR Assessment Engine**: A mixed-modality interview flow supporting both numeric ratings and descriptive MCQs.
- **AI-Driven Evaluation**: Dedicated backend using Gemini 2.5 Flash Lite for automated answer analysis and scoring.
- **Asynchronous Candidate Portal**: Remote assessment platform for candidates to complete screenings independently.
- **Multi-Panelist Collaboration**: Real-time session synchronization via WebSockets for collaborative evaluations.
- **PDF Report Generation**: Professional, high-resolution competence reports with dynamic radar charts.
- **Visual Analytics**: Real-time Radar (Spider) chart integration to visualize candidate competency across all four STAR pillars.

## 🛠️ Technology Stack

- **Frontend**: React 19 + Vite + Recharts
- **Backend**: FastAPI + Python 3.12 + ReportLab
- **Database**: PostgreSQL
- **Cache/Sync**: Redis
- **AI**: Google Generative AI SDK (Gemini 2.5 Flash Lite)
- **Infrastructure**: Docker Compose + Nginx

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Google AI Studio API Key ([Get one here](https://aistudio.google.com/))

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mathurshubham/real-estate-talent-screening.git
   cd real-estate-talent-screening
   ```

2. **Environment Setup**:
   Copy `.env.example` to `.env` and fill in your Gemini API Key:
   ```bash
   cp .env.example .env
   # Edit .env and set VITE_GEMINI_API_KEY
   ```

3. **Run with Docker Compose**:
   ```bash
   docker compose up --build -d
   ```

### 🌐 Accessing the Application

After starting the containers, the application is available at:
- **Frontend**: [http://localhost:8200](http://localhost:8200)
- **Backend API**: [http://localhost:8201/api/v1](http://localhost:8201/api/v1)
- **Database (Postgres)**: `localhost:8202`
- **Cache (Redis)**: `localhost:8203`

## 📁 Project Structure

- `/frontend`: React application (Vite-based).
- `/backend`: FastAPI application and AI logic.
- `/docs/phase_2`: Implementation plan, progress reports, and walkthroughs for the latest features.

## 📄 Documentation

For a detailed user guide and feature overview, see the [User Handbook](docs/guide.md).
