from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Import and include routers
from api.routes import generation, questions, sessions, evaluation, candidate, reports
from api.websockets import websocket_endpoint
app.include_router(generation.router, prefix="/api/v1", tags=["generation"])
app.include_router(questions.router, prefix="/api/v1", tags=["questions"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(candidate.router, prefix="/api/v1", tags=["candidate"])
app.include_router(reports.router, prefix="/api/v1", tags=["reports"])

@app.websocket("/api/v1/ws/session/{session_id}")
async def websocket_route(websocket: WebSocket, session_id: str):
    await websocket_endpoint(websocket, session_id)
