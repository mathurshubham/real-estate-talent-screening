from fastapi import FastAPI
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
from api.routes import generation, questions, sessions
app.include_router(generation.router, prefix="/api/v1", tags=["generation"])
app.include_router(questions.router, prefix="/api/v1", tags=["questions"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
