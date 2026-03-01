import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
import json
import io
import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from core.redis import get_redis

@pytest.fixture
def mock_redis():
    return AsyncMock()

@pytest.mark.asyncio
async def test_candidate_portal_flow(mock_redis):
    # Mock data
    mock_redis.get.return_value = json.dumps({
        "candidate": {"name": "Test User"},
        "questions": [{"id": "q1", "text": "Question 1"}],
        "status": "pending"
    })
    
    # Override dependency
    async def override_get_redis():
        yield mock_redis
    
    app.dependency_overrides[get_redis] = override_get_redis
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Fetch portal data
        response = await ac.get("/api/v1/candidate/assessment/TEST-123")
        assert response.status_code == 200
        assert response.json()["candidate_name"] == "Test User"
        
        # Submit assessment
        with patch("api.routes.candidate.evaluate_all_candidate_answers") as mock_eval_task:
            response = await ac.post(
                "/api/v1/candidate/assessment/TEST-123/submit",
                json={
                    "answers": [
                        {"question_id": "q1", "question_text": "Question 1", "transcript": "My answer"}
                    ]
                }
            )
            assert response.status_code == 200
    
    del app.dependency_overrides[get_redis]

@pytest.mark.asyncio
async def test_completed_assessments_list(mock_redis):
    mock_redis.keys.return_value = ["session:TEST-123"]
    mock_redis.get.return_value = json.dumps({
        "status": "evaluated",
        "candidate": {"name": "Test Candidate"},
        "submitted_at": "2026-03-01T00:00:00Z"
    })
    
    async def override_get_redis():
        yield mock_redis
    app.dependency_overrides[get_redis] = override_get_redis
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/v1/candidate/assessments/completed")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["candidate_name"] == "Test Candidate"
    
    del app.dependency_overrides[get_redis]

@pytest.mark.asyncio
async def test_pdf_generation(mock_redis):
    mock_redis.get.side_effect = [
        json.dumps({
            "candidate": {"name": "Test User", "role": "manager"},
            "questions": [{"id": "q1", "text": "Q1"}],
            "scores": {"q1": 4},
            "status": "evaluated"
        }),
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    ]
    
    async def override_get_redis():
        yield mock_redis
    app.dependency_overrides[get_redis] = override_get_redis
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/v1/sessions/TEST-123/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
    
    del app.dependency_overrides[get_redis]
