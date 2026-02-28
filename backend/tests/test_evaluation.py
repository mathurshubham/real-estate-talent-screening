import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add the backend directory to sys.path to import modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

def test_evaluate_answer_success():
    """Test successful AI evaluation of a candidate answer."""
    mock_response = MagicMock()
    mock_response.text = '{"score": 4, "justification": "Candidate clearly defined the Situation, Task, Action, and Result. Strong focus on quantifiable outcomes."}'
    
    with patch("api.routes.evaluation.client.models.generate_content", return_value=mock_response):
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question_context": "Tell me about a time you handled a difficult client.",
                "candidate_transcript": "I had a client who was upset about a delay. I researched the issue, called them back within an hour, explained the steps to fix it, and they ended up referring a friend."
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 4
        assert "STAR" in data["justification"]

def test_evaluate_answer_api_error():
    """Test backend handling of Gemini API errors."""
    with patch("api.routes.evaluation.client.models.generate_content", side_effect=Exception("Gemini API error")):
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question_context": "Any question",
                "candidate_transcript": "Any answer"
            }
        )
        assert response.status_code == 500
        assert "Gemini API error" in response.json()["detail"]

def test_evaluate_answer_invalid_request():
    """Test validation errors for missing payload fields."""
    response = client.post(
        "/api/v1/evaluate",
        json={"question_context": "Context without transcript"}
    )
    assert response.status_code == 422
