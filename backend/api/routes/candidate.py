from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
from typing import List, Dict
import redis.asyncio as redis
from core.redis import get_redis
import json
from datetime import datetime
from api.routes.evaluation import evaluate_answer, EvaluationRequest
import uuid

router = APIRouter()

class AnswerSubmission(BaseModel):
    question_id: str
    question_text: str
    transcript: str

class CandidateSubmission(BaseModel):
    answers: List[AnswerSubmission]

async def evaluate_all_candidate_answers(session_id: str, answers: List[AnswerSubmission], r: redis.Redis):
    results = {}
    for ans in answers:
        try:
            # We reuse the logic from evaluate_answer
            # Note: In a real app, we'd pass a mock Request or refactor the logic to not depend on it
            # For simplicity here, we'll implement a direct call to the evaluation logic if possible, 
            # or just call Gemini directly here.
            
            # Since evaluate_answer in evaluation.py expects a Request (for rate limiting),
            # let's refactor the core evaluation logic into a service or just replicate the prompt here.
            
            # For now, let's assume we have a helper or just do it here to avoid dependency mess
            from api.routes.evaluation import client, EvaluationResponse
            
            prompt = f"""
            As an expert real estate talent screener, evaluate the following candidate's answer based on the STAR (Situation, Task, Action, Result) framework.
            
            Question Context: {ans.question_text}
            Candidate Transcript: {ans.transcript}
            
            Provide a score from 1 to 5 and a brief justification.
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': EvaluationResponse.model_json_schema()
                }
            )
            result = json.loads(response.text)
            results[ans.question_id] = result
        except Exception as e:
            print(f"Error evaluating answer {ans.question_id}: {e}")
            results[ans.question_id] = {"score": 0, "justification": f"Evaluation failed: {str(e)}"}

    # Save the evaluated results back to Redis
    existing_state = await r.get(f"session:{session_id}")
    if existing_state:
        state = json.loads(existing_state)
        state["ai_evaluations"] = results
        state["status"] = "evaluated"
        await r.set(f"session:{session_id}", json.dumps(state), ex=86400)

@router.get("/candidate/assessment/{access_key}")
async def get_candidate_assessment(access_key: str, r: redis.Redis = Depends(get_redis)):
    # In a real app, we would look up the access_key in the DB to find the session_id
    # For this sandbox, we'll assume access_key == session_id for now or prefix it
    session_id = access_key 
    state = await r.get(f"session:{session_id}")
    if state:
        full_state = json.loads(state)
        # Return only what the candidate needs
        return {
            "candidate_name": full_state.get("candidate", {}).get("name"),
            "questions": full_state.get("questions", []),
            "status": full_state.get("status", "pending")
        }
    raise HTTPException(status_code=404, detail="Assessment not found")

@router.post("/candidate/assessment/{access_key}/submit")
async def submit_candidate_assessment(
    access_key: str, 
    submission: CandidateSubmission, 
    background_tasks: BackgroundTasks,
    r: redis.Redis = Depends(get_redis)
):
    session_id = access_key
    state_raw = await r.get(f"session:{session_id}")
    if not state_raw:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    state = json.loads(state_raw)
    state["candidate_answers"] = [ans.model_dump() for ans in submission.answers]
    state["status"] = "submitted"
    state["submitted_at"] = datetime.utcnow().isoformat()
    await r.set(f"session:{session_id}", json.dumps(state), ex=86400)
    
    # Trigger background evaluation
    background_tasks.add_task(evaluate_all_candidate_answers, session_id, submission.answers, r)
    
    return {"status": "submitted", "message": "Your assessment has been received and is being evaluated."}

@router.get("/candidate/assessments/completed")
async def get_completed_assessments(r: redis.Redis = Depends(get_redis)):
    # In a real app, we'd query the SQL DB for assessments where status='submitted/evaluated'
    # For this sandbox, we'll scan Redis for session:*
    keys = await r.keys("session:*")
    completed = []
    for key in keys:
        key_str = key if isinstance(key, str) else key.decode()
        state_raw = await r.get(key_str)
        if state_raw:
            state = json.loads(state_raw)
            if state.get("status") in ["submitted", "evaluated"]:
                completed.append({
                    "id": key_str.split(":")[-1],
                    "candidate_name": state.get("candidate", {}).get("name"),
                    "submitted_at": state.get("submitted_at", "Unknown"),
                    "status": state.get("status")
                })
    return completed
