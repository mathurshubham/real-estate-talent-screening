from pydantic import BaseModel
from google import genai
from fastapi import APIRouter, Depends, HTTPException, Request
from core.config import settings
from core.redis import get_redis
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

client = genai.Client(api_key=settings.GEMINI_API_KEY)

class EvaluationRequest(BaseModel):
    question_context: str
    candidate_transcript: str

class EvaluationResponse(BaseModel):
    score: int
    justification: str

@router.post("/evaluate", response_model=EvaluationResponse)
@limiter.limit("5/minute")
async def evaluate_answer(request: Request, eval_req: EvaluationRequest):
    try:
        prompt = f"""
        As an expert real estate talent screener, evaluate the following candidate's answer based on the STAR (Situation, Task, Action, Result) framework.
        
        Question Context: {eval_req.question_context}
        Candidate Transcript: {eval_req.candidate_transcript}
        
        Provide a score from 1 to 5 and a brief justification.
        1: Poor - No STAR elements present, irrelevant answer.
        2: Developing - Some elements present but lacks clarity or depth.
        3: Proficient - Most elements present, clear answer.
        4: Advanced - Well-structured STAR response with strong results.
        5: Expert - Perfect STAR structure, exceptional actions and quantifiable results.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': EvaluationResponse.model_json_schema()
            }
        )
        
        # The new genai SDK returns the result in response.text or response.parsed depending on version
	    # For this specific SDK, we often need to parse the JSON from .text if parsed is not available or reliable
        import json
        result = json.loads(response.text)
        return EvaluationResponse(**result)

    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
