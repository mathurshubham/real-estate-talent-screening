from google import genai
from fastapi import APIRouter, Depends, HTTPException, Request
from core.config import settings
from core.redis import get_redis
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

client = genai.Client(api_key=settings.GEMINI_API_KEY)

@router.post("/generate")
@limiter.limit("5/minute")
async def generate_question(request: Request, context: str):
    try:
        prompt = f"Given the following interview context: {context}, generate a relevant follow-up question following the STAR methodology."
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        return {"question": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
