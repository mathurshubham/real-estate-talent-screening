from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
from db.session import get_db
from db.models import QuestionBank
from core.redis import get_redis
import redis.asyncio as redis

router = APIRouter()

@router.get("/questions")
async def get_questions(
    source: str = "standard", 
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis)
):
    cache_key = f"questions:{source}"
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    result = await db.execute(select(QuestionBank).where(QuestionBank.source == source))
    questions = result.scalars().all()
    
    # Convert to serializable format
    questions_list = [{"id": q.id, "text": q.question_text, "category": q.category} for q in questions]
    
    await r.set(cache_key, json.dumps(questions_list), ex=3600) # Cache for 1 hour
    return questions_list
