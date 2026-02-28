from fastapi import APIRouter, Depends, Body
from core.redis import get_redis
import redis.asyncio as redis
import json

router = APIRouter()

@router.post("/sessions/{session_id}")
async def save_session(session_id: str, state: dict = Body(...), r: redis.Redis = Depends(get_redis)):
    await r.set(f"session:{session_id}", json.dumps(state), ex=86400) # 24h
    return {"status": "saved"}

@router.get("/sessions/{session_id}")
async def get_session(session_id: str, r: redis.Redis = Depends(get_redis)):
    state = await r.get(f"session:{session_id}")
    if state:
        return json.loads(state)
    return {"status": "not_found"}
