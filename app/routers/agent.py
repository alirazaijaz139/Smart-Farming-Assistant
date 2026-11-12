from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.graph import run_agent

from .. import models
from ..security import get_current_user

router = APIRouter(prefix="/agent", tags=["Farming Agent"])


class ChatInput(BaseModel):
    message: str = Field(min_length=2, max_length=500)


@router.post("/chat")
def chat(body: ChatInput, user: models.User = Depends(get_current_user)):
    try:
        return run_agent(body.message)
    except Exception as e:
        raise HTTPException(502, f"Agent failed: {e}")