from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from agent.local_issue_agent import probe_ollama
from api.dependencies import get_api_service
from api.schemas import ChatRequest
from api.services import RadarApiService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/status")
def chat_status() -> dict[str, object]:
    """Called once when the client is about to open/use chat; no polling is implied."""
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("COUNCIL_AGENT_MODEL", "exaone3.5:7.8b")
    result = probe_ollama(url, model, timeout_seconds=3.0)
    return {"reachable": result["reachable"], "model_available": result["model_available"],
            "model": model, "error": result["error"]}


@router.post("")
def ask_chat(request: ChatRequest, service: RadarApiService = Depends(get_api_service)):
    try:
        return service.chat(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
