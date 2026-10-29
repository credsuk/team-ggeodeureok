from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_api_service
from api.schemas import IssueListResponse
from api.services import RadarApiService

router = APIRouter(prefix="/issues", tags=["issues"])


@router.get("", response_model=IssueListResponse)
def list_issues(service: RadarApiService = Depends(get_api_service)):
    try:
        items = service.list_issues()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"items": items, "count": len(items)}


@router.get("/{issue_key}")
def get_issue(issue_key: str, service: RadarApiService = Depends(get_api_service)):
    try:
        payload = service.get_issue(issue_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="대표 이슈를 찾을 수 없습니다")
    return service._attach_fixed_followup(payload)
