from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_api_service
from api.services import RadarApiService

router = APIRouter(prefix="/documents", tags=["documents"])
evidence_router = APIRouter(prefix="/evidence", tags=["documents"])


def validate_date(value: str | None, name: str) -> None:
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{name}는 YYYY-MM-DD 형식이어야 합니다") from exc


@router.get("/search")
def search_documents(
    q: str = Query(min_length=1, max_length=500),
    council: str | None = None,
    year: int | None = Query(default=None, ge=2000, le=2100),
    committee: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = Query(default=10, ge=1, le=50),
    service: RadarApiService = Depends(get_api_service),
):
    validate_date(date_from, "date_from")
    validate_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from은 date_to보다 늦을 수 없습니다")
    try:
        return service.document_search(q, council=council, year=year, committee=committee,
                                       date_from=date_from, date_to=date_to, top_k=top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@evidence_router.get("/{chunk_id}")
def get_evidence(chunk_id: str, service: RadarApiService = Depends(get_api_service)):
    try:
        result = service.evidence(chunk_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"원문 근거를 읽지 못했습니다: {type(exc).__name__}") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="근거 청크를 찾을 수 없습니다")
    return result


@router.get("/{document_id}")
def get_document(document_id: str, limit: int = Query(default=100, ge=1, le=500),
                 service: RadarApiService = Depends(get_api_service)):
    try:
        result = service.document(document_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"회의록 원문을 읽지 못했습니다: {type(exc).__name__}") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="회의록을 찾을 수 없습니다")
    return result
