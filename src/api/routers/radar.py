import subprocess

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_api_service
from api.schemas import RadarSearchRequest
from api.services import RadarApiService

router = APIRouter(prefix="/radar", tags=["radar"])


@router.post("/search")
def search_radar(request: RadarSearchRequest, service: RadarApiService = Depends(get_api_service)):
    try:
        return service.radar_search(request.query)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="자유검색 분석이 제한 시간을 초과했습니다") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc))[-2000:]
        raise HTTPException(status_code=502, detail=f"자유검색 분석에 실패했습니다: {detail}") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

