from __future__ import annotations

from functools import lru_cache

from api.services import RadarApiService


@lru_cache(maxsize=1)
def get_api_service() -> RadarApiService:
    return RadarApiService()
