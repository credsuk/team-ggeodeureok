from fastapi import FastAPI

from api.routers.chat import router as chat_router
from api.routers.documents import evidence_router, router as documents_router
from api.routers.issues import router as issues_router
from api.routers.radar import router as radar_router

app = FastAPI(
    title="의정이슈레이더 API",
    version="1.0.0",
    description="지방의회 회의록 이슈 분석 API. 로컬 실행을 위한 백엔드입니다.",
)
for router in (issues_router, radar_router, documents_router, evidence_router, chat_router):
    app.include_router(router, prefix="/api/v1")
