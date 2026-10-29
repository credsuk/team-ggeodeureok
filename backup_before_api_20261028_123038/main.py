from fastapi import FastAPI, HTTPException

import asyncio
import logging
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import os

HTML_ONLY = os.getenv("HTML_ONLY", "0") == "1" # HTML 전용 테스트용 코드

if not HTML_ONLY:
    from config import settings
    from embedding import load_embedding_model, encode_query
    from llm import load_llm_model, generate_answer
    from database import connect_db, close_db, search_minutes


logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if HTML_ONLY:
        yield
        return

    app.state.embedding_model = await run_in_threadpool(
        load_embedding_model, settings.embedding_model_path
    )
    app.state.llm = await run_in_threadpool(
        load_llm_model, settings.llm_model_path
    )
    app.state.db_pool = await connect_db(settings.database_url)

    try:
        yield
    finally:
        await close_db(app.state.db_pool)


app = FastAPI(
    title="Local Meeting Minutes Search",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

search_semaphore = asyncio.Semaphore(1)

@app.post("/api/search")
async def search(request: SearchRequest):
    async with search_semaphore:
        try:
            query = request.query.strip()

            vector = await run_in_threadpool(
                encode_query, app.state.embedding_model, query
            )

            results = await search_minutes(
                app.state.db_pool,
                vector,
                request.limit,
            )

            answer = await run_in_threadpool(
                generate_answer,
                app.state.llm,
                query,
                results,
            )

            return {
                "answer": answer,
                "results": results,
            }

        except HTTPException:
            raise
        except Exception:
            logger.exception("검색 처리 실패")
            raise HTTPException(
                status_code=500,
                detail="검색 또는 답변 생성 중 오류가 발생했습니다.",
            )

# HTML Frontend 연결
from frontend import frontend_app
app.mount("/", frontend_app, name="frontend")
# HTML, JS 파일들을 정적 파일로 등록

# 메인 기본 코드 -----------------------------------------------------------------------------------------------