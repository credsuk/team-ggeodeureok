import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import settings
from embedding import load_embedding_model, encode_query
from llm import load_llm_model, generate_answer
from database import connect_db, close_db, search_minutes
from frontend import frontend_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=20)
    council: str | None = None
    year: int | None = Field(default=None, ge=2000, le=2100)

class DynamicAnalysisRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = None
    try:
        # Docker DB ?곌껐 ??곸쓣 寃利앺빀?덈떎.
        db_target = urlparse(settings.database_url)
        if (
            db_target.hostname != "council-pgvector"
            or db_target.port != 5432
            or db_target.path.lstrip("/") != "council_vectors"
        ):
            raise RuntimeError(
                "Docker DB council-pgvector:5432/council_vectors ?곌껐???꾩슂?⑸땲??"
            )
        os.environ["PGVECTOR_DSN"] = settings.database_url
        logger.info("Loading local embedding model...")
        app.state.embedding_model = await run_in_threadpool(
            load_embedding_model, settings.embedding_model_path
        )
        app.state.llm = load_llm_model()
        pool = await connect_db(settings.database_url)
        app.state.db_pool = pool
        logger.info("Database, embedding and Ollama configuration initialized.")
        yield
    finally:
        if pool is not None:
            await close_db(pool)

app = FastAPI(
    title="Local Meeting Minutes Search",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
search_semaphore = asyncio.Semaphore(1)

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/search")
async def search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="寃?됱뼱瑜??낅젰?섏꽭??")

    async with search_semaphore:
        started = time.perf_counter()
        try:
            vector = await run_in_threadpool(
                encode_query, app.state.embedding_model, query
            )
            results = await search_minutes(
                app.state.db_pool,
                vector,
                request.limit,
                request.council,
                request.year,
                query=query,
            )
            answer = await run_in_threadpool(
                generate_answer, app.state.llm, query, results
            )
            return {
                "answer": answer,
                "results": results,
                "count": len(results),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except Exception as exc:
            logger.exception("Search request failed")
            raise HTTPException(
                status_code=502,
                detail="寃???먮뒗 AI ?듬? ?앹꽦???ㅽ뙣?덉뒿?덈떎. ?쒕쾭 濡쒓렇瑜??뺤씤?섏꽭??",
            ) from exc


@app.post("/api/analyze")
async def analyze_dynamic_issue(request: DynamicAnalysisRequest):
    query = " ".join(request.query.split())
    if not query:
        raise HTTPException(status_code=422, detail="寃?됱뼱瑜??낅젰?섏꽭??")

    # ?숈쟻 遺꾩꽍??吏?뺣맂 Docker DB留??ъ슜?⑸땲??
    async with search_semaphore:
        started = time.perf_counter()
        try:
            root = Path(__file__).resolve().parent
            output_dir = root / "outputs" / "dynamic_issues"
            digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
            result_path = output_dir / f"query_{digest}" / "05_dynamic_result.json"
            env = os.environ.copy()
            env["PGVECTOR_DSN"] = settings.database_url
            env["COUNCIL_RETRIEVAL_MODE"] = "dense"

            command = [
                sys.executable, "-u",
                str(root / "src" / "33_analyze_free_query.py"),
                "--query", query,
                "--output-dir", str(output_dir),
                "--device", "auto",
                "--candidate-k", "50",
                "--rerank-top-k", "10",
                "--expansion-k", "500",
                "--ef-search", "200",
            ]
            completed = await run_in_threadpool(
                lambda: subprocess.run(
                    command,
                    cwd=str(root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
            )
            if completed.returncode != 0:
                logger.error(
                    "Dynamic analysis failed. stdout=%s stderr=%s",
                    completed.stdout[-4000:],
                    completed.stderr[-4000:],
                )
                raise RuntimeError("?숈쟻 遺꾩꽍 ?뚯씠?꾨씪???ㅽ뻾???ㅽ뙣?덉뒿?덈떎. ?쒕쾭 濡쒓렇瑜??뺤씤?섏꽭??")

            if not result_path.is_file():
                raise RuntimeError("遺꾩꽍 寃곌낵 JSON ?뚯씪???앹꽦?섏? ?딆븯?듬땲??")

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") != "provisional_until_review":
                raise RuntimeError("遺꾩꽍 寃곌낵 ?곹깭媛 ?덉긽怨??ㅻ쫭?덈떎.")

            payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            payload["result_path"] = str(result_path)
            return payload
        except subprocess.TimeoutExpired as exc:
            logger.exception("Dynamic analysis timed out")
            raise HTTPException(status_code=504, detail="?숈쟻 遺꾩꽍 ?쒓컙??珥덇낵?섏뿀?듬땲??") from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Dynamic analysis request failed")
            raise HTTPException(
                status_code=502,
                detail="?숈쟻 遺꾩꽍???ㅽ뙣?덉뒿?덈떎. ?쒕쾭 濡쒓렇瑜??뺤씤?섏꽭??",
            ) from exc


app.mount("/", frontend_app, name="frontend")

