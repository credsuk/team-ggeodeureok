#!/usr/bin/env python3
"""활성 코드와 비민감 요약만 복사해 GitHub/제출용 폴더를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_FILES = [
    "README.md", ".env.example", "docker-compose.yml",
    "requirements-base.txt", "requirements-enrichment.txt",
    "requirements-vector.txt", "requirements-forecast.txt",
    "requirements-app.txt", "requirements-issue-flow.txt",
]

ACTIVE_SRC = [
    "01_restart_inventory.py", "02_probe_received_metadata.py",
    "03_validate_row_id_join.py", "04_build_direct_canonical.py",
    "05_build_parent_child_chunks.py", "05b_validate_chunks.py",
    "06_make_embedding_eval_sample.py", "07_embed_eval_sample.py",
    "08_build_blind_retrieval_pool.py", "08_embed_full_corpus.py",
    "09_local_llm_relevance_judge.py", "10_compare_embedding_variants.py",
    "11_build_embedding_model_pool.py", "12_compare_embedding_models.py",
    "15_load_chroma.py", "16_load_pgvector.py",
    "17_benchmark_vector_databases.py", "18_pgvector_retrieval.py",
    "19_rerank_candidates.py", "20_expand_cluster_issue.py",
    "20_validate_embedding_artifact.py", "21_build_issue_timeline.py",
    "22_build_forecast_features.py", "23_forecast_issue.py",
    "24_build_web_payload.py", "25_build_issue_candidates.py",
    "25a_show_candidate_status.py", "26_cluster_threshold_sweep.py",
    "27_make_llm_extraction_evalset.py", "30_finalize_reviewed_issues.py",
    "31_build_multi_issue_panel.py", "32_train_multi_issue_forecast.py",
    "32a_external_followup_flow.py",
    "33_analyze_free_query.py", "34_build_final_web_bundle.py",
    "35_build_bm25_index.py", "bm25_runtime.py",
    "council_vector_runtime.py", "search_aliases.py",
    "external_policy_sources.py", "external_followup.py", "regional_stats.py",
]

ACTIVE_SCRIPTS = [
    "00_check_environment.py", "build_bm25_hybrid.sh",
    "check_post_embedding_setup.py", "run_final_app.sh",
    "run_forecast_evaluation.sh", "run_issue_review.sh",
    "run_representative_issues.sh", "41_prepare_release.py",
    "export_pgvector_for_share.sh", "restore_pgvector_from_dump.sh",
]

ACTIVE_DOCS = [
    "ACTIVE_PIPELINE.md", "BENCHMARK_PROTOCOL.md", "EXPERIMENT_LOG.md",
    "FORECAST_EVALUATION.md", "HYBRID_BM25_RUNBOOK.md",
    "POST_EMBEDDING_RUNBOOK.md", "REPRESENTATIVE_ISSUES_AND_AGENT_RUNBOOK.md",
    "EXTERNAL_DATA_INTEGRATION.md",
    "POSTGRES_DB_TEAM_SHARE.md",
    "FINAL_GITHUB_UPLOAD.md",
]

ACTIVE_DEMO = ["final_app.py", "issue_review_app.py"]
ACTIVE_AGENT = [
    "__init__.py", "local_issue_agent.py", "tools.py",
    "query_pipeline.py", "followup_tools.py",
]

RELEASE_GITIGNORE = """.venv/
.agents/
.env
__pycache__/
*.py[cod]
.idea/
.vscode/
.DS_Store
*:Zone.Identifier
logs/
archive/
deliverables/
*.zip

data/input/*
data/canonical/*
data/canonical_v3/*
data/embeddings/*
data/bm25/*
data/chroma/*
data/postgres/*
data/external/*
models/huggingface/*
outputs/*
exports/

!data/README.md
!models/README.md
!results/*.json
!results/*.md

*.npy
*.faiss
*.sqlite3
*.parquet
*.log
*.dump
*.dump.sha256
*.dump.info.txt
*.dump.part-*
"""

DATA_README = """# 데이터 배치

경진대회 제공 데이터는 이용 조건상 GitHub에 포함하지 않습니다.

로컬 실행 시 다음 경로에 배치합니다.

```text
data/input/                       원본 회의록과 제공 metadata
data/canonical_v3/corpus.parquet 통합 발화 corpus
data/canonical_v3/chunks_v1/     부모·자식 청크
data/embeddings/arctic_raw_v1/   전체 임베딩과 metadata
data/bm25/arctic_raw_v1/         SQLite FTS5 BM25 인덱스
data/postgres/                    Docker PostgreSQL 볼륨
```

원본 또는 원문이 포함된 중간 산출물을 공개 저장소에 업로드하지 마세요.
"""

MODEL_README = """# 로컬 모델

모델 가중치는 GitHub에 포함하지 않습니다.

| 용도 | 모델 |
|---|---|
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 재순위 | `BAAI/bge-reranker-v2-m3` |
| Agent | Ollama `exaone3.5:7.8b` |

Hugging Face 모델은 최초 1회 다운로드한 뒤 `models/huggingface/`에 보관하고,
제출·오프라인 실행 시에는 `local_files_only=True`를 사용합니다.

```bash
ollama pull exaone3.5:7.8b
ollama list
```
"""

RESULTS_README = """# 공개 가능한 결과

이 폴더에는 원문이 없는 성능 요약만 포함합니다. 회의록 인용문, 청크 본문,
검색 후보 및 이슈 구성원 JSON은 데이터 이용 조건상 공개하지 않습니다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT.parent / "council-issue-radar-release",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="기존 release 폴더가 있으면 삭제하지 않고 timestamp backup으로 이동",
    )
    return parser.parse_args()


def copy_file(source: Path, output: Path, relative: str, missing: list[str]) -> None:
    source_path = source / relative
    if not source_path.is_file():
        missing.append(relative)
        return
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


def copy_directory_files(
    source: Path, output: Path, directory: str, names: list[str], missing: list[str],
) -> None:
    for name in names:
        copy_file(source, output, f"{directory}/{name}", missing)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def copy_config(source: Path, output: Path) -> None:
    config_dir = source / "config"
    if not config_dir.is_dir():
        return
    for path in sorted(config_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {
            ".csv", ".json", ".txt", ".yaml", ".yml",
        }:
            destination = output / "config" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def sanitized_forecast(source: Path) -> dict[str, Any] | None:
    path = (
        source / "outputs" / "forecast" / "multi_issue_v1" / "model" /
        "multi_issue_forecast.json"
    )
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": report.get("status"),
        "method": report.get("method"),
        "leakage_control": report.get("leakage_control"),
        "issue_count": report.get("issue_count"),
        "training_rows": report.get("training_rows"),
        "validation_predictions": report.get("validation_predictions"),
        "selection": report.get("selection"),
        "required_evaluation": report.get("required_evaluation"),
        "rolling_backtest": report.get("rolling_backtest"),
        "limitations": report.get("limitations"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release(output: Path) -> list[str]:
    problems = []
    forbidden_names = {".env", "embeddings.f16.npy", "bm25.sqlite3"}
    forbidden_suffixes = {".npy", ".parquet", ".sqlite3", ".faiss", ".pyc"}
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes:
            problems.append(f"금지 파일: {relative}")
        if "Zone.Identifier" in path.name:
            problems.append(f"Windows 잔여 파일: {relative}")
        if path.stat().st_size > 25 * 1024 * 1024:
            problems.append(f"25MB 초과: {relative}")
    return problems


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "src" / "18_pgvector_retrieval.py").is_file():
        raise FileNotFoundError(f"프로젝트 루트가 올바르지 않습니다: {source}")
    if output == source or source in output.parents:
        raise ValueError("release 폴더는 원본 프로젝트 바깥에 생성해야 합니다")
    backup = None
    if output.exists():
        if not args.replace:
            raise FileExistsError(
                f"release 폴더가 이미 있습니다: {output}\n"
                "교체하려면 --replace를 사용하세요. 기존 폴더는 backup됩니다."
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.name}_backup_{timestamp}")
        output.replace(backup)
    output.mkdir(parents=True)

    missing: list[str] = []
    for relative in ROOT_FILES:
        copy_file(source, output, relative, missing)
    copy_file(source, output, "docs/README_RELEASE.md", missing)
    copy_directory_files(source, output, "src", ACTIVE_SRC, missing)
    copy_directory_files(source, output, "scripts", ACTIVE_SCRIPTS, missing)
    copy_directory_files(source, output, "docs", ACTIVE_DOCS, missing)
    copy_directory_files(source, output, "demo", ACTIVE_DEMO, missing)
    copy_directory_files(source, output, "agent", ACTIVE_AGENT, missing)
    copy_config(source, output)
    copy_file(source, output, "eval/queries.csv", missing)

    figure_source = source / "모델 구조.png"
    if figure_source.is_file():
        figure_output = output / "figures" / "model_architecture.png"
        figure_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_source, figure_output)

    atomic_text(output / ".gitignore", RELEASE_GITIGNORE)
    atomic_text(output / "data" / "README.md", DATA_README)
    atomic_text(output / "models" / "README.md", MODEL_README)
    atomic_text(output / "results" / "README.md", RESULTS_README)
    forecast = sanitized_forecast(source)
    if forecast is not None:
        atomic_json(output / "results" / "forecast_metrics.json", forecast)

    problems = validate_release(output)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema_version": "council_issue_radar_release_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "output": str(output),
        "backup": str(backup) if backup else None,
        "missing_optional_or_required_files": missing,
        "validation_problems": problems,
        "file_count": len(files),
        "files": files,
    }
    atomic_json(output / "RELEASE_MANIFEST.json", manifest)

    print("\n=== GitHub/제출용 release 생성 완료 ===")
    print(f"원본: {source}")
    print(f"release: {output}")
    if backup:
        print(f"이전 release 백업: {backup}")
    print(f"파일: {len(files):,}")
    print(f"누락: {len(missing):,}")
    for relative in missing:
        print(f"  MISSING | {relative}")
    print(f"검증 문제: {len(problems):,}")
    for problem in problems:
        print(f"  FAIL | {problem}")
    if problems:
        raise RuntimeError("release 검증에 실패했습니다")
    print("최종 결과: PASS")


if __name__ == "__main__":
    main()

