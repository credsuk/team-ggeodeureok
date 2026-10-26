#!/usr/bin/env python3
"""Canonical corpus에서 재현 가능한 연도×의회 층화 임베딩 표본을 만든다.

DB 비교용 표본이므로 절차성 발언도 원 분포대로 유지한다. 지나치게 긴 발언은
후속 토큰 기반 부모-자식 분할 대상으로 남겨두고 초기 표본에서는 제외한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "corpus.parquet",
    )
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--min-per-stratum", type=int, default=100)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4_000,
        help="이 값 미만의 발언만 초기 임베딩 표본에 포함",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def allocate_quotas(
    counts: dict[tuple[int, str], int], sample_size: int, minimum: int
) -> dict[tuple[int, str], int]:
    total_eligible = sum(counts.values())
    if sample_size > total_eligible:
        raise ValueError(
            f"표본 크기({sample_size:,})가 적격 행({total_eligible:,})보다 큽니다."
        )
    if sample_size < len(counts):
        raise ValueError(
            f"표본 크기는 연도×의회 층 수({len(counts)}) 이상이어야 합니다."
        )

    effective_minimum = min(minimum, sample_size // len(counts))
    quotas = {key: min(effective_minimum, count) for key, count in counts.items()}
    remaining = sample_size - sum(quotas.values())
    available = {key: counts[key] - quotas[key] for key in counts}
    available_total = sum(available.values())

    if remaining and not available_total:
        raise ValueError("층별 최소 할당 후 추가로 선택할 행이 없습니다.")

    fractional = []
    for key in sorted(counts):
        exact = remaining * available[key] / available_total if available_total else 0.0
        addition = min(available[key], int(exact))
        quotas[key] += addition
        fractional.append((exact - addition, key))

    remainder = sample_size - sum(quotas.values())
    for _, key in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if remainder == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remainder -= 1

    if remainder:
        for key in sorted(counts):
            while remainder and quotas[key] < counts[key]:
                quotas[key] += 1
                remainder -= 1
            if remainder == 0:
                break

    if sum(quotas.values()) != sample_size:
        raise RuntimeError("층화 표본 할당 합계가 sample_size와 일치하지 않습니다.")
    return quotas


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.sample_size < 1 or args.min_per_stratum < 0 or args.max_chars < 1:
        raise ValueError("sample-size/max-chars는 양수이고 min-per-stratum은 0 이상이어야 합니다.")

    suffix = f"{args.sample_size // 1000}k" if args.sample_size % 1000 == 0 else str(args.sample_size)
    output_path = (
        args.output.expanduser().resolve()
        if args.output else PROJECT_ROOT / "data" / "canonical" / f"embedding_sample_{suffix}.parquet"
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest else PROJECT_ROOT / "outputs" / "embedding_sample" / f"sample_{suffix}_manifest.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"출력이 이미 존재합니다. 덮어쓰려면 --overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".building")
    if temporary_output.exists():
        temporary_output.unlink()

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit={sql_string(args.memory_limit)}")
    connection.execute(f"SET threads={args.threads}")
    input_sql = sql_string(str(input_path))

    rows = connection.execute(
        f"""
        SELECT year, council, COUNT(*) AS row_count
        FROM read_parquet({input_sql})
        WHERE char_length < ?
        GROUP BY year, council
        ORDER BY year, council
        """,
        [args.max_chars],
    ).fetchall()
    counts = {(int(year), str(council)): int(count) for year, council, count in rows}
    if not counts:
        raise ValueError("표본 조건을 통과한 행이 없습니다.")
    quotas = allocate_quotas(counts, args.sample_size, args.min_per_stratum)

    total_rows, eligible_rows = connection.execute(
        f"""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE char_length < ?)
        FROM read_parquet({input_sql})
        """,
        [args.max_chars],
    ).fetchone()
    total_rows = int(total_rows)
    eligible_rows = int(eligible_rows)

    connection.execute("CREATE TEMP TABLE sample_quotas(year SMALLINT, council VARCHAR, quota BIGINT)")
    connection.executemany(
        "INSERT INTO sample_quotas VALUES (?, ?, ?)",
        [(year, council, quotas[(year, council)]) for year, council in sorted(quotas)],
    )

    output_sql = sql_string(str(temporary_output))
    print(
        f"층화 표본 생성: 전체 {total_rows:,}행 / 적격 {eligible_rows:,}행 / "
        f"선택 {args.sample_size:,}행"
    )
    connection.execute(
        f"""
        COPY (
            WITH ranked AS (
                SELECT
                    corpus.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY corpus.year, corpus.council
                        ORDER BY md5(corpus.chunk_id)
                    ) AS sample_rank
                FROM read_parquet({input_sql}) AS corpus
                WHERE corpus.char_length < {args.max_chars}
            )
            SELECT ranked.* EXCLUDE (sample_rank)
            FROM ranked
            JOIN sample_quotas
              ON ranked.year = sample_quotas.year
             AND ranked.council = sample_quotas.council
            WHERE ranked.sample_rank <= sample_quotas.quota
            ORDER BY ranked.year, ranked.council, ranked.chunk_id
        ) TO {output_sql} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
        """
    )

    selected_rows = int(
        connection.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_string(str(temporary_output))})"
        ).fetchone()[0]
    )
    if selected_rows != args.sample_size:
        temporary_output.unlink()
        raise RuntimeError(
            f"생성된 표본 수가 다릅니다: expected={args.sample_size}, actual={selected_rows}"
        )
    temporary_output.replace(output_path)

    selected_distribution = connection.execute(
        f"""
        SELECT year, council, COUNT(*) AS row_count
        FROM read_parquet({sql_string(str(output_path))})
        GROUP BY year, council
        ORDER BY year, council
        """
    ).fetchall()
    connection.close()

    manifest = {
        "created_by": Path(__file__).name,
        "input_file": str(input_path.relative_to(PROJECT_ROOT))
        if input_path.is_relative_to(PROJECT_ROOT) else input_path.name,
        "output_file": str(output_path.relative_to(PROJECT_ROOT))
        if output_path.is_relative_to(PROJECT_ROOT) else output_path.name,
        "output_sha256": file_sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "excluded_long_rows": total_rows - eligible_rows,
        "selected_rows": selected_rows,
        "selection": {
            "strata": ["year", "council"],
            "within_stratum_order": "md5(chunk_id)",
            "min_per_stratum": args.min_per_stratum,
            "char_length_rule": f"char_length < {args.max_chars}",
            "procedural_rows": "원 분포대로 유지",
            "short_rows": "원본 최소 길이가 30자이므로 별도 제외 없음",
        },
        "stratum_counts": [
            {
                "year": year,
                "council": council,
                "eligible": counts[(year, council)],
                "quota": quotas[(year, council)],
            }
            for year, council in sorted(counts)
        ],
        "selected_distribution": [
            {"year": int(year), "council": str(council), "rows": int(count)}
            for year, council, count in selected_distribution
        ],
        "raw_text_in_manifest": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 임베딩 표본 생성 완료 ===")
    print(f"표본: {selected_rows:,}행 | 층: {len(counts)}개")
    print(f"4,000자 이상 제외: {total_rows - eligible_rows:,}행")
    print(f"sample: {output_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
