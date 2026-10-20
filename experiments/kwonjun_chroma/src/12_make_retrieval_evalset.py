#!/usr/bin/env python3
"""광역 지방의회 정책 검색 성능평가용 질의 목록을 만든다.

이 단계에서는 검색 결과를 아직 정답으로 간주하지 않는다. 13번 스크립트가
Arctic-ko, KURE-v1, BGE-M3의 Top-K 합집합을 만든 뒤 사람이 관련도를
0/1/2로 판정한다. 개발(dev) 질의로 모델과 검색 설정을 선택하고, 시험(test)
질의는 최종 설정을 확정한 뒤 한 번만 확인한다.

출력
    outputs/retrieval_eval/queries.csv
    outputs/retrieval_eval/LABELING_GUIDE.md
    outputs/retrieval_eval/evalset_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


SCRIPT_VERSION = "2026-10-20-retrieval-evalset-v1"

# 서로 비슷한 표현만 반복하지 않고, 실제 사용자가 입력할 법한 짧은 질의와
# 정책 의도가 드러나는 질의를 함께 배치한다. 각 분야 8개 중 앞 6개는 dev,
# 뒤 2개는 최종 test로 사용한다.
QUERY_GROUPS = {
    "인구·복지": [
        ("저출생과 출산율 감소 대응 정책", "출산·양육 지원과 인구 감소 대응"),
        ("공공산후조리원 설치와 산모 지원", "산후조리 공공성 및 산모 건강"),
        ("지방소멸 대응과 생활인구 확대", "인구감소지역 정주·유입 정책"),
        ("청년 인구 유출 방지와 지역 정착", "청년 정착·주거·일자리"),
        ("노인 돌봄과 고령사회 대응", "노인 돌봄 및 고령화 대응"),
        ("장애인 이동권과 복지서비스 확대", "장애인 접근성·복지"),
        ("다문화가족과 외국인 주민 지원", "다문화·외국인 지역사회 정착"),
        ("아동 돌봄 공백과 보육 지원", "보육·돌봄 인프라"),
    ],
    "교육·청년": [
        ("학령인구 감소와 학교 통폐합", "학생 감소에 따른 교육체계 개편"),
        ("늘봄학교와 방과후 돌봄", "초등 돌봄·방과후 교육"),
        ("지역대학 위기와 지방대학 육성", "지역대학 경쟁력·정주 인재"),
        ("교육격차와 농어촌 교육 지원", "지역·소득별 교육격차 완화"),
        ("학교폭력 예방과 학생 정신건강", "학생 안전·상담·정신건강"),
        ("청년 창업과 취업 지원", "청년 고용·창업 생태계"),
        ("무상급식과 친환경 학교급식", "학교급식 보편성·품질"),
        ("특수교육과 장애학생 지원", "특수교육 인력·시설"),
    ],
    "경제·산업": [
        ("반도체 산업과 첨단산업 클러스터", "첨단산업 유치·육성"),
        ("소상공인 매출 감소와 지역상권 회복", "골목상권·소상공인 지원"),
        ("지역화폐 발행과 지역경제 활성화", "지역화폐 효과·운영"),
        ("중소기업 금융 지원과 판로 확대", "중소기업 자금·판로"),
        ("산업단지 조성과 기업 유치", "산업단지·투자유치"),
        ("플랫폼 노동자와 비정규직 보호", "취약노동자 권익"),
        ("전통시장 현대화와 빈 점포 문제", "전통시장·상권 재생"),
        ("관광산업 육성과 체류형 관광", "지역관광 체류시간·소비 확대"),
    ],
    "농림·수산": [
        ("농촌 인력 부족과 외국인 계절근로자", "농번기 인력 수급"),
        ("청년농 육성과 농업 승계", "청년 농업인 정착"),
        ("농산물 가격 폭락과 소득 안정", "농가소득·가격안정"),
        ("스마트팜 보급과 미래 농업", "스마트농업 기술·시설"),
        ("가축전염병 방역과 축산농가 지원", "방역·축산 피해"),
        ("산불 예방과 산림 관리", "산불 대응·산림자원"),
        ("어촌소멸과 수산업 인력 부족", "어촌 정주·수산업 지속성"),
        ("해양쓰레기 수거와 연안 환경", "해양폐기물·연안 생태"),
    ],
    "환경·에너지": [
        ("탄소중립과 온실가스 감축", "지역 탄소중립 이행"),
        ("재생에너지 확대와 주민 수용성", "태양광·풍력 갈등"),
        ("전기차 충전 인프라 확대", "친환경차 충전시설"),
        ("생활폐기물 처리와 소각장 갈등", "폐기물 시설·주민갈등"),
        ("미세먼지 저감과 대기질 개선", "대기오염 저감"),
        ("하천 수질 개선과 녹조 대응", "수질·수생태 관리"),
        ("기후위기 폭염 대응과 취약계층 보호", "폭염 적응·취약계층"),
        ("분산에너지와 지역 전력망", "지역 에너지 자립"),
    ],
    "교통·도시": [
        ("버스 노선 부족과 대중교통 개선", "버스 접근성·노선 개편"),
        ("교통약자 이동지원 차량 확대", "장애인·고령자 특별교통수단"),
        ("광역철도와 도시철도 건설", "철도망 확충"),
        ("주차난 해소와 공영주차장 확충", "주차시설·주차정책"),
        ("빈집 정비와 원도심 재생", "빈집·도시재생"),
        ("공공임대주택과 주거복지", "주거취약계층 지원"),
        ("자전거도로와 보행자 안전", "보행·자전거 인프라"),
        ("개발제한구역 규제 완화", "토지이용 규제·개발"),
    ],
    "보건·안전": [
        ("지역의료 공백과 공공병원 확충", "공공의료·의료접근성"),
        ("응급의료체계와 소아응급 문제", "응급환자 이송·진료"),
        ("의료인력 부족과 지역 의사 확보", "의사·간호 인력 수급"),
        ("정신건강 상담과 자살 예방", "정신건강·자살예방"),
        ("감염병 대응과 방역체계", "감염병 감시·대응"),
        ("침수 피해와 도시 홍수 예방", "풍수해·배수시설"),
        ("전세사기 피해자 지원", "전세사기 구제·주거안정"),
        ("스쿨존과 어린이 교통안전", "어린이보호구역 안전"),
    ],
    "자치·문화": [
        ("주민참여예산과 주민 의견 반영", "주민참여·재정 민주성"),
        ("지방재정 악화와 재정 건전성", "세입·채무·재정운영"),
        ("공공기관 이전과 지역균형발전", "공공기관 유치·균형발전"),
        ("지방의회 인사권과 의정활동 지원", "의회 독립성·전문성"),
        ("지역축제 예산과 성과 평가", "축제 효율성·관광 효과"),
        ("문화예술인 지원과 지역문화 활성화", "문화예술 생태계"),
        ("체육시설 확충과 생활체육", "공공체육시설·생활체육"),
        ("도정 홍보비와 언론 예산", "홍보예산 투명성"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="지방의회 정책 검색 평가 질의 생성")
    parser.add_argument("--output-dir", default="outputs/retrieval_eval")
    parser.add_argument(
        "--extra-query-file",
        default="",
        help="추가 질의 파일. txt는 한 줄에 하나, CSV는 query/category/intent 컬럼 사용",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_extra_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"추가 질의 파일이 없습니다: {path}")
    rows = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                query = str(row.get("query", "")).strip()
                if query:
                    rows.append({
                        "category": str(row.get("category", "사용자 추가") or "사용자 추가").strip(),
                        "query": query,
                        "intent": str(row.get("intent", "") or "").strip(),
                        "split": str(row.get("split", "dev") or "dev").strip().lower(),
                    })
    else:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            query = line.strip()
            if query:
                rows.append({"category": "사용자 추가", "query": query, "intent": "", "split": "dev"})
    return rows


def make_rows(extra_rows: list[dict]) -> list[dict]:
    rows = []
    for category, items in QUERY_GROUPS.items():
        for index, (query, intent) in enumerate(items):
            rows.append({
                "category": category,
                "query": query,
                "intent": intent,
                "split": "dev" if index < 6 else "test",
                "source": "built_in",
            })
    for item in extra_rows:
        split = item.get("split", "dev")
        if split not in {"dev", "test"}:
            raise SystemExit(f"추가 질의 split은 dev/test만 가능합니다: {split}")
        rows.append({**item, "source": "user"})

    seen = set()
    unique = []
    for item in rows:
        canonical = " ".join(item["query"].split()).lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(item)

    for index, item in enumerate(unique, start=1):
        item["query_id"] = f"Q{index:03d}"
        item["query_sha1"] = hashlib.sha1(item["query"].encode("utf-8")).hexdigest()[:12]
    return unique


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.csv"
    guide_path = output_dir / "LABELING_GUIDE.md"
    summary_path = output_dir / "evalset_summary.json"

    existing = [path for path in (queries_path, guide_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "기존 평가셋이 있습니다. 보존하려면 다른 --output-dir을 사용하고, "
            "교체하려면 --overwrite를 지정하세요."
        )

    extra_rows = []
    if args.extra_query_file:
        extra_rows = read_extra_queries(Path(args.extra_query_file).expanduser().resolve())
    rows = make_rows(extra_rows)

    fields = ["query_id", "category", "query", "intent", "split", "source", "query_sha1"]
    with queries_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    guide = """# 검색 평가 라벨링 안내

`13_embedding_benchmark.py --mode prepare`가 만드는 `retrieval_judgments_union.csv`를
엑셀·LibreOffice·VS Code Data Wrangler 중 편한 도구로 엽니다.

`label` 열에는 숫자만 입력합니다.

- `2`: 질의의 정책 문제나 해결방안을 직접 논의함
- `1`: 같은 정책 분야이며 일부 맥락은 관련되지만 직접적인 답은 아님
- `0`: 동음이의어, 단순 언급, 회의 진행문 등 실질적으로 무관함

판정할 때 `source_models`, `source_ranks`, `similarity_scores`는 보지 않는 것을
권장합니다. 모델 이름과 점수를 보면 특정 모델에 유리한 편향이 생길 수 있습니다.
`semantic_text`를 먼저 보고 애매하면 `agenda_title`, `meeting_name`, `original_excerpt`를
확인합니다. 관련 청크가 하나도 없는 질의도 정상이며 `no_answer=1`로 표시할 수 있습니다.

`dev` 질의는 모델·Top-K·하이브리드 가중치 선택에 사용합니다. `test` 질의는 설정을
확정한 뒤 최종 성능을 한 번만 확인합니다. test 결과를 보고 설정을 반복 변경하면
실제 일반화 성능보다 점수가 높아질 수 있습니다.
"""
    guide_path.write_text(guide, encoding="utf-8")

    summary = {
        "script_version": SCRIPT_VERSION,
        "query_count": len(rows),
        "dev_count": sum(row["split"] == "dev" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows),
        "categories": sorted({row["category"] for row in rows}),
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "security": "질의 템플릿만 생성하며 회의록 원문을 외부로 전송하지 않음",
    }
    atomic_write_json(summary_path, summary)

    print("완료")
    print(f"질의: {queries_path} ({summary['query_count']}개)")
    print(f"dev/test: {summary['dev_count']}/{summary['test_count']}")
    print(f"라벨링 안내: {guide_path}")


if __name__ == "__main__":
    main()
