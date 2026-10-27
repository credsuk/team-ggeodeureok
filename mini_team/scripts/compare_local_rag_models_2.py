"""2차: 동일한 Chroma 검색 + 실측 군집/토픽 결과 + Ollama.
1차 실행 후, 군집화/토픽 모델링 결과 CSV를 준비한 뒤 실행한다.
실행: python compare_local_rag_models_2.py
"""
import csv
from pathlib import Path

from compare_local_rag_models_1 import ISSUES, run

EXPERIMENT = '2차_군집토픽RAG'
# 실제 군집화·토픽 모델링 결과를 사용자가 생성한 뒤 아래 파일로 배치한다.
TOPIC_CSV = Path(r'C:\mini_team\scripts\results\topic_context_by_issue.csv')


def load_topic_results():
    if not TOPIC_CSV.is_file():
        raise FileNotFoundError(
            f'군집화·토픽 모델링 결과 없음: {TOPIC_CSV}\n'
            'CSV 필수 컬럼: issue_keyword, topic_summary\n'
            '각 이슈별 실제 군집·토픽 분석 결과를 만든 뒤 2차를 실행하세요.'
        )
    rows = {}
    with TOPIC_CSV.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'issue_keyword', 'topic_summary'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError('topic_context_by_issue.csv 필수 컬럼: issue_keyword, topic_summary')
        for row in reader:
            issue = (row.get('issue_keyword') or '').strip()
            summary = (row.get('topic_summary') or '').strip()
            if issue and summary:
                if issue in rows:
                    raise ValueError(f'이슈 중복: {issue} (이슈별 한 행만 사용)')
                rows[issue] = {'summary': summary[:1800], 'source': str(TOPIC_CSV)}
    missing = [i for i in ISSUES if i not in rows]
    if missing:
        raise ValueError(f'군집화/토픽 결과가 없는 이슈: {missing}')
    return rows


if __name__ == '__main__':
    run(experiment=EXPERIMENT, topic_data=load_topic_results())
