"""Recheck ALL retrieved original documents of a recorded REAL negative search.

The vector ranking is unchanged. This validates the added deterministic target
check against the actual SQLite documents, not synthetic or substituted text.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from engine import DB, COLLECTION, explicit_identifiers, identifiers_present, save_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    args = parser.parse_args()
    result = json.loads(args.record.read_text(encoding='utf-8'))
    query = result['query']
    if not explicit_identifiers(query):
        raise ValueError('고유 식별번호가 포함된 실제 검색 기록이 필요합니다.')
    rejected = []
    conn = sqlite3.connect((DB/'chroma.sqlite3').as_uri()+'?mode=ro', uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        segment = conn.execute("SELECT s.id FROM segments s JOIN collections c ON c.id=s.collection WHERE s.scope='METADATA' AND c.name=?", (COLLECTION,)).fetchone()[0]
        for item in result['retrieved']:
            row = conn.execute("SELECT m.string_value FROM embeddings e JOIN embedding_metadata m ON m.id=e.id AND m.key='chroma:document' WHERE e.segment_id=? AND e.embedding_id=?", (segment, item['id'])).fetchone()
            if not row:
                raise ValueError('원문 누락')
            if not identifiers_present(query, row[0]):
                rejected.append(item['id'])
    finally:
        conn.close()
    if len(rejected) != len(result['retrieved']):
        raise ValueError('고유 식별번호가 확인된 후보가 있습니다. LLM 재검토가 필요합니다.')
    result.update(status='insufficient', evidence=[], first=[], flows=[], forecast=None,
                  required_identifiers=explicit_identifiers(query), identifier_rejected=rejected,
                  message='실제 전체 벡터 검색 상위 후보의 원문에 질문의 고유 식별번호가 없어 근거 부족으로 처리했습니다.',
                  identifier_recheck_source={'path': str(args.record.resolve()),
                    'sha256': hashlib.sha256(args.record.read_bytes()).hexdigest(), 'all_retrieved_originals_rechecked': len(rejected)})
    for key in ('forecast_evidence_ids', 'forecast_citations', 'forecast_reference_date'):
        result.pop(key, None)
    print(save_result(result), flush=True)


if __name__ == '__main__':
    main()
