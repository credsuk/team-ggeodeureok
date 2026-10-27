"""Backup and clean 1st RAG rows. Keep valid rows only; regenerate other cases with updated runner."""
import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

CSV_PATH = Path(r'C:\mini_team\scripts\results\local_rag_model_comparison_v2.csv')
FIRST_EXPERIMENT = '1차_기본RAG'


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)
    with CSV_PATH.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise RuntimeError('CSV header missing')
    backup = CSV_PATH.with_name(f'{CSV_PATH.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.csv')
    shutil.copy2(CSV_PATH, backup)

    # Keep one successful 200-char-or-less result per experiment/issue/model.
    # For 1st RAG, invalid results are excluded so they can be regenerated.
    chosen = {}
    dropped = Counter()
    for row in rows:
        key = (row.get('experiment'), row.get('issue_keyword'), row.get('model'))
        length = len(row.get('prediction', '') or '')
        valid = row.get('status') == 'OK' and bool(row.get('prediction')) and length <= 200
        if key[0] == FIRST_EXPERIMENT and not valid:
            dropped['retry_needed'] += 1
            continue
        if key not in chosen:
            chosen[key] = row
        elif valid and not (chosen[key].get('status') == 'OK' and len(chosen[key].get('prediction', '') or '') <= 200):
            chosen[key] = row
            dropped['duplicate_replaced'] += 1
        else:
            dropped['duplicate_removed'] += 1
    cleaned = list(chosen.values())
    temporary = CSV_PATH.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cleaned)
    temporary.replace(CSV_PATH)
    count_first = sum(r.get('experiment') == FIRST_EXPERIMENT for r in cleaned)
    print(f'Backup: {backup}')
    print(f'Before: {len(rows)} rows / After: {len(cleaned)} rows')
    print(f'Valid first-phase rows retained: {count_first} / target 20')
    print(f'Removed for regeneration: {dropped["retry_needed"]}')
    print('Next: run compare_local_rag_models_1.py to fill missing rows.')

if __name__ == '__main__':
    main()
