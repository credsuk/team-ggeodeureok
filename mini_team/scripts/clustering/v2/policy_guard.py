"""Conservative lexical gates supplement, never replace, original-text LLM review."""
import re

CONCEPTS = {
    '실업급여': ['실업급여', '구직급여', '실업수당', '실업보험급여', '실직수당'],
    '전세사기': ['전세사기', '전세피해', '임대차사기', '전세보증금미반환', '깡통전세'],
    '디지털 교과서': ['디지털교과서', 'AI교과서', '인공지능교과서', '전자교과서'],
    '기후위기': ['기후위기', '기후변화', '기후재난', '지구온난화', '탄소중립'],
}

def compact(text):
    return re.sub(r'[^가-힣a-z0-9]', '', text.casefold())

def concepts(query):
    value = compact(query)
    found = {key: terms for key, terms in CONCEPTS.items() if any(compact(term) in value for term in terms)}
    if not found and ('실직' in query or '실업' in query) and any(x in query for x in ('급여', '수당')):
        found['실업급여'] = CONCEPTS['실업급여']
    return found

def matches(query, text):
    value = compact(text)
    return all(any(compact(term) in value for term in terms) for terms in concepts(query).values())

def variants(query, proposed):
    # A known target is immutable even when the model proposes another policy.
    target = concepts(query)
    if len(target) == 1:
        # Stable concept queries prevent broad natural-language expansions from
        # crowding the top-k. The original question remains the review/forecast
        # input, so its beneficiary, qualifiers and requested policy stay intact.
        core = next(iter(target))
        return [core + ' 관련 논의가 있나요?', core, core + ' 관련 논의', core + ' 현황']
    result = [query.strip()]
    if target:
        result += list(target)
    for text in proposed:
        if isinstance(text, str) and 1 <= len(text.strip()) <= 500 and (not target or matches(query, text)):
            result.append(text.strip())
    return list(dict.fromkeys(result))[:4]
