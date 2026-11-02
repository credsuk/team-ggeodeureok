from __future__ import annotations

from collections.abc import Sequence


def extract_relevant_excerpt(text: str, query_terms: Sequence[str], max_chars: int = 360) -> str:
    """Return an exact substring near the densest matching query terms."""
    if max_chars < 1:
        raise ValueError("max_chars는 1 이상이어야 합니다.")
    if len(text) <= max_chars:
        return text

    terms: list[str] = []
    seen: set[str] = set()
    for raw in query_terms:
        phrase = " ".join(str(raw).split())
        if len(phrase) >= 2 and phrase.casefold() not in seen:
            terms.append(phrase)
            seen.add(phrase.casefold())
        for token in phrase.split():
            if len(token) >= 2 and token.casefold() not in seen:
                terms.append(token)
                seen.add(token.casefold())

    folded = text.casefold()
    occurrences: list[int] = []
    for term in terms:
        start = 0
        needle = term.casefold()
        while True:
            found = folded.find(needle, start)
            if found < 0:
                break
            occurrences.append(found)
            start = found + max(1, len(needle))

    if not occurrences:
        window_start = 0
    else:
        candidates: list[tuple[int, int, int]] = []
        for position in occurrences:
            start = max(0, position - max_chars // 3)
            end = min(len(text), start + max_chars)
            window = folded[start:end]
            matched_terms = [term for term in terms if term.casefold() in window]
            candidates.append((len(matched_terms), max((len(t) for t in matched_terms), default=0), -start))
        best = max(candidates)
        window_start = -best[2]

    window_end = min(len(text), window_start + max_chars)
    excerpt = text[window_start:window_end].strip()
    if window_start:
        excerpt = "… " + excerpt
    if window_end < len(text):
        excerpt += " …"
    return excerpt
