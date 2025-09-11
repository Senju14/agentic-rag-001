from typing import List, Dict, Optional


def deduplicate(results: List[Dict], key: str = "text") -> List[Dict]:
    seen, unique = set(), []
    for result in results:
        txt = (result.get(key) or "").strip().lower()
        if txt not in seen:
            seen.add(txt)
            unique.append(result)
    return unique



def filter_results(
    results: List[Dict],
    min_len: int = 10,
    allowed_sources: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None,
    min_score: Optional[float] = None,
) -> List[Dict]:
    out = []
    for result in results:
        txt = result.get("text") or result.get("chunk_text") or ""
        md = result.get("metadata", {})

        # 1. filter by length
        if len(txt) < min_len:
            continue

        # 2. filter by source (semantic / keyword / …)
        if allowed_sources and result.get("source") not in allowed_sources:
            continue

        # 3. filter by file type (txt, pdf,…)
        if allowed_types and result.get("file_type") not in allowed_types:
            continue

        # 4. filter by score
        score = result.get("score") or result.get("raw_score")
        if min_score and score and score < min_score:
            continue

        out.append(result)
    return out



def postprocess(
    results: List[Dict],
    min_len: int = 10,
    allowed_sources: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None,
    min_score: Optional[float] = None,
) -> List[Dict]:
    """Pipeline: deduplicate → filter"""
    results = deduplicate(results)
    results = filter_results(
        results,
        min_len=min_len,
        allowed_sources=allowed_sources,
        allowed_types=allowed_types,
        min_score=min_score,
    )
    return results
