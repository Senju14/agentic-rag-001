from typing import List, Dict, Optional

def deduplicate(results: List[Dict], key: str = "text") -> List[Dict]:
    seen, unique = set(), []
    for r in results:
        txt = (r.get(key) or "").strip().lower()
        if txt not in seen:
            seen.add(txt)
            unique.append(r)
    return unique

def filter_results(
    results: List[Dict],
    min_len: int = 10,
    allowed_sources: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None,
    min_score: Optional[float] = None,
) -> List[Dict]:
    out = []
    for r in results:
        txt = r.get("text") or r.get("chunk_text") or ""
        md = r.get("metadata", {})

        # 1. filter theo độ dài
        if len(txt) < min_len:
            continue

        # 2. filter theo nguồn (semantic / keyword / …)
        if allowed_sources and r.get("source") not in allowed_sources:
            continue

        # 3. filter theo loại file (txt, pdf,…)
        if allowed_types and md.get("file_type") not in allowed_types:
            continue

        # 4. filter theo score
        score = r.get("score") or r.get("raw_score")
        if min_score and score and score < min_score:
            continue

        out.append(r)
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
