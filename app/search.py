from vectordb import upsert_vectors, query_vector
from embeddings import embed_chunks, embed_text
from postgres import create_tables, insert_document, insert_chunk, fetch_chunks_by_text
from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def retrieve_and_rerank(question: str, top_k: int = 5, alpha: float = 0.5):
    q_emb = embed_text(question)

    # 1. Semantic hits
    semantic_hits = query_vector(q_emb, top_k=top_k * 2)

    # 2. Full-text hits
    keyword_hits = fetch_chunks_by_text(question, limit=top_k * 2)

    # 3. Merge candidates, remove duplicates (by chunk text)
    seen_texts = set()
    candidates = []

    for h in semantic_hits:
        text = h.get("metadata", {}).get("chunk_text") or h.get("text", "")
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append({
                "id": h["id"],
                "text": text,
                "metadata": h.get("metadata"),
                "source": "semantic",
                "raw_score": h.get("score", 0.0)
            })

    for h in keyword_hits:
        text = h["chunk_text"]
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append({
                "id": f"pg_{h['id']}",
                "text": text,
                "metadata": h,
                "source": "keyword",
                "raw_score": h["rank"]
            })

    if not candidates:
        return semantic_hits, keyword_hits, []

    # 4. Rerank using cross-encoder
    pairs = [(question, c["text"]) for c in candidates]
    scores = _reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        # Combine raw_score + reranker_score
        c["rerank_score"] = alpha * float(s) + (1 - alpha) * float(c["raw_score"])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return semantic_hits, keyword_hits, candidates[:top_k]


# -------------------------
# Test

if __name__ == "__main__":
    query = "What services does Smith & Johnson Law Firm provide?"
    semantic_hits, keyword_hits, results = retrieve_and_rerank(query, top_k=5)

    print(f"\nQuery: {query}\n")

    # 1. Semantic search raw
    print("=== Semantic Search Results ===")
    for h in semantic_hits:
        txt = h.get("metadata", {}).get("chunk_text") or h.get("text", "")
        print(f"[semantic] {txt[:80]}... (raw={h.get('score')})")

    # 2. Full-text search raw
    print("\n=== Full-text Search Results ===")
    for h in keyword_hits:
        print(f"[keyword] {h['chunk_text'][:80]}... (rank={h['rank']})")

    # 3. Final hybrid reranked
    print("\n=== Final Hybrid Reranked Results ===")
    for r in results:
        print(f"[{r['source']}] {r['text'][:80]}... "
              f"(rerank={r['rerank_score']:.4f}, raw={r.get('raw_score')})")
