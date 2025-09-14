from pineconedb import upsert_vectors, query_vector
from embeddings import embed_chunks, embed_text
from postgres import create_tables, insert_document, insert_chunk, fetch_chunks_by_text
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def retrieve_and_rerank(question: str, top_k: int = 5, alpha: float = 0.5, k_rrf: int = 60):
    """
    Hybrid search: semantic + full-text + RRF + cross-encoder rerank
    """
    question_emb = embed_text(question)

    # 1. Semantic hits
    semantic_hits = query_vector(question_emb, top_k=top_k * 2)

    # 2. Full-text hits
    keyword_hits = fetch_chunks_by_text(question, limit=top_k * 2)

    # 3. Merge candidates and compute RRF score
    seen_texts = set()
    candidates = []

    for rank, hit in enumerate(semantic_hits, start=1):
        text = hit.get("metadata", {}).get("chunk_text") or hit.get("text", "")
        if text not in seen_texts:
            seen_texts.add(text)
            rrf_score = 1 / (k_rrf + rank)
            candidates.append({
                "id": hit["id"],
                "text": text,
                "metadata": hit.get("metadata"),
                "source": "semantic",
                "raw_score": hit.get("score", 0.0),
                "rrf_score": rrf_score
            })

    for rank, hit in enumerate(keyword_hits, start=1):
        text = hit["chunk_text"]
        if text not in seen_texts:
            seen_texts.add(text)
            rrf_score = 1 / (k_rrf + rank)
            candidates.append({
                "id": f"pg_{hit['id']}",
                "text": text,
                "metadata": hit,
                "source": "keyword",
                "raw_score": hit["rank"],
                "rrf_score": rrf_score
            })

    if not candidates:
        return semantic_hits, keyword_hits, []

    # 4. Rerank candidates using cross-encoder
    pairs = [(question, candidate["text"]) for candidate in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        # Combine RRF score + cross-encoder score
        candidate["final_score"] = alpha * float(score) + (1 - alpha) * float(candidate["rrf_score"])

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    return semantic_hits, keyword_hits, candidates[:top_k]


# -------------------------
# Test
if __name__ == "__main__":
    query = "What services does Smith & Johnson Law Firm provide?"
    semantic_hits, keyword_hits, candidates = retrieve_and_rerank_rrf(query, top_k=5)

    # 1. Semantic search raw
    print("=== Semantic Search Results ===")
    for hit in semantic_hits:
        txt = hit.get("metadata", {}).get("chunk_text") or hit.get("text", "")
        print(f"[semantic] {txt[:80]}... (raw={hit.get('score')})")

    # 2. Full-text search raw
    print("\n=== Full-text Search Results ===")
    for hit in keyword_hits:
        print(f"[keyword] {hit['chunk_text'][:80]}... (rank={hit['rank']})")

    # 3. Final Hybrid RRF + Cross-Encoder
    print("\n=== Final Hybrid Reranked (RRF + CE) ===")
    for hit in candidates:
        print(f"{hit['text'][:80]}... "
              f"(final_score={hit['final_score']:.4f}, rrf_score={hit['rrf_score']:.4f}, raw={hit.get('raw_score')})")
