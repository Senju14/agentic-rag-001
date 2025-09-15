from sentence_transformers import CrossEncoder
from embeddings import embed_text
from postgres import fetch_chunks_by_text
from pineconedb import query_vector

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def retrieve_and_rerank(query: str, top_k: int = 5, rrf_k: int = 60):
    """
    Hybrid search pipeline:
    1. Semantic search (Pinecone)
    2. Full-text search (Postgres)
    3. Combine with Reciprocal Rank Fusion (RRF)
    4. Cross-encoder rerank
    Returns: semantic_hits, keyword_hits, candidates
    """

    # Semantic search
    query_emb = embed_text(query)
    semantic_hits = query_vector(query_emb, top_k=top_k * 2)    # * 2 because we need to rerank after combining

    # Full-text search
    keyword_hits = fetch_chunks_by_text(query, limit=top_k * 2)

    # Combine with RRF
    candidates = {}
    for rank, hit in enumerate(semantic_hits, start=1):
        text = hit.get("metadata", {}).get("chunk_text") or hit.get("text", "")
        chunk_id = hit.get("metadata", {}).get("chunk_id") or text
        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "text": text,
                "semantic_rank": rank,
                "semantic_score": hit.get("score"),
                "keyword_rank": None,
                "keyword_score": None,
            }
        else:
            candidates[chunk_id]["semantic_rank"] = rank
            candidates[chunk_id]["semantic_score"] = hit.get("score")

    for rank, hit in enumerate(keyword_hits, start=1):
        text = hit.get("chunk_text", "")
        chunk_id = hit.get("id") or text
        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "text": text,
                "semantic_rank": None,
                "semantic_score": None,
                "keyword_rank": rank,
                "keyword_score": hit.get("rank"),  # use rank as score
            }
        else:
            candidates[chunk_id]["keyword_rank"] = rank
            candidates[chunk_id]["keyword_score"] = hit.get("rank")

    # Calculate RRF score
    for cand in candidates.values():
        rrf_score = 0.0
        if cand["semantic_rank"] is not None:
            rrf_score += 1 / (rrf_k + cand["semantic_rank"])
        if cand["keyword_rank"] is not None:
            rrf_score += 1 / (rrf_k + cand["keyword_rank"])
        cand["rrf_score"] = rrf_score

    combined = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Cross-encoder rerank
    texts = [comb["text"] for comb in combined[: 2 * top_k]]
    if texts:
        scores = reranker.predict([(query, text) for text in texts])
        for comb, score in zip(combined[: 2 * top_k], scores):
            comb["final_score"] = float(score)

        combined = sorted(combined[: 2 * top_k], key=lambda x: x["final_score"], reverse=True)

    candidates = combined[:top_k]

    return semantic_hits, keyword_hits, candidates


# ------------------------- Test
# if __name__ == "__main__":
#     query = "What services does Smith & Johnson Law Firm provide?"
#     semantic_hits, keyword_hits, candidates = retrieve_and_rerank(query, top_k=5)

#     print("=== Semantic Search Results ===")
#     for i, r in enumerate(semantic_hits, start=1):
#         text = r.get("metadata", {}).get("chunk_text") or r.get("text", "")
#         print(f"[semantic] rank={i} | {text[:100]}... (raw_score={r.get('score')})")

#     print("\n=== Full-Text Search Results ===")
#     for i, r in enumerate(keyword_hits, start=1):
#         print(f"[keyword] rank={i} | {r.get('chunk_text')[:100]}... (rank={r.get('rank')})")

#     print("\n=== Hybrid (RRF + Rerank) Candidates ===")
#     for i, r in enumerate(candidates, start=1):
#         print(
#             f"{i}. {r['text'][:120]}..."
#             f" (sem_score={r.get('semantic_score')}, kw_score={r.get('keyword_score')},"
#             f" rrf_score={r['rrf_score']:.4f}, final_score={r.get('final_score')})"
#         )
