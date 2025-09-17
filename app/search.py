from sentence_transformers import CrossEncoder
from app.embeddings import embed_text
from app.pineconedb import query_vector  

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def retrieve_and_rerank(query: str, top_k: int = 5):
    """
    Semantic search + Cross-encoder rerank.
    Returns: list of hits with both semantic_score and rerank_score
    """
    query_emb = embed_text(query)
    semantic_hits = query_vector(query_emb, top_k=top_k)

    if not semantic_hits:
        return []

    for hit in semantic_hits:
        hit["semantic_score"] = float(hit.get("score", 0.0))

    # Cross-encoder rerank
    candidate_texts = [hit.get("metadata", {}).get("chunk_text") or hit.get("text", "") for hit in semantic_hits]
    rerank_scores = reranker.predict([(query, text) for text in candidate_texts])

    # Add rerank_score to each hit
    for hit, score in zip(semantic_hits, rerank_scores):
        hit["rerank_score"] = float(score)

    # Sort by rerank_score in descending order
    semantic_hits = sorted(semantic_hits, key=lambda x: x["rerank_score"], reverse=True)
    return semantic_hits


# ------------------------- Test
if __name__ == "__main__":
    query = "What services does Smith & Johnson Law Firm provide?"
    hits = retrieve_and_rerank(query, top_k=5)

    print("=== Semantic + Rerank Search Results ===")
    for i, r in enumerate(hits, start=1):
        text = r.get("metadata", {}).get("chunk_text") or r.get("text", "")
        print(f"[rank={i}] semantic_score={r['semantic_score']:.4f} | rerank_score={r['rerank_score']:.4f} | {text[:100]}...")
