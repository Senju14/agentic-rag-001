from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def embed_text(text):
    is_str = isinstance(text, str)
    texts = [text] if is_str else text
    embeddings = sbert.encode(texts)
    embeddings = [embedding.tolist() for embedding in embeddings]
    return embeddings[0] if is_str else embeddings

def embed_chunks(chunks):
    embeddings = sbert.encode(chunks)
    results = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        results.append({
            "chunk_text": chunk,
            "chunk_index": i,
            "embedding": embedding.tolist()
        })
    return results

# # Test nhanh
# if __name__ == "__main__":
#     print("Single:", len(embed_text("hello world")))
#     print("List:", [len(v) for v in embed_text(["hello", "world"])])
#     chunks = ["Deep learning is powerful.", "Football is popular worldwide."]
#     out = embed_chunks(chunks)
#     for r in out:
#         print(r["chunk_index"], len(r["embedding"]), r["chunk_text"])
