from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def embed_text(text):
    embeddings = sbert.encode(text)
    return embeddings.tolist()

# No tolist() the result will be <ndarry>
# Pinecone work with <list> 

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

# Test nhanh

# if __name__ == "__main__":
#     print("Single:", len(embed_text("hello world")))
#     print("List:", [len(vector) for vector in embed_text(["hello", "world"])])
#     chunks = ["Deep learning is powerful.", "Football is popular worldwide."]
#     out = embed_chunks(chunks)
#     for result in out:
#         print(result["chunk_index"], len(result["embedding"]), result["chunk_text"])
