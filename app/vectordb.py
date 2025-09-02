import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
region = os.getenv("PINECONE_ENV")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region),
    )
    time.sleep(2)

index = pc.Index(index_name)

def upsert_vectors(vectors):
    payload = [(v["id"], v["embedding"], v.get("metadata", {})) for v in vectors]
    index.upsert(payload)

def query_vector(vec, top_k=5):
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in res.matches]
