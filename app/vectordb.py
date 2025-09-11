import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
region = os.getenv("PINECONE_ENV")

if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region),
    )
    time.sleep(2)
 
index = pinecone.Index(index_name)

def upsert_vectors(vectors):
    payload = [(vector["id"], vector["embedding"], vector.get("metadata", {})) for vector in vectors]
    index.upsert(payload)

def query_vector(vector, top_k=5):
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{"id": match.id, "score": match.score, "metadata": match.metadata} for match in result.matches]
