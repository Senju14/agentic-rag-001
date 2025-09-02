import os
import psycopg2
from dotenv import load_dotenv
from pinecone import Pinecone 

load_dotenv()

# -------------------------
# PostgreSQL config
conn_str = os.getenv("DATABASE_URL")

def clear_postgres():
    with psycopg2.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks;")
        cur.execute("DELETE FROM documents;")
        conn.commit()
    print("Postgres: All documents and chunks deleted.")


# -------------------------
# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_HOST = os.getenv("PINECONE_HOST") 

def clear_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    indexes = pc.list_indexes().names()
    if PINECONE_INDEX not in indexes:
        print(f"Pinecone index '{PINECONE_INDEX}' not found.")
        return

    index_args = {"name": PINECONE_INDEX}
    if PINECONE_HOST:
        index_args["host"] = PINECONE_HOST

    index = pc.Index(**index_args)
    index.delete(delete_all=True)
    print(f"Pinecone: All vectors deleted from index '{PINECONE_INDEX}'.")
 

# -------------------------
if __name__ == "__main__":
    confirm = input("Are you sure you want to DELETE ALL data in Postgres and Pinecone? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborted.")
    else:
        clear_postgres()
        clear_pinecone()
        print("All data cleared.")
