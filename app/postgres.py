import os, psycopg2, json
from dotenv import load_dotenv
load_dotenv()

conn_str = os.getenv("DATABASE_URL")
 
def run(sql, params=None, fetch=False):
    with psycopg2.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone() if fetch else None

def create_tables():
    run("""
    CREATE TABLE IF NOT EXISTS documents(
        id SERIAL PRIMARY KEY,
        title TEXT,
        file_name TEXT,
        file_type TEXT
    );
    
    CREATE TABLE IF NOT EXISTS chunks(
        id SERIAL PRIMARY KEY,
        document_id INT REFERENCES documents(id) ON DELETE CASCADE,
        chunk_text TEXT,
        chunk_index INT,
        metadata JSONB
    );
    
    -- Create GIN index to support full-text search
    CREATE INDEX IF NOT EXISTS idx_chunks_tsvector
        ON chunks
        USING gin(to_tsvector('english', chunk_text));
    """)


def insert_document(title, file_name, file_type):
    return run("INSERT INTO documents(title,file_name,file_type) VALUES(%s,%s,%s) RETURNING id;",
               (title,file_name,file_type), fetch=True)[0]

def insert_chunk(document_id, chunk_text, chunk_index, metadata=None):
    return run("INSERT INTO chunks(document_id,chunk_text,chunk_index,metadata) VALUES(%s,%s,%s,%s) RETURNING id;",
               (document_id,chunk_text,chunk_index,json.dumps(metadata or {})), fetch=True)[0]


def fetch_chunks_by_text(query: str, limit: int = 5):
    sql = """
    SELECT id, document_id, chunk_text, chunk_index, metadata,
           ts_rank_cd(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) AS rank
    FROM chunks
    WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
    ORDER BY rank DESC
    LIMIT %s;
    """
    with psycopg2.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(sql, (query, query, limit))
        rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "document_id": row[1],
            "chunk_text": row[2],
            "chunk_index": row[3],
            "metadata": row[4],
            "rank": row[5],
        }
        for row in rows
    ]
