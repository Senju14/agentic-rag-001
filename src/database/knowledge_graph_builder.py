# src/database/knowledge_graph_builder.py
import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain.schema import Document
from groq import Groq
from src.utils.file_loader import read_file
from src.core.text_chunker import semantic_chunk


# === Load environment variables ===
load_dotenv()
DATA_FOLDER = "src/data/"

# === Groq LLM Client ===
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL")

# === Neo4j Connector ===
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    refresh_schema=True,
)


# -----------------------------------------------------
async def extract_graph_with_groq(text: str) -> dict:
    """
    Send a text chunk to Groq → request to generate JSON with nodes and relationships.
    """
    system_prompt = """
    You are a knowledge graph extractor.
    Analyze the text and extract entities and relationships.

    Return STRICT JSON ONLY with this structure:
    {
        "nodes": [
            {"id": "string", "type": "string", "properties": {}}
        ],
        "relationships": [
            {
                "source": "node_id",
                "target": "node_id",
                "type": "string",
                "properties": {}
            }
        ]
    }

    Example:
    {
        "nodes": [
            {"id": "GreenFields BioTech", "type": "Company"},
            {"id": "Zurich", "type": "City"},
            {"id": "Switzerland", "type": "Country"}
        ],
        "relationships": [
            {"source": "GreenFields BioTech", "target": "Zurich", "type": "HEADQUARTERS"},
            {"source": "Zurich", "target": "Switzerland", "type": "LOCATED_IN"}
        ]
    }
    """

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.strip("```").replace("json", "").strip()

    try:
        data = json.loads(raw)
        return data
    except Exception as e:
        print(f"JSON parse error: {e}\nRaw output:\n{raw[:500]}")
        return {"nodes": [], "relationships": []}


# -----------------------------------------------------
async def build_graph():
    for fname in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, fname)
        text = read_file(path)
        chunks = semantic_chunk(text)
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]

        print(f"\nFile: {fname} | {len(chunks)} chunks found.")

        for idx, chunk in enumerate(chunk_texts):
            print(f"→ Extracting chunk {idx+1}/{len(chunks)}...")

            # Call Groq to extract graph
            graph_data = await extract_graph_with_groq(chunk)
            nodes = graph_data.get("nodes", [])
            rels = graph_data.get("relationships", [])

            # Push to Neo4j
            with graph._driver.session() as session:
                for n in nodes:
                    session.run(
                        """
                        MERGE (a:Entity {id: $id})
                        SET a.type = $type
                        """,
                        id=n.get("id"),
                        type=n.get("type", "Unknown"),
                    )

                for r in rels:
                    session.run(
                        """
                        MATCH (a:Entity {id: $source})
                        MATCH (b:Entity {id: $target})
                        MERGE (a)-[rel:RELATION {type: $type}]->(b)
                        """,
                        source=r.get("source"),
                        target=r.get("target"),
                        type=r.get("type", "RELATED_TO"),
                    )

    print("\nGraph successfully built and pushed to Neo4j!")


# python -m src.database.knowledge_graph_builder
# -----------------------------------------------------
# if __name__ == "__main__":
#     asyncio.run(build_graph())
