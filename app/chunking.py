from typing import List, Dict
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import embed_text
import PyPDF2
import os


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Semantic chunking: group sentences based on cosine similarity.
    If similarity < threshold then start new chunk.
    """
    if not text:
        return []

    # Split into sentences
    sentences = sent_tokenize(text)

    # Generate embedding for each sentence
    embeddings = np.array(embed_text(sentences))

    chunks: List[Dict] = []
    visited = set()
    idx = 0

    for i, sentence in enumerate(sentences):
        if i in visited:
            continue

        # Start new chunk
        chunk = [sentence]
        visited.add(i)

        for j in range(i + 1, len(sentences)):
            if j not in visited:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > similarity_threshold:
                    chunk.append(sentences[j])
                    visited.add(j)

        chunks.append({
            "chunk_index": idx,
            "chunk_text": " ".join(chunk)
        })
        idx += 1

    return chunks

# # Quick Test 01
# def load_pdf_text(pdf_path: str) -> str:
#     """Read all text from PDF file"""
#     text = ""
#     with open(pdf_path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         for page in reader.pages:
#             text += page.extract_text() + "\n"
#     return text.strip()

# if __name__ == "__main__":
#     pdf_path = "utils/test.pdf"
#     if not os.path.exists(pdf_path):
#         print("Cannot find test.pdf in folder.")
#     else:
#         pdf_text = load_pdf_text(pdf_path)
#         print(f"PDF length: {len(pdf_text)} chars")

#         chunks = semantic_chunk(pdf_text, similarity_threshold=0.7)
#         for c in chunks:
#             print(c["chunk_index"], "|", c["chunk_text"][:200], "...")

# --------------------------------------------------
# # Quick Test 02
# if __name__ == "__main__":
#     from sentence_transformers import SentenceTransformer
#     from sklearn.metrics.pairwise import cosine_similarity

#     sbert = SentenceTransformer('all-MiniLM-L6-v2')
#     sentences = ["I love cats", "I like cats", "The weather is sunny"]
#     embeddings = sbert.encode(sentences)

#     # Calculate similarity between sentence 0 and 1
#     sim_0_1 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#     print("Similarity between 0 and 1:", sim_0_1) # 0.9046357

#     # Calculate similarity between sentence 0 and 2
#     sim_0_2 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
#     print("Similarity between 0 and 2:", sim_0_2) # 0.005693812

#     # If delete [0][0] -> [[0.9046357]] 1x1