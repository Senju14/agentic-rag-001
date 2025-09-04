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
    Semantic chunking: nhóm các câu dựa trên độ tương đồng cosine.
    Nếu similarity < threshold thì bắt đầu chunk mới.
    """
    if not text:
        return []

    # Tách thành câu
    sentences = sent_tokenize(text)

    # Sinh embedding cho từng câu
    embeddings = np.array(embed_text(sentences))

    chunks: List[Dict] = []
    visited = set()
    idx = 0

    for i, sentence in enumerate(sentences):
        if i in visited:
            continue

        # Bắt đầu chunk mới
        chunk = [sentence]
        visited.add(i)

        for j in range(i + 1, len(sentences)):
            if j not in visited:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])    [0][0]
                if sim > similarity_threshold:
                    chunk.append(sentences[j])
                    visited.add(j)

        chunks.append({
            "chunk_index": idx,
            "chunk_text": " ".join(chunk)
        })
        idx += 1

    return chunks


def load_pdf_text(pdf_path: str) -> str:
    """Đọc toàn bộ text từ file PDF"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()


# # Test nhanh
# if __name__ == "__main__":
#     pdf_path = os.path.join(os.path.dirname(__file__), "test.pdf")
#     if not os.path.exists(pdf_path):
#         print("Không tìm thấy test.pdf trong folder.")
#     else:
#         pdf_text = load_pdf_text(pdf_path)
#         print(f"PDF length: {len(pdf_text)} chars")

#         chunks = semantic_chunk(pdf_text, similarity_threshold=0.7)
#         for c in chunks:
#             print(c["chunk_index"], "|", c["chunk_text"][:200], "...")
