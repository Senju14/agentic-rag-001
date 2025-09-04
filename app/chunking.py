from typing import List, Dict
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import embed_text

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


# Test nhanh
if __name__ == "__main__":
    text = (
        "Deep learning is a subfield of machine learning. "
        "Football is a popular sport played worldwide. "
        "It uses neural networks with many layers. "
        "Teams try to score goals by kicking a ball into the net. "
        "Transformers have changed natural language processing dramatically. "
        "They allow models to capture long-range dependencies."
    )

    chunks = semantic_chunk(text, similarity_threshold=0.7)
    for c in chunks:
        print(c["chunk_index"], "|", c["chunk_text"])
