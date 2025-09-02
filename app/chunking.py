from typing import List, Dict
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import embed_text  

nltk.download("punkt_tab", quiet=True)

def semantic_chunk(
    text: str,
    chunk_size: int = 200,
    overlap: int = 30,
    similarity_threshold: float = 0.75
) -> List[Dict]:
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    embeddings = np.array(embed_text(sentences))

    chunks, cur_words = [], []
    idx = 0

    for i, sent in enumerate(sentences):
        words = sent.split()
        sim = cosine_similarity([embeddings[i]], [embeddings[i - 1]])[0][0] if i > 0 else 1.0

        if (len(cur_words) + len(words) > chunk_size) or (sim < similarity_threshold):
            chunks.append({"chunk_index": idx, "chunk_text": " ".join(cur_words)})
            idx += 1
            cur_words = cur_words[-overlap:] + words
        else:
            cur_words.extend(words)

    if cur_words:
        chunks.append({"chunk_index": idx, "chunk_text": " ".join(cur_words)})

    return chunks


# if __name__ == "__main__":
#     text = (
#         "Deep learning is a subfield of machine learning. "
#         "It uses neural networks with many layers. "
#         "Football is a popular sport played worldwide. "
#         "Teams try to score goals by kicking a ball into the net. "
#         "Transformers have changed natural language processing dramatically. "
#         "They allow models to capture long-range dependencies."
#     )

#     chunks = semantic_chunk(text, chunk_size=30, overlap=5, similarity_threshold=0.75)
#     for c in chunks:
#         print(c["chunk_index"], "|", c["chunk_text"])
