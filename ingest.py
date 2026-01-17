from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

DATA_PATH = "data/knowledge.txt"
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.txt"


def build_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n") if c.strip()]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")


if __name__ == "__main__":
    build_index()
