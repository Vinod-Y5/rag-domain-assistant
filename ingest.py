from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

DATA_PATH = "data/knowledge.txt"
INDEX_PATH = "data/index.faiss"

def ingest():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n") if c.strip()]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)

    # save chunks
    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
