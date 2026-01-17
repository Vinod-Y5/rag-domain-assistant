import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.txt"


class RAGEngine:
    def __init__(self, hf_token: str):
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        if not os.path.exists(INDEX_PATH):
    raise RuntimeError(
        "FAISS index not found. Please run ingest.py to build the index."
    )

self.index = faiss.read_index(INDEX_PATH)


        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = [line.strip() for line in f.readlines()]

        self.client = InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",
            token=hf_token
        )

    def retrieve(self, query, k=3):
        query_embedding = self.embedder.encode([query]).astype("float32")
        _, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]

    def generate(self, query):
        context_chunks = self.retrieve(query)

        context = "\n".join(context_chunks)

        prompt = f"""
You are a technical assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful technical assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.3
        )

        return response.choices[0].message.content

