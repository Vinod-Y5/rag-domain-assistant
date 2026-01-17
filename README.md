# Domain-Specific RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions using a custom knowledge base.

## Overview
The system retrieves relevant domain-specific context using semantic search and injects it into the prompt before generation, reducing hallucinations and improving factual accuracy.

## Architecture
- Sentence-Transformers for text embeddings
- FAISS for vector similarity search
- Open-source LLM via HuggingFace Inference API
- Streamlit for the user interface

## Workflow
1. Ingest domain text and generate embeddings
2. Store embeddings in a FAISS index
3. Retrieve relevant chunks for a user query
4. Inject retrieved context into the prompt
5. Generate grounded responses using an LLM

## Tech Stack
Python, FAISS, Sentence-Transformers, HuggingFace, Streamlit

## How to Run
```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
