# Domain-Specific RAG Assistant (Deployed)

A deployed **Retrieval-Augmented Generation (RAG)** application that answers questions using a custom domain knowledge base by combining semantic retrieval with open-source language models.

---

## 🔗 Live App
👉 **Live Demo:** https://rag-domain-assistant-3xjp8q2nttmrgwvxnxdpkm.streamlit.app/

---

## Overview

This project implements a domain-specific RAG system to improve factual accuracy and reduce hallucinations in language model outputs.  
Instead of relying solely on a language model’s internal knowledge, relevant context is retrieved from a knowledge base and injected into the prompt before generation.

The application is deployed on Streamlit Cloud and automatically builds its vector index at startup, making it cloud-safe and stateless.

---

## Architecture

- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (local similarity search)
- **LLM:** Open-source chat-based model via Hugging Face Inference API
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud

---

## Workflow

1. Load domain-specific text from a knowledge base
2. Split text into chunks
3. Generate embeddings for each chunk
4. Store embeddings in a FAISS index
5. Retrieve relevant chunks for a user query
6. Inject retrieved context into the prompt
7. Generate a grounded response using an LLM

---

## Project Structure

