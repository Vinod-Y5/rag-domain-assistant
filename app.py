import streamlit as st
from rag import RAGEngine
import os
import subprocess

if not os.path.exists("data/index.faiss"):
    subprocess.run(["python", "ingest.py"], check=True)


st.set_page_config(page_title="Domain RAG Assistant", layout="centered")
st.title("Domain-Specific RAG Assistant")

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.error("HF_TOKEN not set")
    st.stop()

engine = RAGEngine(hf_token)

question = st.text_input("Ask a question about the knowledge base")

if question:
    with st.spinner("Retrieving and generating..."):
        answer = engine.generate(question)
        st.markdown(answer)

