import streamlit as st
import os
from rag import RAGEngine

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
