"""app.py: Streamlit web application for the Moroccan Constitution Q&A system."""

import streamlit as st
import sys
import os

from retriever import load_vector_store, get_relevant_documents, VECTOR_STORE_PATH, EMBEDDING_MODEL
from generator import initialize_llm_pipeline, generate_answer, LLM_MODEL_NAME

# --- Page Configuration ---
st.set_page_config(
    page_title="Moroccan Constitution Q&A",
    page_icon="🇲🇦",
    layout="wide"
)

# --- Load Models and Vector Store (cached for performance) ---
@st.cache_resource
def load_resources():
    """Loads the vector store and LLM pipeline once."""
    st.write("Loading resources... This may take a moment.")
    vector_store = load_vector_store(VECTOR_STORE_PATH, EMBEDDING_MODEL)
    if vector_store is None:
        st.error(f"Failed to load vector store from {VECTOR_STORE_PATH}. Please ensure it has been created by running retriever.py.")
        return None, None

    llm_pipeline_instance = initialize_llm_pipeline(LLM_MODEL_NAME)
    if llm_pipeline_instance is None:
        st.error(f"Failed to initialize LLM pipeline with model {LLM_MODEL_NAME}. Check logs.")
        return vector_store, None

    st.success("Resources loaded successfully!")
    return vector_store, llm_pipeline_instance

vector_store, llm_pipeline = load_resources()

# --- Application UI ---
st.title("🇲🇦 Interactive Q&A on the Moroccan Constitution (2011)")
st.markdown("Ask questions in English about the Moroccan Constitution, and the system will attempt to answer based on its articles.")
st.markdown("**Disclaimer:** This system provides information based on the Constitution and is not a substitute for legal advice.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your question here..."):
    if vector_store is None or llm_pipeline is None:
        st.error("System is not ready. Resources could not be loaded. Please check the application logs.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            answer = generate_answer(prompt, vector_store, llm_pipeline, k_retrieval=3)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar for additional information
st.sidebar.header("About")
st.sidebar.info(
    "This is a RAG (Retrieval Augmented Generation) system designed to answer questions "
    "about the Moroccan Constitution of 2011. It uses a vector database of constitutional articles "
    "and a Large Language Model to generate answers."
)
st.sidebar.markdown("--- ")
st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
st.sidebar.markdown(f"**LLM Model:** `{LLM_MODEL_NAME}`")
st.sidebar.markdown(f"**Vector Store:** FAISS at `{VECTOR_STORE_PATH}`")

if __name__ == "__main__":
    if vector_store is None or llm_pipeline is None:
        st.error("Application cannot start due to resource loading failure. Check console logs.")
    else:
        st.info("Application ready. Ask your questions about the Moroccan Constitution.")
