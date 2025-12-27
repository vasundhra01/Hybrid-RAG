# src/retrieval/vector_retriever.py

# This script creates a retriever to search for relevant data chunks stored in ChromaDB vector store.

from langchain_community.vectorstores import Chroma

def get_vector_retriever(vector_store, **kwargs):
    """Returns a ChromaDB vector store retriever."""
    return vector_store.as_retriever(**kwargs)