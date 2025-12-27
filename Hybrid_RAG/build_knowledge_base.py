# build_knowledge_base.py 

# This script scraps data from wikipedia pages and adds it to the existing chromaDB vector store

# topics added to vector_db = ["Influenza", "Tuberculosis", "Cardiovascular Disease", "Cancer", "Diabetes", "Alzheimer's Disease & Dementia", "Liver Disease", "Mental Health Disorders", "Stroke", "Chronic Respiratory Diseases"]


import os
from dotenv import load_dotenv
import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.processing.data_loader import load_wikipedia_data
from langchain_core.documents import Document

def add_new_documents_to_store(topics):
    load_dotenv()
    
    # 1. Load the existing Chroma store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./vector_db"
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    
    # 2. Scrape and prepare ONLY the new documents
    print("--- Scraping new data... ---")
    new_docs = load_wikipedia_data(topics)
    
    if not new_docs:
        print("No new documents to add. Exiting.")
        return
        
    # Convert dictionaries to Document objects
    new_langchain_docs = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
        for doc in new_docs
    ]

    # 3. Add the new documents to the existing store
    print("--- Adding new documents to existing vector store... ---")
    vector_store.add_documents(new_langchain_docs)
    
    print("--- Vector store updated successfully. ---")

if __name__ == "__main__":
    topics_to_add = ["Chronic Respiratory Diseases"]
    add_new_documents_to_store(topics=topics_to_add)