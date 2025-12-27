# src/processing/data_loader.py

# This script functions as the data ingestion pipeline for the RAG system, responsible for scraping source documents from Wikipedia and then chunking and embedding them into a persistent ChromaDB vector store for later retrieval.

import os
import wikipedia
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_wikipedia_data(page_titles):
    """Fetches full content of specified Wikipedia pages using a robust search."""
    docs = []
    print("Scraping Wikipedia pages...")
    for title in page_titles:
        try:
            search_results = wikipedia.search(title, results=1)
            if not search_results:
                print(f"Error: No search results found for '{title}'. Skipping.")
                continue
            
            page_title = search_results[0]
            wiki_page = wikipedia.page(page_title, auto_suggest=False)
            
            doc_content = {
                "page_content": wiki_page.content,
                "metadata": {"source": wiki_page.title}
            }
            docs.append(doc_content)
            print(f"Successfully scraped: {wiki_page.title}")
        except wikipedia.exceptions.PageError:
            print(f"Error: Page '{title}' does not exist. Skipping.")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Error: Disambiguation page for '{title}'. Suggestions: {e.options}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred for '{title}': {e}. Skipping.")
            
    return docs

def load_documents_from_vector_store():
    """Loads documents from the existing vector store."""
    load_dotenv()
    
    persist_dir = "./vector_db"
    if not os.path.exists(persist_dir):
        print("Error: Vector store directory not found. Please build the vector store first.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    documents = vector_store.get(include=["metadatas", "documents"])
    docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(documents['documents'], documents['metadatas'])
    ]
    
    return docs

def build_vector_store(docs):
    load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    print("Building Vector Store...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.create_documents([doc["page_content"] for doc in docs])
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_dir = "./vector_db"
    vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
    print("Vector store built successfully.")
    return vector_store