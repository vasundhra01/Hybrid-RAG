# main_vectordb.py

# This script runs a standard RAG chatbot that retrieves information only from the ChromaDB vector store to answer questions and saves the conversation data to "evaluation_data_vector_only.json"for evaluation.

# This script is used to get the responses for the system that only uses the data stored in Vector Database for answering the questions

# This data is used for making comparisons with our Hybrid RAG system

import os
import json
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# Import project components
# Router and KG retriever imports are removed
from src.retrieval.vector_retriever import get_vector_retriever

def get_retriever():
    """
    Initializes and returns only the vector store retriever.
    """
    load_dotenv()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_dir = "./vector_db"
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    vector_retriever = get_vector_retriever(vector_store, k=6)

    return vector_retriever

def run_chatbot():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY missing in .env file")

    # Only get the single vector retriever
    retriever = get_retriever()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )

    print("\nHello! I am a medical chatbot using a standard RAG system.")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    PROMPT_TEMPLATE = """
    You are a helpful AI assistant. Your task is to answer the user's question strictly based on the provided context.
    Do not use any external knowledge. If the answer cannot be found within the provided context, you must state that the information is not available.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    final_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Create an empty list to store the data for RAGAS evaluation
    evaluation_data = []

    # Create the single QA chain that will be used for all queries
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": final_prompt}
    )

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # --- Routing logic is removed ---
        # The chatbot now directly uses the QA chain
        
        print("Searching and generating response...")
        try:
            result = qa_chain.invoke({"query": user_query})
            
            # Access the retrieved context and store the data
            retrieved_context = [doc.page_content for doc in result['source_documents']]
            
            evaluation_data.append({
                "question": user_query,
                "answer": result['result'],
                "contexts": retrieved_context
            })

            print("\nFinal Answer: ", result['result'])
        
        except Exception as e:
            print(f"An error occurred while generating the response: {e}")

    # Save the collected data to a JSON file
    with open("evaluation_data_vector_only.json", "w") as f:
        json.dump(evaluation_data, f, indent=4)
        
    print("\nEvaluation data saved to evaluation_data_vector_only.json.")

if __name__ == "__main__":
    run_chatbot()