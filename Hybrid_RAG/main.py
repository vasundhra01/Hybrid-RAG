# main.py

# This script is used to run hybrid RAG chatbot, using an AI router to dynamically choose the best data source—vector store, knowledge graph, or both—to answer a user's question.

# Same as main2.py.

# Only difference is that this script uses LLM's knowledge to answer the question if it cannot find the sufficient information from the available data.

# Although for evaluation purpose this script is not used, instead main2.py is used

import os
import json
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 

# Import project components
from src.routing.router import get_router_chain
from src.retrieval.kg_retriever import get_kg_retriever
from src.retrieval.vector_retriever import get_vector_retriever
from src.routing.router import RouteQuery

def get_retrievers():
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_dir = "./vector_db"
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    vector_retriever = get_vector_retriever(vector_store, k=6)

    kg_retriever = None
    try:
        graph = Neo4jGraph()
        kg_retriever = get_kg_retriever(graph, k=6)
    except Exception as e:
        print(f"Failed to connect to Neo4j. Knowledge graph will be skipped. Error: {e}")
        kg_retriever = None

    return vector_retriever, kg_retriever

def run_chatbot():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY missing in .env file")

    vector_retriever, kg_retriever = get_retrievers()
    router_chain = get_router_chain()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )

    print("\nHello! I am a medical chatbot with dynamic retrieval capabilities.")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    PROMPT_TEMPLATE = """
    You are a helpful AI assistant. Answer the user's question based on the provided context.
    If you cannot find the answer in the provided context, answer the question based on your own knowledge.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    final_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    #  Create an empty list to store the data for RAGAS evaluation
    evaluation_data = []

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        print("Routing query...")
        try:
            route_info = router_chain.invoke({"query": user_query})
            if isinstance(route_info, dict):
                route = route_info.get("route", "vector_store")
                reason = route_info.get("reason", "N/A")
            else:
                route = route_info.route
                reason = route_info.reason

            print(f"Query routed to: {route.upper()} | Reason: {reason}")
            
        except Exception as e:
            print(f"Failed to route query. Defaulting to Vector Store. Error: {e}")
            route = "vector_store"
            
        selected_retriever = None
        if route == "knowledge_graph" or route == "hybrid":
            if kg_retriever:
                selected_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, kg_retriever],
                    weights=[0.5, 0.5]
                )
            else:
                print("Knowledge Graph is not available. Using Vector Store instead.")
                selected_retriever = vector_retriever
        elif route == "vector_store":
            selected_retriever = vector_retriever
        else:
            print(f"Unexpected route '{route}' returned by router. Defaulting to Vector Store.")
            selected_retriever = vector_retriever
            
        if selected_retriever:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=selected_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": final_prompt}
            )
            print("Searching and generating response...")
            result = qa_chain.invoke({"query": user_query})
            
            #  Access the retrieved context and store the data
            retrieved_context = [doc.page_content for doc in result['source_documents']]
            
            evaluation_data.append({
                "question": user_query,
                "answer": result['result'],
                "contexts": retrieved_context
            })

            print("\nFinal Answer: ", result['result'])
        else:
            print("No retriever could be selected. Please check your configuration.")

    #  Save the collected data to a JSON file
    with open("evaluation_data.json", "w") as f:
        json.dump(evaluation_data, f, indent=4)
        
    print("\nEvaluation data saved to evaluation_data.json.")

if __name__ == "__main__":
    run_chatbot()