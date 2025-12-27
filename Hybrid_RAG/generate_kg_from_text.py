# generate_kg_from_text.py

# This script is used to generate knowledge graph from text data and then store the graph into Neo4j Database

# topics added to knowledgr graph = ["Diabetes", "Influenza", "Cardiovascular disease", "Tuberculosis"]


import os
import time
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_credentials():
    """Loads credentials from the .env file."""
    load_dotenv()
    credentials = {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "neo4j_uri": os.getenv("NEO4J_URI"),
        "neo4j_username": os.getenv("NEO4J_USERNAME"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
    }
    if not all(credentials.values()):
        raise ValueError("One or more credentials not found in .env file.")
    return credentials

def generate_knowledge_graph(document, credentials):
    """Generates and stores a knowledge graph from a single document."""
    print("Initializing LLM for graph generation...")
    os.environ["GOOGLE_API_KEY"] = credentials["google_api_key"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=credentials["google_api_key"],
    )
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    # Instantiate the same embedding model used for the vector store
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Convert the single document
    print("Processing document for graph generation...")
    try:
        graph_documents = llm_transformer.convert_to_graph_documents([document])
    except Exception as e:
        print(f"Error processing the document. Error: {e}. Skipping this document.")
        graph_documents = []

    # Add embeddings to the nodes
    if graph_documents:
        print("Generating and adding embeddings to graph documents...")
        for graph_doc in graph_documents:
            for node in graph_doc.nodes:
                # Generate embedding for the node's 'name' property
                if node.properties and 'name' in node.properties:
                    node_name_embedding = embeddings_model.embed_query(node.properties['name'])
                    # Add the embedding to the node's properties
                    node.properties['embedding'] = node_name_embedding
        
        # Store the graph in Neo4j
        print(f"Connecting to Neo4j and storing the graph...")
        graph = Neo4jGraph(
            url=credentials["neo4j_uri"],
            username=credentials["neo4j_username"],
            password=credentials["neo4j_password"],
        )
        
        # Add a vector index for the new embeddings
        # The index must be created with the same embedding dimension
        graph.query("""
            CREATE VECTOR INDEX `vector-index` IF NOT EXISTS
            FOR (n:__Entity__) ON (n.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
        
        # Add the graph documents with the new embedding property
        graph.add_graph_documents(graph_documents)
        print("Knowledge graph created and stored successfully!")
    else:
        print("No graph documents were generated. Skipping database connection.")

def main():
    """Main function to orchestrate the process for a single text input."""
    try:
        credentials = get_credentials()
        
        text_content = """
        

        """
        
        if text_content:
            # Create a single Document object from the text string
            document = Document(page_content=text_content, metadata={"source": "manual_input"})
            
            # Pass the single document to the generation function
            generate_knowledge_graph(document, credentials)
        else:
            print("No text content provided. Exiting.")

    except ValueError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()