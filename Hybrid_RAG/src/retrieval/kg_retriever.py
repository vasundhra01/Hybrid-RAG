# src/retrieval/kg_retriever.py

# This script initializes the retriever to perform vector similarity searches on the Neo4j knowledge graph.

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_kg_retriever(graph, **kwargs):
    """Returns a retriever for a Neo4j knowledge graph."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    kg_retriever = Neo4jVector(
        graph=graph,
        embedding=embeddings,
        text_node_property="id",
        index_name="vector-index" 
    )
    
    return kg_retriever.as_retriever(search_type="similarity", **kwargs)