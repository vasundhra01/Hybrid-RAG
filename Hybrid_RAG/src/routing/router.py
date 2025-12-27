# src/routing/router.py

# This script uses an LLM to decide whether to retrieve answers from the knowledge graph, the vector store, or both, based on the user's question.

import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class RouteQuery(BaseModel):
    """Router for a user's query."""
    route: Literal["knowledge_graph", "vector_store", "hybrid"] = Field(
        ...,
        description="Classify the user's query based on its content."
    )
    reason: str = Field(
        ...,
        description="A brief explanation of why this route was chosen."
    )

def get_router_chain():
    """Returns a LangChain Expression Language (LCEL) chain for query routing."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )
    
    # We will use the structured output method which is often more robust
    # than a parser for this use case.
    llm_with_structured_output = llm.with_structured_output(RouteQuery)

    prompt = ChatPromptTemplate.from_template("""
        You are an expert query router for a healthcare knowledge base. Your task is to classify a user's query as either 'knowledge_graph', 'vector_store', or 'hybrid'.
        - 'knowledge_graph' is for queries that require specific facts or relationships between medical entities (e.g., diseases, drugs, symptoms).
        - 'vector_store' is for queries that are broad, descriptive, or require a general understanding of a topic from a single passage of text.
        - 'hybrid' is for queries that could benefit from both structured facts and broad context.

        Query: {query}
    """)
    
    # The chain now passes the prompt to the LLM with structured output
    chain = prompt | llm_with_structured_output
    
    return chain