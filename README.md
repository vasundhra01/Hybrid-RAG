# Hybrid Retrieval-Augmented Generation (Hybrid RAG)

## Overview
This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** system aimed at
improving **multi-hop question answering** by combining vector-based semantic retrieval with
knowledge-graph-based reasoning.

Traditional RAG systems that rely only on vector similarity search often fail on queries that
require reasoning across multiple related entities. This project explores a hybrid approach
that integrates unstructured text retrieval with structured knowledge graphs and dynamically
selects the most suitable retrieval strategy per query.

---

## Problem Statement
Vector-based RAG systems perform well for descriptive queries but struggle with complex
multi-hop questions. Knowledge graphs can handle structured relationships effectively but
lack the contextual richness of unstructured text.

The goal of this project is to combine the strengths of both approaches to produce more
accurate, grounded, and reliable answers.

---

## System Architecture
The system consists of three main components:
- **Vector Store Retrieval** for semantic similarity search over unstructured text
- **Knowledge Graph Retrieval** for structured, multi-hop reasoning over entities and relationships
- **LLM-based Query Router** that dynamically selects vector, graph, or hybrid retrieval
  based on the query type

Retrieved context from the selected retriever(s) is passed to a large language model for
answer generation.

---

## Technologies Used
- **Vector Store:** ChromaDB  
- **Knowledge Graph:** Neo4j (AuraDB)  
- **Embeddings:** all-MiniLM-L6-v2 (Hugging Face)  
- **LLM:** Gemini 2.5 Flash  
- **Frameworks / Tools:** LangChain, Python  
- **Evaluation:** RAGAS  

---

## Methodology
- Documents are chunked and embedded for storage in a vector database
- Structured entities and relationships are extracted and stored in a knowledge graph
- Queries are analyzed and routed dynamically to the appropriate retriever
- Retrieved context is aggregated and used for grounded answer generation

---

## Evaluation
The system is evaluated against a traditional vector-only RAG baseline using the **RAGAS**
framework.

Evaluation metrics include:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

The hybrid approach demonstrates higher faithfulness and recall for multi-hop queries,
with fewer retrieval failures compared to vector-only RAG.

---

## Limitations
- This is a research prototype, not a production-ready system
- Knowledge graph construction requires curated, domain-specific data
- Latency is higher than vector-only RAG due to multiple retrieval paths

---

## Future Work
- Improve query routing accuracy
- Optimize retrieval performance
- Extend the approach to additional domains
- Explore alternative hybrid weighting strategies

---

## Conclusion
This project demonstrates that combining vector-based retrieval with knowledge-graph
reasoning can significantly improve the robustness of Retrieval-Augmented Generation
systems for multi-hop question answering tasks.
