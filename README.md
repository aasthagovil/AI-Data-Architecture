# AI Data Architecture: Advanced RAG Agent
A professional-grade Retrieval-Augmented Generation (RAG) agent featuring persistent vector storage, hybrid search, and multi-stage reranking.

## Project Status
- [x] Environment setup (uv + Python)
- [x] Fundamentals: Vector Databases (ChromaDB)
- [x] Advanced RAG: Conversational Memory
- [x] Production-Ready: Hybrid Search (BM25 + Vector)
- [x] Precision: Cross-Encoder Reranking
- [ ] Final Milestone: Agentic Orchestration (Tool Use)

## Architectural Overview
This pipeline uses a two-stage retrieval process to maximize accuracy:
1. **Discovery (Hybrid Search)**: Combines Semantic Search (`nomic-embed-text`) with Lexical Search (`BM25`) to ensure both intent and keyword precision.
2. **Refinement (Reranking)**: Uses a Cross-Encoder (`BGE-Reranker-v2-m3`) to filter retrieved context, ensuring only the most relevant snippets reach the LLM.

## 📝 Learning Diary
* **June 10, 2026:** Initialized project structure and professional Git workflow.
  **June 19, 2026:** Implemented 'Hello World' RAG pipeline using LangChain and Ollama.  
  **June 19, 2026:** Upgraded RAG pipeline to ingest PDF documents using PyPDFLoader.
   **June 20, 2026:** Upgraded RAG pipeline to a persistent vector database architecture using ChromaDB, complete with an interactive query loop.
  - **June 29, 2026**: Completed Advanced RAG pipeline. Optimized retrieval with Hybrid Search and integrated a Cross-Encoder reranker to act as an enterprise-grade "Quality Gatekeeper."

## Project Versions
- `main.py`: The production-ready RAG agent.
- `rag_v1_basic.py`: The original prototype using hardcoded data.
- `rag_v2_pdf.py`: Implementation using `PyPDFLoader` to ingest external PDF knowledge.
- `rag_v3_vector_db.py`: Advanced RAG with persistent ChromaDB vector storage and an interactive chat loop
- `rag_v4_memory.py`: Added `ChatMessageHistory` for multi-turn stateful conversations.
- `rag_v5_hybrid.py`: Integrated `EnsembleRetriever` to balance semantic and keyword retrieval.
- `rag_v6_rerank.py`: Finalized architecture with `ContextualCompressionRetriever` to reduce noise and LLM hallucination.