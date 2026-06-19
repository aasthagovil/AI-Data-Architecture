# AI Data Architecture Journey
Building enterprise-grade, agentic data pipelines on Microsoft Fabric.

## 🚀 Project Status
- [x] Environment setup (uv + Python)
- [x] GitHub repository initialization
- [x] Fundamentals: Embeddings & Vector Databases
- [x] Advanced RAG implementation
- [ ] Agentic Orchestration

## 📝 Learning Diary
* **June 10, 2026:** Initialized project structure and professional Git workflow.
  **June 19, 2026:** Implemented 'Hello World' RAG pipeline using LangChain and Ollama.  
  **June 19, 2026:** Upgraded RAG pipeline to ingest PDF documents using PyPDFLoader.
   **June 20, 2026:** Upgraded RAG pipeline to a persistent vector database architecture using ChromaDB, complete with an interactive query loop.

## Project Versions
- `main.py`: The production-ready RAG agent.
- `rag_v1_basic.py`: The original prototype using hardcoded data.
- `rag_v2_pdf.py`: Implementation using `PyPDFLoader` to ingest external PDF knowledge.
-`rag_v3_vector_db.py`: Advanced RAG with persistent ChromaDB vector storage and an interactive chat loop