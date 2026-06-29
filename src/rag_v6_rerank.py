from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Setup Data & Splitting
loader = PyPDFLoader("data/knowledge.pdf")
splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

# 2. Setup Hybrid Retrieval (The "Discovery" Phase)
# Vector Store for semantic search
vectorstore = Chroma.from_documents(splits, OllamaEmbeddings(model="nomic-embed-text"))

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 

# BM25 for keyword search
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 10

# Ensemble Retriever (Combines both)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], 
    weights=[0.5, 0.5]
)

# 3. Add Reranker (The "Quality Gatekeeper")
# BGE-Reranker-v2-m3 is the industry standard for lightweight, high-performance reranking
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

# The Compression Retriever wraps the ensemble, reranking its output
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# 4. Chain Configuration
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

model = OllamaLLM(model="llama3")
chain = prompt | model

# 5. History Management
def get_session_history(session_id: str):
    return ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 6. Interactive Loop
print("Advanced RAG Agent (Hybrid + Reranking) ready! Type 'exit' to stop.")
session_config = {"configurable": {"session_id": "gold_standard_session_1"}}

while True:
    user_query = input("\nYou: ")
    if user_query.lower() == "exit": break
    
    # Retrieve & Rerank
    docs = compression_retriever.invoke(user_query)
    context_text = "\n".join([d.page_content for d in docs])
    
    # Run with memory
    response = chain_with_history.invoke(
        {"question": user_query, "context": context_text},
        config=session_config
    )
    print(f"Agent: {response}")