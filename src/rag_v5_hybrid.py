from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Setup Data
loader = PyPDFLoader("data/knowledge.pdf")
docs = loader.load()
splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

# 2. Initialize Retrievers
# Vector Store for semantic retrieval
vectorstore = Chroma.from_documents(splits, OllamaEmbeddings(model="nomic-embed-text"))
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# BM25 for keyword retrieval
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2

# Ensemble Retriever (Hybrid Search)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], 
    weights=[0.5, 0.5]
)

# 3. Setup Prompt & Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

model = OllamaLLM(model="llama3")
chain = prompt | model

# 4. History Management
def get_session_history(session_id: str):
    return ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 5. Interactive Loop
print("Hybrid Agent ready! Type 'exit' to stop.")
session_config = {"configurable": {"session_id": "hybrid_session_1"}}

while True:
    user_query = input("\nYou: ")
    if user_query.lower() == "exit": break
    
    # Retrieve using the Ensemble Retriever
    docs = ensemble_retriever.invoke(user_query)
    context_text = "\n".join([d.page_content for d in docs])
    
    # Invoke with memory
    response = chain_with_history.invoke(
        {"question": user_query, "context": context_text},
        config=session_config
    )
    print(f"Agent: {response}")