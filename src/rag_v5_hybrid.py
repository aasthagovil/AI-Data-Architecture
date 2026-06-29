from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from config_loader import load_config
from services.llm_provider import get_embeddings, get_llm

config = load_config()
retrieval_cfg = config["retrieval"]
paths_cfg = config["paths"]
top_k = retrieval_cfg["top_k_initial"]

loader = PyPDFLoader(paths_cfg["knowledge_base"])
docs = loader.load()
splits = RecursiveCharacterTextSplitter(
    chunk_size=retrieval_cfg["chunk_size"],
    chunk_overlap=retrieval_cfg["chunk_overlap"],
).split_documents(docs)

vectorstore = Chroma.from_documents(splits, get_embeddings(config))
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = top_k

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

model = get_llm(config)
chain = prompt | model


def get_session_history(session_id: str):
    return ChatMessageHistory()


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

print("Hybrid Agent ready! Type 'exit' to stop.")
session_config = {"configurable": {"session_id": "hybrid_session_1"}}

while True:
    user_query = input("\nYou: ")
    if user_query.lower() == "exit":
        break

    docs = ensemble_retriever.invoke(user_query)
    context_text = "\n".join([d.page_content for d in docs])

    response = chain_with_history.invoke(
        {"question": user_query, "context": context_text},
        config=session_config,
    )
    print(f"Agent: {response}")
