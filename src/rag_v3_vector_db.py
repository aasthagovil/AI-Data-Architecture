from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config_loader import load_config
from services.llm_provider import get_embeddings, get_llm

config = load_config()
retrieval_cfg = config["retrieval"]
paths_cfg = config["paths"]

loader = PyPDFLoader(paths_cfg["knowledge_base"])
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=retrieval_cfg["chunk_size"],
    chunk_overlap=retrieval_cfg["chunk_overlap"],
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=get_embeddings(config),
    persist_directory=paths_cfg["chroma_db"],
)

retriever = vectorstore.as_retriever()
model = get_llm(config)
prompt = ChatPromptTemplate.from_template("Use this context: {context}\n\nQuestion: {question}")
chain = prompt | model

print("Agent ready! Type 'exit' to stop.")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() == "exit":
        break

    relevant_docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = chain.invoke({"context": context, "question": user_query})
    print(f"Agent: {response}")
