from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Setup Vector Store (Existing Logic)
loader = PyPDFLoader("data/knowledge.pdf")
splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
vectorstore = Chroma.from_documents(splits, OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

# 2. Setup Prompt with Memory Placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

model = OllamaLLM(model="llama3")

# 3. Create a chain that includes history
def get_session_history(session_id: str):
    return ChatMessageHistory()

chain = prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 4. Interactive Loop
print("Agent with Memory ready! Type 'exit' to stop.")
session_config = {"configurable": {"session_id": "unique_session_1"}}

while True:
    user_query = input("\nYou: ")
    if user_query.lower() == "exit": break
    
    # Retrieve context
    docs = retriever.invoke(user_query)
    context_text = "\n".join([d.page_content for d in docs])
    
    # Run with history
    response = chain_with_history.invoke(
        {"question": user_query, "context": context_text},
        config=session_config
    )
    print(f"Agent: {response}")