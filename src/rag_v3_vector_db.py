from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. Load and Split
loader = PyPDFLoader("data/knowledge.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Embed and Store in Vector DB
# Note: This creates a folder named 'chroma_db' in your project
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./chroma_db"
)

# 3. Setup the Chain
retriever = vectorstore.as_retriever()
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template("Use this context: {context}\n\nQuestion: {question}")
chain = prompt | model

# 4. The Interactive Loop
print("Agent ready! Type 'exit' to stop.")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() == "exit":
        break
    
    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(user_query)
    
    # Generate response
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = chain.invoke({"context": context, "question": user_query})
    print(f"Agent: {response}")