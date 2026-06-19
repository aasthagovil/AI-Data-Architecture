from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# 1. Load the document
loader = PyPDFLoader("data/knowledge.pdf")
pages = loader.load()
context = "\n".join([page.page_content for page in pages])

# 2. Setup the model
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template("Answer based on this: {context}\n\nQuestion: {question}")
chain = prompt | model

# 3. Query
question = "What are the key takeaways from this document?"
response = chain.invoke({"context": context, "question": question})

print(f"AI Analysis: {response}")