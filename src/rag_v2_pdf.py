from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from config_loader import load_config
from services.llm_provider import get_llm

config = load_config()
paths_cfg = config["paths"]

loader = PyPDFLoader(paths_cfg["knowledge_base"])
pages = loader.load()
context = "\n".join([page.page_content for page in pages])

model = get_llm(config)
prompt = ChatPromptTemplate.from_template("Answer based on this: {context}\n\nQuestion: {question}")
chain = prompt | model

question = "What are the key takeaways from this document?"
response = chain.invoke({"context": context, "question": question})

print(f"AI Analysis: {response}")
