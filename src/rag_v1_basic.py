from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def run_rag_pipeline(context, question):
    """
    Orchestrates the RAG process: 
    1. Define the model 
    2. Set up the prompt template 
    3. Invoke the chain
    """
    # 1. Initialize the local model
    model = OllamaLLM(model="llama3")

    # 2. Define how the AI should use the context
    template = """
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don't know.
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Create and run the chain
    chain = prompt | model
    return chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    # Your 'Gold Layer' source data
    my_knowledge_base = "Microsoft Fabric is an all-in-one analytics solution."
    user_query = "What is Microsoft Fabric?"
    
    response = run_rag_pipeline(my_knowledge_base, user_query)
    print(f"AI Response: {response}")