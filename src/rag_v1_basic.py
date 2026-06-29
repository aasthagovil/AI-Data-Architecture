from langchain_core.prompts import ChatPromptTemplate

from config_loader import load_config
from services.llm_provider import get_llm


def run_rag_pipeline(context, question, config):
    """
    Orchestrates the RAG process:
    1. Define the model
    2. Set up the prompt template
    3. Invoke the chain
    """
    model = get_llm(config)

    template = """
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don't know.
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    config = load_config()
    my_knowledge_base = "Microsoft Fabric is an all-in-one analytics solution."
    user_query = "What is Microsoft Fabric?"

    response = run_rag_pipeline(my_knowledge_base, user_query, config)
    print(f"AI Response: {response}")
