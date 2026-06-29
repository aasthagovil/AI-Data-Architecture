from langchain_ollama import OllamaEmbeddings, OllamaLLM
# You can add imports for Azure here later:
# from langchain_openai import AzureChatOpenAI


def get_llm(config: dict):
    """
    Factory function to provide an LLM instance based on configuration.
    This encapsulates vendor-specific logic so the rest of your app remains clean.
    """
    llm_cfg = config["llm"]
    provider = llm_cfg.get("provider", "ollama").lower()
    model_name = llm_cfg.get("model_name", "llama3")
    temperature = llm_cfg.get("temperature", 0.0)

    if provider == "ollama":
        return OllamaLLM(
            model=model_name,
            temperature=temperature,
        )

    # Example of how you will add Azure support later:
    # elif provider == "azure":
    #     return AzureChatOpenAI(
    #         azure_deployment=llm_cfg.get("deployment_name"),
    #         openai_api_version=llm_cfg.get("api_version"),
    #         ...
    #     )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embeddings(config: dict):
    llm_cfg = config["llm"]
    provider = llm_cfg.get("provider", "ollama").lower()
    model_name = llm_cfg.get("embedding_model", "nomic-embed-text")

    if provider == "ollama":
        return OllamaEmbeddings(model=model_name)

    raise ValueError(f"Unsupported embedding provider: {provider}")
