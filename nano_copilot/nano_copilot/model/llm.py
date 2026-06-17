import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_agent_model():
    """Factory utility to switch between model providers seamlessly using environment variables."""
    provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        # Production cloud provider configuration
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    elif provider == "vllm":
        # High-performance local distributed serving engine
        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL_NAME", "qwen2.5-coder:32b"),
            temperature=0.0,
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key="vllm-token-if-configured" 
        )
        
    elif provider == "ollama":
        # Default local lightweight development workstation provider
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL_NAME", "qwen2.5-coder:7b"),
            temperature=0.0,
            format="json" # Forces the local LLM engine to strictly respect JSON schemas
        )
        
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

# Initialize your model dynamically inside your LangGraph setup
# llm = get_agent_model()

# if __name__ == "__main__":
#     prompt = """testing"""
#     print(llm.invoke(prompt))