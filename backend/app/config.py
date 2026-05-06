import os
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if deepseek_api_key:
    default_provider = "deepseek"
    default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    default_base_url = "https://api.deepseek.com/v1"
    default_api_key = deepseek_api_key
else:
    default_provider = "ollama"
    default_model = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
    default_base_url = ollama_base_url
    default_api_key = os.getenv("OLLAMA_API_KEY", "ollama")

config = {
    "app": {
        "port": int(os.getenv("PORT", "3000")),
        "node_env": os.getenv("NODE_ENV", "development"),
    },
    "langgraph": {
        "provider": os.getenv("LANGGRAPH_PROVIDER", default_provider),
        "model": os.getenv("LANGGRAPH_MODEL", default_model),
        "base_url": os.getenv("LANGGRAPH_BASE_URL", default_base_url),
        "api_key": os.getenv("LANGGRAPH_API_KEY", default_api_key),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "num_predict": int(os.getenv("NUM_PREDICT", "8192")),
    },
    "cors": {
        "origins": os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3001").split(","),
    },
}
