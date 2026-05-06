import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "app": {
        "port": int(os.getenv("PORT", "3000")),
        "node_env": os.getenv("NODE_ENV", "development"),
    },
    "langgraph": {
        "model": os.getenv("LANGGRAPH_MODEL", "qwen3.5:0.8b"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "num_predict": int(os.getenv("NUM_PREDICT", "512")),
    },
    "cors": {
        "origins": os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3001").split(","),
    },
}
