from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import config


def get_llm(temperature: float | None = None) -> any:
    cfg = config["langgraph"]
    temp = temperature if temperature is not None else cfg["temperature"]

    if cfg["provider"] == "ollama":
        return ChatOllama(
            model=cfg["model"],
            temperature=temp,
            base_url=cfg["base_url"],
            num_predict=cfg["num_predict"],
        )
    else:
        return ChatOpenAI(
            model=cfg["model"],
            temperature=temp,
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
        )
