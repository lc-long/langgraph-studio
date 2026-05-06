import os
from typing import Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=config["langgraph"]["temperature"],
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)

memory_checkpointer = MemorySaver()


def build_simple_graph():
    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("callModel", call_model)
    graph.add_edge(START, "callModel")
    graph.add_edge("callModel", END)
    return graph.compile()


def build_memory_graph():
    def call_model_with_memory(state: MessagesState):
        messages = [
            SystemMessage(content="你是专业的 AI 助手，请记住对话上下文。"),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("callModel", call_model_with_memory)
    graph.add_edge(START, "callModel")
    graph.add_edge("callModel", END)
    return graph.compile(checkpointer=memory_checkpointer)


simple_graph = build_simple_graph()
memory_graph = build_memory_graph()


class LanggraphService:
    @staticmethod
    async def simple_chat(message: str) -> str:
        result = await simple_graph.ainvoke({
            "messages": [
                SystemMessage(content="你是专业的 AI 助手，回答简洁清晰。"),
                HumanMessage(content=message),
            ]
        })
        return result["messages"][-1].content

    @staticmethod
    async def memory_chat(thread_id: str, message: str) -> str:
        result = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content

    @staticmethod
    async def get_history(thread_id: str):
        state = await memory_graph.aget_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        return [
            {
                "index": i,
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for i, m in enumerate(messages)
        ]
