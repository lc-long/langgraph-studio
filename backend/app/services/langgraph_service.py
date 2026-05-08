from typing import Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.services.llm_factory import get_llm

memory_checkpointer = MemorySaver()
_llm = None


def get_llm_instance():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def build_simple_graph():
    llm = get_llm_instance()

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("callModel", call_model)
    graph.add_edge(START, "callModel")
    graph.add_edge("callModel", END)
    return graph.compile()


def build_memory_graph():
    llm = get_llm_instance()

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
