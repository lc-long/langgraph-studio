from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=0,
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)


class RoutingState(TypedDict):
    userInput: str
    category: str
    response: str


def build_graph():
    def classify(state: RoutingState):
        res = llm.invoke([
            HumanMessage(
                content=f"把用户问题分类，只输出类别名，不要其他内容：\n- technical（技术/编程类）\n- pricing（价格/费用类）\n- general（其他）\n\n用户问题：{state['userInput']}"
            )
        ])
        cat = res.content.strip().lower()
        valid = ["technical", "pricing", "general"]
        return {"category": cat if cat in valid else "general"}

    def make_handler(system_prompt: str):
        def handler(state: RoutingState):
            res = llm.invoke([
                HumanMessage(content=f"{system_prompt}\n\n用户问题：{state['userInput']}")
            ])
            return {"response": res.content}
        return handler

    graph = StateGraph(RoutingState)
    graph.add_node("classify", classify)
    graph.add_node("technical", make_handler("你是技术专家，给出专业的技术解答。"))
    graph.add_node("pricing", make_handler("你是商务专员，友好回答，具体价格引导联系 dawei@example.com。"))
    graph.add_node("general", make_handler("你是客服，友好回答用户问题。"))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: state["category"],
        {"technical": "technical", "pricing": "pricing", "general": "general"},
    )
    graph.add_edge("technical", END)
    graph.add_edge("pricing", END)
    graph.add_edge("general", END)
    return graph.compile()


graph = build_graph()


class RoutingService:
    @staticmethod
    async def handle(user_input: str):
        result = await graph.ainvoke({"userInput": user_input})
        return {
            "input": user_input,
            "category": result["category"],
            "response": result["response"],
        }
