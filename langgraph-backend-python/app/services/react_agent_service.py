from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=0,
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)


class CalculatorInput(BaseModel):
    expression: str


@tool("calculator", description="计算数学表达式，例如：(2 + 3) * 4", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression)  # noqa: S307
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool("get_weather", description="查询指定城市的当前天气")
def get_weather(city: str) -> str:
    mock = {
        "北京": "晴，25°C，东北风 3 级",
        "上海": "多云，28°C，东风 2 级",
        "武汉": "晴，30°C，南风 1 级",
        "广州": "雷阵雨，32°C，南风 2 级",
    }
    return mock.get(city, f"{city}：晴，22°C，微风")


tools = [calculator_tool, get_weather]
llm_with_tools = llm.bind_tools(tools)
memory_checkpointer = MemorySaver()


def build_graph():
    def call_model(state: MessagesState):
        messages = [
            SystemMessage(content="你是专业助手，可用工具：\n- calculator：数学计算\n- get_weather：查询天气\n根据问题决定是否调用工具。"),
            *state["messages"],
        ]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return END
        return END if not last.tool_calls else "tools"

    graph = StateGraph(MessagesState)
    graph.add_node("callModel", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "callModel")
    graph.add_conditional_edges(
        "callModel",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "callModel")
    return graph.compile(checkpointer=memory_checkpointer)


graph = build_graph()


class ReactAgentService:
    @staticmethod
    async def chat(thread_id: str, message: str) -> str:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 20,
            },
        )
        return result["messages"][-1].content
