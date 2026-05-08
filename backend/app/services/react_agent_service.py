from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel
import os
import sys
import tempfile
import uuid

from app.services.llm_factory import get_llm


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


class WebSearchInput(BaseModel):
    query: str


@tool("web_search", description="搜索互联网获取相关信息", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo"""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=5)):
                results.append(f"[{i+1}] {r['title']}\n   {r['href']}\n   {r['body'][:200]}")
        if not results:
            return "未找到相关结果"
        return "搜索结果：\n\n" + "\n\n".join(results)
    except ImportError:
        return "搜索功能需要 duckduckgo-search 库，请联系管理员安装"
    except Exception as e:
        return f"搜索失败：{str(e)}"


class WebFetchInput(BaseModel):
    url: str
    max_length: int = 2000


@tool("web_content_fetch", description="获取指定URL的网页内容", args_schema=WebFetchInput)
def web_content_fetch(url: str, max_length: int = 2000) -> str:
    """Fetch and parse content from a URL"""
    try:
        import aiohttp
        import asyncio

        async def _fetch():
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    text = await resp.text()
                    return text[:max_length]

        content = asyncio.run(_fetch())
        return f"内容已获取（前 {len(content)} 字符）：\n\n{content}"
    except Exception as e:
        return f"获取失败：{str(e)}"


class FileReadInput(BaseModel):
    file_path: str


@tool("file_read", description="读取本地文件内容", args_schema=FileReadInput)
def file_read(file_path: str) -> str:
    """Read content from a local file"""
    try:
        safe_path = os.path.abspath(file_path)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if not safe_path.startswith(project_root):
            return "错误：禁止访问项目目录之外的文件"
        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"文件内容（{len(content)} 字符）：\n\n{content[:3000]}"
    except FileNotFoundError:
        return f"文件不存在：{file_path}"
    except PermissionError:
        return f"权限拒绝：{file_path}"
    except Exception as e:
        return f"读取失败：{str(e)}"


class CodeInterpreterInput(BaseModel):
    code: str
    language: str = "python"


@tool("code_interpreter", description="执行Python代码并返回结果", args_schema=CodeInterpreterInput)
def code_interpreter(code: str, language: str = "python") -> str:
    """Execute Python code in a sandboxed environment"""
    if language != "python":
        return f"仅支持Python语言，当前语言：{language}"

    import re
    forbidden = ["import os", "import sys", "import subprocess", "import socket", "open(", "eval(", "exec("]
    for f in forbidden:
        if f in code:
            return f"错误：禁止使用 {f}"

    output = []
    try:
        import io
        sys.stdout = io.StringIO()
        exec(code, {"__builtins__": __builtins__})
        result = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
        if result:
            output.append(result)
        return "执行成功：\n" + "\n".join(output) if output else "执行完成，无输出"
    except SyntaxError as e:
        return f"语法错误：{e}"
    except Exception as e:
        return f"执行错误：{type(e).__name__}: {e}"
    finally:
        sys.stdout = sys.__stdout__


tools = [calculator_tool, get_weather, web_search, web_content_fetch, file_read, code_interpreter]
memory_checkpointer = MemorySaver()


def build_graph():
    def call_model(state: MessagesState):
        llm = get_llm(temperature=0)
        llm_with_tools = llm.bind_tools(tools)
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
