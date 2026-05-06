from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=config["langgraph"]["temperature"],
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)


class PipelineState(TypedDict):
    topic: str
    research: str
    outline: str
    draft: str
    finalArticle: str
    progress: list[str]


def build_graph():
    def research_agent(state: PipelineState):
        res = llm.invoke([
            HumanMessage(
                content=f"""你是研究员，为主题"{state['topic']}"收集素材：
1. 背景介绍（2-3 句）
2. 核心要点（3-5 个）
3. 典型案例（1-2 个）
每条不超过 50 字。"""
            )
        ])
        return {"research": res.content, "progress": ["✅ 素材收集完成"]}

    def outline_agent(state: PipelineState):
        res = llm.invoke([
            HumanMessage(
                content=f"""你是内容策划，根据素材为"{state['topic']}"生成大纲：
素材：{state['research']}
格式：# 章节 / - 子项，共 3-5 章"""
            )
        ])
        return {"outline": res.content, "progress": ["✅ 大纲生成完成"]}

    def writing_agent(state: PipelineState):
        res = llm.invoke([
            HumanMessage(
                content=f"""你是撰稿人，根据大纲写文章（400-600 字）：
主题：{state['topic']}
大纲：{state['outline']}
参考素材：{state['research']}"""
            )
        ])
        return {"draft": res.content, "progress": ["✅ 初稿写作完成"]}

    def review_agent(state: PipelineState):
        res = llm.invoke([
            HumanMessage(
                content=f"你是编辑，优化以下文章，直接输出优化后全文：\n{state['draft']}"
            )
        ])
        return {"finalArticle": res.content, "progress": ["✅ 审校优化完成"]}

    graph = StateGraph(PipelineState)
    graph.add_node("researchAgent", research_agent)
    graph.add_node("outlineAgent", outline_agent)
    graph.add_node("writingAgent", writing_agent)
    graph.add_node("reviewAgent", review_agent)
    graph.add_edge(START, "researchAgent")
    graph.add_edge("researchAgent", "outlineAgent")
    graph.add_edge("outlineAgent", "writingAgent")
    graph.add_edge("writingAgent", "reviewAgent")
    graph.add_edge("reviewAgent", END)
    return graph.compile()


graph = build_graph()


class PipelineService:
    @staticmethod
    async def create_content(topic: str):
        import time
        t0 = time.time()
        result = await graph.ainvoke({"topic": topic})
        return {
            "topic": topic,
            "progress": result["progress"],
            "finalArticle": result["finalArticle"],
            "totalTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }
