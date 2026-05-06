from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=0.3,
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)


class ArticleState(TypedDict):
    article: str
    keywords: list[str]
    summary: str
    log: list[str]


def build_graph():
    def extract_keywords(state: ArticleState):
        import time
        t0 = time.time()
        res = llm.invoke([
            HumanMessage(
                content=f"从以下文章提取 5-8 个核心关键词，只输出关键词，逗号分隔，不要其他内容：\n\n{state['article']}"
            )
        ])
        keywords = [k.strip() for k in res.content.split(",") if k.strip()]
        return {
            "keywords": keywords,
            "log": [f"关键词提取完成（{(time.time() - t0) * 1000:.0f}ms）"],
        }

    def generate_summary(state: ArticleState):
        import time
        t0 = time.time()
        res = llm.invoke([
            HumanMessage(
                content=f"根据以下文章生成 200 字以内的摘要。\n关键词参考：{', '.join(state['keywords'])}\n\n文章：\n{state['article']}"
            )
        ])
        return {
            "summary": res.content,
            "log": [f"摘要生成完成（{(time.time() - t0) * 1000:.0f}ms）"],
        }

    graph = StateGraph(ArticleState)
    graph.add_node("extractKeywords", extract_keywords)
    graph.add_node("generateSummary", generate_summary)
    graph.add_edge(START, "extractKeywords")
    graph.add_edge("extractKeywords", "generateSummary")
    graph.add_edge("generateSummary", END)
    return graph.compile()


graph = build_graph()


class ArticleService:
    @staticmethod
    async def process(article: str):
        result = await graph.ainvoke({"article": article})
        return {
            "keywords": result["keywords"],
            "summary": result["summary"],
            "log": result["log"],
        }
