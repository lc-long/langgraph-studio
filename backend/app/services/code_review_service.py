import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import operator

from app.services.llm_factory import get_llm


class ReviewState(TypedDict):
    code: str
    language: str
    reviewResults: Annotated[list[dict], operator.add]
    report: str


class SingleReviewState(TypedDict):
    code: str
    language: str
    aspect: str
    prompt: str


ASPECTS = [
    {
        "aspect": "安全性",
        "prompt": '检查代码安全问题（SQL 注入、XSS、敏感信息泄露等）。\n输出 JSON（不要其他内容）：{"issues":["问题描述"],"score":7}',
    },
    {
        "aspect": "性能",
        "prompt": '检查代码性能问题（算法复杂度、N+1 查询、内存泄漏等）。\n输出 JSON（不要其他内容）：{"issues":["问题描述"],"score":7}',
    },
    {
        "aspect": "代码规范",
        "prompt": '检查代码规范（命名、注释、DRY 原则、错误处理等）。\n输出 JSON（不要其他内容）：{"issues":["问题描述"],"score":7}',
    },
]


def build_graph():
    checkpointer = MemorySaver()

    def dispatch(state: ReviewState):
        return Command(
            goto=[
                Send("reviewAgent", {
                    "code": state["code"],
                    "language": state["language"],
                    "aspect": t["aspect"],
                    "prompt": t["prompt"],
                })
                for t in ASPECTS
            ]
        )

    def review_agent(state: SingleReviewState):
        llm = get_llm(temperature=0)
        res = llm.invoke([
            HumanMessage(
                content=f"{state['prompt']}\n\n{state['language']} 代码：\n```\n{state['code']}\n```"
            )
        ])
        try:
            clean = res.content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except Exception:
            parsed = {"issues": ["结果解析失败"], "score": 5}
        return {
            "reviewResults": [{"aspect": state["aspect"], **parsed}],
        }

    def generate_report(state: ReviewState):
        llm = get_llm()
        if not state["reviewResults"]:
            return {"report": "无法生成报告"}
        avg_score = round(
            sum(r["score"] for r in state["reviewResults"])
            / len(state["reviewResults"])
        )
        detail = "\n\n".join(
            f"【{r['aspect']}】评分：{r['score']}/10\n问题：\n"
            + "\n".join(f"  - {i}" for i in r["issues"])
            for r in state["reviewResults"]
        )
        res = llm.invoke([
            HumanMessage(
                content=f"根据以下代码审查结果生成综合报告（综合评分、主要问题、改进建议）：\n\n{detail}"
            )
        ])
        return {"report": f"综合评分：{avg_score}/10\n\n{res.content}"}

    graph = StateGraph(ReviewState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("reviewAgent", review_agent)
    graph.add_node("generateReport", generate_report)
    graph.add_edge(START, "dispatch")
    graph.add_edge("reviewAgent", "generateReport")
    graph.add_edge("generateReport", END)
    return graph.compile(checkpointer=checkpointer)


graph = build_graph()


class CodeReviewService:
    @staticmethod
    async def review(code: str, language: str = "TypeScript"):
        import time
        t0 = time.time()
        thread_id = f"review-{id(code)}"
        result = await graph.ainvoke(
            {"code": code, "language": language},
            config={"configurable": {"thread_id": thread_id}}
        )
        return {
            "language": language,
            "reviewResults": result["reviewResults"],
            "report": result["report"],
            "totalTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }
