import json
from typing import TypedDict, Literal, Any
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=config["langgraph"]["temperature"],
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)

FALLBACK_DIMENSIONS = [
    {"dimension": "技术能力与性能", "focusPoints": ["功能完整性", "并发性能", "延迟表现"]},
    {"dimension": "开发体验", "focusPoints": ["上手难度", "调试便利性", "社区文档"]},
    {"dimension": "运维与可靠性", "focusPoints": ["部署复杂度", "故障处理", "监控支持"]},
]


class TechResearchState(TypedDict):
    question: str
    researchResults: list[dict]
    analysis: str
    techOptions: list[dict]
    report: str
    humanFeedback: str
    reviewStatus: Literal["pending", "approved", "rejected", "need_revision"]
    revisionCount: int
    executionLog: list[str]


class SingleResearchState(TypedDict):
    question: str
    dimension: str
    focusPoints: list[str]


class TechResearchService:
    _graph = None

    @classmethod
    def get_graph(cls):
        if cls._graph is not None:
            return cls._graph

        checkpointer = MemorySaver()

        def safe_parse_array(raw: str, fallback: list) -> list:
            try:
                clean = raw.replace("```json", "").replace("```", "").strip()
                start = clean.index("[")
                end = clean.rindex("]")
                parsed = json.loads(clean[start : end + 1])
                if not isinstance(parsed, list) or len(parsed) == 0:
                    return fallback
                return parsed
            except Exception:
                return fallback

        def safe_parse_object(raw: str, fallback: dict) -> dict:
            try:
                clean = raw.replace("```json", "").replace("```", "").strip()
                start = clean.index("{")
                end = clean.rindex("}")
                parsed = json.loads(clean[start : end + 1])
                if not isinstance(parsed, dict):
                    return fallback
                return parsed
            except Exception:
                return fallback

        def parse_task(state: TechResearchState):
            res = llm.invoke([
                HumanMessage(
                    content=f"""你是技术分析师。把以下技术选型问题拆分成 3 个独立的调研维度。
问题：{state['question']}

严格按照以下 JSON 数组格式输出，不要输出任何其他内容，不要加说明文字：
[
  {{"dimension":"维度名称","focusPoints":["关注点1","关注点2"]}},
  {{"dimension":"维度名称","focusPoints":["关注点1","关注点2"]}},
  {{"dimension":"维度名称","focusPoints":["关注点1","关注点2"]}}
]"""
                )
            ])
            dimensions = safe_parse_array(res.content, FALLBACK_DIMENSIONS)
            return [
                Send("researchAgent", {
                    "question": state["question"],
                    "dimension": d["dimension"],
                    "focusPoints": d.get("focusPoints", ["待调研"]),
                })
                for d in dimensions[:4]
            ]

        def research_agent(state: SingleResearchState):
            res = llm.invoke([
                HumanMessage(
                    content=f"""你是技术专家。针对以下技术选型维度，给出客观分析。
原始问题：{state['question']}
当前调研维度：{state['dimension']}
重点关注：{', '.join(state['focusPoints'])}

严格按照以下 JSON 格式输出，不要输出任何其他内容：
{{"findings":"主要发现，2-3句话","pros":["优势1","优势2"],"cons":["劣势1","劣势2"]}}"""
                )
            ])
            fallback = {"findings": f"{state['dimension']}分析完成", "pros": ["待补充"], "cons": ["待补充"]}
            result = safe_parse_object(res.content, fallback)
            if not isinstance(result.get("pros"), list):
                result["pros"] = ["待补充"]
            if not isinstance(result.get("cons"), list):
                result["cons"] = ["待补充"]
            return {
                "researchResults": [{"dimension": state["dimension"], **result}],
                "executionLog": [f"✅ 完成调研：{state['dimension']}"],
            }

        def analyze_results(state: TechResearchState):
            text = "\n\n".join(
                f"【{r['dimension']}】\n发现：{r['findings']}\n优势：{', '.join(r.get('pros', []))}\n劣势：{', '.join(r.get('cons', []))}"
                for r in state["researchResults"]
            )
            res = llm.invoke([
                HumanMessage(
                    content=f"""根据以下多维度调研结果，给出综合技术分析和选型建议。
原始问题：{state['question']}
各维度调研结果：
{text}

严格按照以下 JSON 格式输出，不要输出任何其他内容：
{{"analysis":"综合结论，2-3句话","techOptions":[{{"name":"技术方案名","score":8,"bestFor":"最适合的场景"}}]}}"""
                )
            ])
            fallback = {"analysis": "综合分析完成，建议结合实际场景选型", "techOptions": []}
            result = safe_parse_object(res.content, fallback)
            if not isinstance(result.get("techOptions"), list):
                result["techOptions"] = []
            return {
                "analysis": result.get("analysis", fallback["analysis"]),
                "techOptions": result.get("techOptions", []),
                "executionLog": ["✅ 综合分析完成"],
            }

        def generate_report(state: TechResearchState):
            version_note = (
                f"\n\n重要：请根据以下修改意见重新生成报告：{state['humanFeedback']}"
                if state.get("humanFeedback")
                else ""
            )
            options_text = (
                "\n".join(
                    f"- **{t['name']}**（评分 {t.get('score', 0)}/10）：{t.get('bestFor', '')}"
                    for t in state.get("techOptions", [])
                )
                or "- 暂无明确推荐，建议结合团队实际情况决策"
            )
            res = llm.invoke([
                HumanMessage(
                    content=f"""你是技术文档专家。根据以下调研结果生成一份技术选型报告。{version_note}

原始问题：{state['question']}
综合分析：{state.get('analysis', '')}
技术选项对比：
{options_text}

要求：
- 使用 Markdown 格式
- 400-600 字
- 包含：背景说明、各维度分析摘要、技术方案对比表格、最终推荐及理由
- 语言专业，结论明确"""
                )
            ])
            revision = 1 if state.get("humanFeedback") else 0
            return {
                "report": res.content,
                "revisionCount": revision,
                "executionLog": [f"✅ 报告生成（第 {(state.get('revisionCount') or 0) + 1} 版）"],
            }

        def human_review(state: TechResearchState):
            decision = interrupt({
                "type": "report_review",
                "message": f"请审核技术选型报告（第 {(state.get('revisionCount') or 0) + 1} 版）",
                "report": state.get("report", ""),
                "meta": {
                    "question": state["question"],
                    "dimensionsCount": len(state.get("researchResults", [])),
                    "optionsCount": len(state.get("techOptions", [])),
                },
                "actions": {
                    "approve": "批准发布",
                    "revision": "需要修改（请附修改意见）",
                    "reject": "拒绝",
                },
            })
            if isinstance(decision, str):
                return {"reviewStatus": decision}
            if isinstance(decision, dict) and decision.get("action") == "revision":
                return {
                    "reviewStatus": "need_revision",
                    "humanFeedback": decision.get("feedback", ""),
                }
            return {"reviewStatus": "rejected"}

        def route_after_review(state: TechResearchState):
            if state["reviewStatus"] == "approved":
                return END
            if state["reviewStatus"] == "need_revision":
                return "generateReport"
            return END

        graph = StateGraph(TechResearchState)
        graph.add_node("parseTask", parse_task)
        graph.add_node("researchAgent", research_agent)
        graph.add_node("analyzeResults", analyze_results)
        graph.add_node("generateReport", generate_report)
        graph.add_node("humanReview", human_review)
        graph.add_edge(START, "parseTask")
        graph.add_edge("researchAgent", "analyzeResults")
        graph.add_edge("analyzeResults", "generateReport")
        graph.add_edge("generateReport", "humanReview")
        graph.add_conditional_edges(
            "humanReview",
            route_after_review,
            {"generateReport": "generateReport", END: END},
        )
        cls._graph = graph.compile(checkpointer=checkpointer)
        return cls._graph

    @staticmethod
    async def start_research(question: str, thread_id: str):
        import time
        t0 = time.time()
        graph = TechResearchService.get_graph()
        result = await graph.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50},
        )
        if result.get("__interrupt__"):
            return {
                "status": "waiting_for_review",
                "threadId": thread_id,
                "reviewData": result["__interrupt__"][0].value,
                "executionTime": f"{(time.time() - t0) * 1000:.0f}ms",
            }
        return {"status": "completed", "threadId": thread_id, "executionTime": f"{(time.time() - t0) * 1000:.0f}ms"}

    @staticmethod
    async def approve(thread_id: str):
        graph = TechResearchService.get_graph()
        await graph.ainvoke(
            Command(resume="approved"),
            config={"configurable": {"thread_id": thread_id}},
        )
        state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})
        return {
            "status": "published",
            "report": state.values.get("report"),
            "executionLog": state.values.get("executionLog"),
        }

    @staticmethod
    async def revise(thread_id: str, feedback: str):
        graph = TechResearchService.get_graph()
        result = await graph.ainvoke(
            Command(resume={"action": "revision", "feedback": feedback}),
            config={"configurable": {"thread_id": thread_id}},
        )
        if result.get("__interrupt__"):
            return {
                "status": "waiting_for_review",
                "message": "报告已修改，请重新审核",
                "reviewData": result["__interrupt__"][0].value,
            }
        return {"status": "completed"}

    @staticmethod
    async def reject(thread_id: str):
        graph = TechResearchService.get_graph()
        await graph.ainvoke(
            Command(resume="rejected"),
            config={"configurable": {"thread_id": thread_id}},
        )
        return {"status": "rejected", "message": "调研报告已拒绝"}

    @staticmethod
    async def get_state(thread_id: str):
        graph = TechResearchService.get_graph()
        state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})
        return {
            "executionLog": state.values.get("executionLog"),
            "reviewStatus": state.values.get("reviewStatus"),
            "revisionCount": state.values.get("revisionCount"),
            "nextNodes": state.next,
        }
