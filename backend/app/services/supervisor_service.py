from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import operator

from app.services.llm_factory import get_llm


class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    nextAgent: str
    completedAgents: Annotated[list[str], operator.add]
    step: str


def build_graph():
    WORKFLOW_STEPS = [
        ("researcher", "你是研究员，擅长收集整理信息。", "正在收集资料，请稍候..."),
        ("analyst", "你是分析师，擅长数据分析和推理。", "正在分析数据，请稍候..."),
        ("writer", "你是写作专家，擅长生成清晰报告。", "正在撰写报告，请稍候..."),
        ("FINISH", "", "工作流已完成。"),
    ]

    def supervisor(state: SupervisorState):
        done = state.get("completedAgents", [])
        step = state.get("step", "start")

        step_order = ["start", "researcher", "analyst", "writer", "FINISH"]
        if step not in step_order:
            step = "start"

        current_idx = step_order.index(step) if step in step_order else 0
        next_step = step_order[current_idx + 1] if current_idx + 1 < len(step_order) else "FINISH"

        next_name = next_step if next_step != "FINISH" else "FINISH"
        log_msg = f"[Supervisor] 下一步 → {next_name}"

        return Command(
            update={
                "nextAgent": next_name,
                "step": next_step,
                "messages": [AIMessage(content=log_msg)],
            },
            goto=END if next_step == "FINISH" else next_step,
        )

    def create_worker(name: str, system_prompt: str):
        def worker(state: SupervisorState):
            llm = get_llm()
            user_msg = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                "",
            )
            context = "\n".join(
                m.content for m in state["messages"][-4:] if isinstance(m, AIMessage)
            )
            res = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"任务：{user_msg}\n\n当前上下文：\n{context}"),
            ])
            return Command(
                update={
                    "messages": [AIMessage(content=f"[{name}] {res.content}")],
                    "completedAgents": [name],
                },
                goto="supervisor",
            )
        return worker

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", create_worker("researcher", "你是研究员，擅长收集整理信息。"))
    graph.add_node("analyst", create_worker("analyst", "你是分析师，擅长数据分析和推理。"))
    graph.add_node("writer", create_worker("writer", "你是写作专家，擅长生成清晰报告。"))
    graph.add_edge(START, "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile(checkpointer=MemorySaver())


graph = build_graph()


class SupervisorService:
    @staticmethod
    async def run(user_input: str):
        import time
        t0 = time.time()
        thread_id = f"supervisor-{id(user_input)}"
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 30,
            },
        )
        agent_logs = [
            m.content
            for m in result["messages"]
            if isinstance(m, AIMessage) and m.content.startswith("[")
        ]
        writers = [m for m in agent_logs if m.startswith("[writer]")]
        final_report = (
            writers[-1].replace("[writer] ", "")
            if writers
            else (agent_logs[-1] if agent_logs else "无输出")
        )
        return {
            "agentLog": agent_logs,
            "completedAgents": result["completedAgents"],
            "finalReport": final_report,
            "totalTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }
