from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=config["langgraph"]["temperature"],
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)


class SupervisorState(TypedDict):
    messages: list
    nextAgent: str
    completedAgents: list[str]


def build_graph():
    def supervisor(state: SupervisorState):
        done = (
            f"已完成：{', '.join(state['completedAgents'])}"
            if state["completedAgents"]
            else "尚未调用任何 Agent"
        )
        res = llm.invoke([
            SystemMessage(
                content=f"""你是任务协调者，管理以下专业 Agent：
- researcher：收集信息、搜索资料
- analyst：数据分析、逻辑推理
- writer：撰写报告、优化表达

规则：
1. 根据任务需求选择合适的 Agent
2. {done}
3. 所有必要工作完成后输出 FINISH
4. 只输出下一个 Agent 名称或 FINISH，不要其他内容

可选值：researcher | analyst | writer | FINISH"""
            ),
            *state["messages"],
        ])
        next_agent = res.content.strip()
        valid = ["researcher", "analyst", "writer", "FINISH"]
        safe_next = next_agent if next_agent in valid else "FINISH"
        update = {
            "nextAgent": safe_next,
            "messages": [AIMessage(content=f"[Supervisor] 下一步 → {safe_next}")],
        }
        goto = END if safe_next == "FINISH" else safe_next
        return Command(update=update, goto=goto)

    def create_worker(name: str, system_prompt: str):
        def worker(state: SupervisorState):
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
            return {
                "messages": [AIMessage(content=f"[{name}] {res.content}")],
                "completedAgents": [name],
            }
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
    return graph.compile()


graph = build_graph()


class SupervisorService:
    @staticmethod
    async def run(user_input: str):
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": 30},
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
        }
