from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import operator

from app.services.llm_factory import get_llm


class ParallelState(TypedDict):
    task: str
    results: Annotated[list[dict], operator.add]
    finalReport: str


class SubState(TypedDict):
    task: str


def build_graph():
    checkpointer = MemorySaver()

    def split_task(state: ParallelState):
        llm = get_llm()
        res = llm.invoke([
            HumanMessage(
                content=f"把以下任务拆成 3 个独立子任务，每个子任务单独一行，不要编号：\n\n{state['task']}"
            )
        ])
        sub_tasks = [t.strip() for t in res.content.split("\n") if t.strip()][:3]
        return Command(
            goto=[
                Send("processSubTask", {"task": task})
                for task in sub_tasks
            ]
        )

    def process_sub_task(state: SubState):
        llm = get_llm()
        res = llm.invoke([
            HumanMessage(content=f"完成以下任务，100 字以内：\n{state['task']}")
        ])
        return {"results": [{"task": state["task"], "result": res.content}]}

    def merge_results(state: ParallelState):
        llm = get_llm()
        text = "\n\n".join(
            f"子任务 {i + 1}：{r['task']}\n结果：{r['result']}"
            for i, r in enumerate(state["results"])
        )
        res = llm.invoke([
            HumanMessage(content=f"根据以下子任务结果，生成 200 字综合报告：\n\n{text}")
        ])
        return {"finalReport": res.content}

    graph = StateGraph(ParallelState)
    graph.add_node("splitTask", split_task)
    graph.add_node("processSubTask", process_sub_task)
    graph.add_node("mergeResults", merge_results)
    graph.add_edge(START, "splitTask")
    graph.add_edge("processSubTask", "mergeResults")
    graph.add_edge("mergeResults", END)
    return graph.compile(checkpointer=checkpointer)


graph = build_graph()


class ParallelService:
    @staticmethod
    async def run(task: str):
        import time
        t0 = time.time()
        thread_id = f"parallel-{id(task)}"
        result = await graph.ainvoke(
            {"task": task},
            config={"configurable": {"thread_id": thread_id}}
        )
        return {
            "subTasks": [r["task"] for r in result["results"]],
            "results": result["results"],
            "finalReport": result["finalReport"],
            "totalTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }
