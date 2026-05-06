import json
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage

from app.config import config

llm = ChatOllama(
    model=config["langgraph"]["model"],
    temperature=config["langgraph"]["temperature"],
    base_url=config["langgraph"]["base_url"],
    num_predict=config["langgraph"]["num_predict"],
)

memory_checkpointer = None


class EmailState(TypedDict):
    emailRequest: str
    draftEmail: dict
    approvalStatus: Literal["pending", "approved", "rejected", "need_modify"]
    modifyFeedback: str
    revisionCount: int
    finalStatus: str


_graph_instance = None


def get_graph():
    global memory_checkpointer
    if memory_checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        memory_checkpointer = MemorySaver()

    def draft_node(state: EmailState):
        is_revision = bool(state.get("modifyFeedback"))
        prompt = (
            f"根据修改意见重新起草邮件：\n"
            f"修改意见：{state['modifyFeedback']}\n"
            f"原始需求：{state['emailRequest']}\n"
            f"上次草稿：{json.dumps(state.get('draftEmail', {}))}"
            if is_revision
            else f"根据需求起草一封专业邮件：{state['emailRequest']}"
        )
        res = llm.invoke([
            HumanMessage(
                content=f"{prompt}\n\n输出 JSON（不要其他内容）：\n"
                + '{"subject":"邮件主题","recipient":"收件人","body":"正文内容"}'
            )
        ])
        try:
            clean = res.content.replace("```json", "").replace("```", "").strip()
            draft = json.loads(clean)
        except Exception:
            draft = {"subject": "草稿", "recipient": "未知", "body": res.content}
        return {
            "draftEmail": draft,
            "approvalStatus": "pending",
            "revisionCount": 1 if is_revision else 0,
        }

    def wait_node(state: EmailState):
        decision = interrupt(
            {
                "type": "email_review",
                "message": f"请审查邮件草稿（第 {(state.get('revisionCount') or 0) + 1} 版）",
                "draft": state.get("draftEmail", {}),
                "options": {
                    "approve": "批准发送",
                    "reject": "拒绝（取消发送）",
                    "modify": "需要修改（附修改意见）",
                },
            }
        )
        if isinstance(decision, str):
            return {"approvalStatus": decision}
        if isinstance(decision, dict) and decision.get("action") == "modify":
            return {
                "approvalStatus": "need_modify",
                "modifyFeedback": decision.get("feedback", ""),
            }
        return {"approvalStatus": "rejected"}

    def route_after_approval(state: EmailState):
        if state["approvalStatus"] == "approved":
            return "sendNode"
        if state["approvalStatus"] == "need_modify":
            return "draftNode"
        return "cancelNode"

    def send_node(state: EmailState):
        draft = state.get("draftEmail", {})
        return {
            "finalStatus": (
                f"✅ 邮件已发送\n"
                f"收件人：{draft.get('recipient', '未知')}\n"
                f"主题：{draft.get('subject', '无')}"
            ),
        }

    def cancel_node(state: EmailState):
        return {
            "finalStatus": f"❌ 邮件已取消（审批状态：{state['approvalStatus']}）",
        }

    graph = StateGraph(EmailState)
    graph.add_node("draftNode", draft_node)
    graph.add_node("waitNode", wait_node)
    graph.add_node("sendNode", send_node)
    graph.add_node("cancelNode", cancel_node)
    graph.add_edge(START, "draftNode")
    graph.add_edge("draftNode", "waitNode")
    graph.add_conditional_edges(
        "waitNode",
        route_after_approval,
        {"sendNode": "sendNode", "draftNode": "draftNode", "cancelNode": "cancelNode"},
    )
    graph.add_edge("sendNode", END)
    graph.add_edge("cancelNode", END)
    return graph.compile(checkpointer=memory_checkpointer)


class EmailApprovalService:
    _graph = None

    @classmethod
    def get_graph(cls):
        if cls._graph is None:
            cls._graph = get_graph()
        return cls._graph

    @staticmethod
    async def start(email_request: str, thread_id: str):
        graph = EmailApprovalService.get_graph()
        result = await graph.ainvoke(
            {"emailRequest": email_request},
            config={"configurable": {"thread_id": thread_id}},
        )
        if result.get("__interrupt__"):
            return {
                "status": "waiting_for_approval",
                "threadId": thread_id,
                "reviewData": result["__interrupt__"][0].value,
                "message": "邮件草稿已生成，请审批",
            }
        return {"status": "completed", "result": result}

    @staticmethod
    async def approve(thread_id: str):
        graph = EmailApprovalService.get_graph()
        await graph.ainvoke(
            Command(resume="approved"),
            config={"configurable": {"thread_id": thread_id}},
        )
        state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})
        return {"status": "email_sent", "finalStatus": state.values.get("finalStatus")}

    @staticmethod
    async def reject(thread_id: str):
        graph = EmailApprovalService.get_graph()
        await graph.ainvoke(
            Command(resume="rejected"),
            config={"configurable": {"thread_id": thread_id}},
        )
        return {"status": "cancelled", "message": "邮件已取消发送"}

    @staticmethod
    async def request_modify(thread_id: str, feedback: str):
        graph = EmailApprovalService.get_graph()
        result = await graph.ainvoke(
            Command(resume={"action": "modify", "feedback": feedback}),
            config={"configurable": {"thread_id": thread_id}},
        )
        if result.get("__interrupt__"):
            return {
                "status": "waiting_for_approval",
                "reviewData": result["__interrupt__"][0].value,
                "message": "邮件已修改，请重新审批",
            }
        return {"status": "completed"}

    @staticmethod
    async def get_state(thread_id: str):
        graph = EmailApprovalService.get_graph()
        state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})
        return state.values
