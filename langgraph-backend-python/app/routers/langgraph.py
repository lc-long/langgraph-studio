from fastapi import APIRouter

from app.services import (
    LanggraphService,
    ArticleService,
    ReactAgentService,
    RoutingService,
    ParallelService,
    SupervisorService,
    PipelineService,
    CodeReviewService,
    EmailApprovalService,
)

router = APIRouter(prefix="/langgraph", tags=["langgraph"])


@router.get("ping")
async def ping():
    import random
    return {"message": random.choice(["ping", "pong"])}


@router.post("simple-chat")
async def simple_chat(body: dict):
    message = body["message"]
    answer = await LanggraphService.simple_chat(message)
    return {"answer": answer}


@router.post("memory-chat")
async def memory_chat(body: dict):
    thread_id = body["threadId"]
    message = body["message"]
    answer = await LanggraphService.memory_chat(thread_id, message)
    return {"answer": answer}


@router.get("history/{thread_id}")
async def get_history(thread_id: str):
    return await LanggraphService.get_history(thread_id)


@router.post("article")
async def process_article(body: dict):
    return await ArticleService.process(body["article"])


@router.post("react-chat")
async def react_chat(body: dict):
    thread_id = body["threadId"]
    message = body["message"]
    answer = await ReactAgentService.chat(thread_id, message)
    return {"answer": answer}


@router.post("route")
async def route(body: dict):
    return await RoutingService.handle(body["input"])


@router.post("parallel")
async def parallel(body: dict):
    return await ParallelService.run(body["task"])


@router.post("supervisor")
async def supervisor(body: dict):
    return await SupervisorService.run(body["input"])


@router.post("pipeline")
async def pipeline(body: dict):
    return await PipelineService.create_content(body["topic"])


@router.post("code-review")
async def code_review(body: dict):
    return await CodeReviewService.review(body["code"], body.get("language"))


@router.post("email/start")
async def email_start(body: dict):
    return await EmailApprovalService.start(body["request"], body["threadId"])


@router.post("email/{thread_id}/approve")
async def email_approve(thread_id: str):
    return await EmailApprovalService.approve(thread_id)


@router.post("email/{thread_id}/reject")
async def email_reject(thread_id: str):
    return await EmailApprovalService.reject(thread_id)


@router.post("email/{thread_id}/modify")
async def email_modify(thread_id: str, body: dict):
    return await EmailApprovalService.request_modify(thread_id, body["feedback"])


@router.get("email/{thread_id}/state")
async def email_state(thread_id: str):
    return await EmailApprovalService.get_state(thread_id)
