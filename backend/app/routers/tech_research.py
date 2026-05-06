from fastapi import APIRouter

from app.services.tech_research_service import TechResearchService

router = APIRouter(prefix="/langgraph/research", tags=["tech-research"])


@router.post("/start")
async def start(body: dict):
    return await TechResearchService.start_research(body["question"], body["threadId"])


@router.post("/{thread_id}/approve")
async def approve(thread_id: str):
    return await TechResearchService.approve(thread_id)


@router.post("/{thread_id}/revise")
async def revise(thread_id: str, body: dict):
    return await TechResearchService.revise(thread_id, body["feedback"])


@router.post("/{thread_id}/reject")
async def reject(thread_id: str):
    return await TechResearchService.reject(thread_id)


@router.get("/{thread_id}/state")
async def get_state(thread_id: str):
    return await TechResearchService.get_state(thread_id)
