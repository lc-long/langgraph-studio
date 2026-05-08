from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.workflow_service import WorkflowService

router = APIRouter(prefix="/workflow", tags=["workflow"])


class SaveWorkflowDto(BaseModel):
    name: str
    description: str | None = None
    nodes: list
    edges: list


class RunWorkflowDto(BaseModel):
    input: str = ""


class RunDirectDto(BaseModel):
    nodes: list
    edges: list
    input: str = ""


class TestNodeDto(BaseModel):
    nodeData: dict
    input: str = ""


@router.post("")
async def create(dto: SaveWorkflowDto):
    return WorkflowService.create(dto.model_dump())


@router.get("")
async def find_all():
    return WorkflowService.find_all()


@router.get("/{wf_id}")
async def find_one(wf_id: str):
    try:
        return WorkflowService.find_one(wf_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{wf_id}")
async def update(wf_id: str, dto: SaveWorkflowDto):
    try:
        return WorkflowService.update(wf_id, dto.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{wf_id}")
async def remove(wf_id: str):
    try:
        WorkflowService.remove(wf_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{wf_id}/run")
async def run(wf_id: str, dto: RunWorkflowDto):
    try:
        return await WorkflowService.run(wf_id, dto.input)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/run-direct")
async def run_direct(dto: RunDirectDto):
    return await WorkflowService.run_direct(dto.nodes, dto.edges, dto.input)


@router.post("/test-node")
async def test_node(dto: TestNodeDto):
    return await WorkflowService.test_node(dto.nodeData, dto.input)
