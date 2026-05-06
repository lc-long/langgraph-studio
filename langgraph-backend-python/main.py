from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.routers.langgraph import router as langgraph_router
from app.routers.tech_research import router as tech_research_router
from app.routers.workflow import router as workflow_router

app = FastAPI(title="LangGraph NestJS 后端（Python 版）", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors"]["origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(langgraph_router)
app.include_router(tech_research_router)
app.include_router(workflow_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config["app"]["port"], reload=True)