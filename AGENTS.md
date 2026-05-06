# AGENTS.md

## 项目结构

- `backend/` — FastAPI + LangGraph Python 后端
- `frontend/` — React 前端（Vite）

## 启动命令

```bash
# 后端
cd backend
uv venv && uv pip install -r requirements.txt
uv run python main.py

# 前端
cd frontend && pnpm install && pnpm run dev
```

## API 接口路径

所有路径已修复（FastAPI 需要 route decorator 带前导斜杠）：

### 文档一
- `POST /langgraph/simple-chat` — 无记忆问答
- `POST /langgraph/memory-chat` — 有记忆多轮对话
- `GET /langgraph/history/{thread_id}` — 查看对话历史
- `POST /langgraph/article` — 文章摘要流水线

### 文档二
- `POST /langgraph/react-chat` — ReAct Agent
- `POST /langgraph/route` — 分类路由
- `POST /langgraph/parallel` — 并行分支任务

### 文档三
- `POST /langgraph/supervisor` — Supervisor 监督模式
- `POST /langgraph/pipeline` — 内容创作流水线
- `POST /langgraph/code-review` — 并行代码审查

### 文档四
- `POST /langgraph/email/start`
- `POST /langgraph/email/{thread_id}/approve`
- `POST /langgraph/email/{thread_id}/reject`
- `POST /langgraph/email/{thread_id}/modify`
- `GET /langgraph/email/{thread_id}/state`

### 文档五
- `POST /langgraph/research/start`
- `POST /langgraph/research/{thread_id}/approve`
- `POST /langgraph/research/{thread_id}/revise`
- `POST /langgraph/research/{thread_id}/reject`
- `GET /langgraph/research/{thread_id}/state`

## 配置

`backend/.env` 中的配置优先于 `app/config.py`：

```bash
# DeepSeek（默认）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-chat
LANGGRAPH_PROVIDER=deepseek

# Ollama（备用）
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
```

## 技术栈

- **FastAPI** — Web 框架
- **LangGraph 1.x** — AI 工作流编排
- **langchain-openai** / **langchain-ollama** — LLM 集成
- **uvicorn** — ASGI 服务器
