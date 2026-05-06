# AGENTS.md

## 项目结构

- `langgraph-backend-python/` — FastAPI + LangGraph Python 后端（端口 3000）
- `langgraph-ui/` — React 前端（Vite，端口 5173）

## 启动命令

```bash
# 后端（推荐 uv 运行）
cd langgraph-backend-python
uv venv && uv pip install -r requirements.txt
uv run python main.py

# 或用 uvicorn 直接运行
uvicorn main:app --host 0.0.0.0 --port 3000 --reload

# 前端
cd langgraph-ui && pnpm install && pnpm run dev
```

## 构建

```bash
# 后端构建（Python 无需编译，检查语法即可）
uv run python -m py_compile main.py

# 前端构建
cd langgraph-ui && pnpm run build
```

## API 接口（与原 NestJS 版兼容）

所有接口路径保持不变，前端可直接复用。

### 文档一
- `POST /langgraph/simple-chat` — 无记忆问答
- `POST /langgraph/memory-chat` — 有记忆多轮对话
- `GET /langgraph/history/{threadId}` — 查看对话历史
- `POST /langgraph/article` — 文章摘要流水线

### 文档二
- `POST /langgraph/react-chat` — ReAct Agent（工具调用）
- `POST /langgraph/route` — 分类路由
- `POST /langgraph/parallel` — 并行分支任务

### 文档三
- `POST /langgraph/supervisor` — Supervisor 监督模式
- `POST /langgraph/pipeline` — 内容创作流水线
- `POST /langgraph/code-review` — 并行代码审查

### 文档四
- `POST /langgraph/email/start` — 启动邮件审批
- `POST /langgraph/email/{threadId}/approve` — 批准
- `POST /langgraph/email/{threadId}/reject` — 拒绝
- `POST /langgraph/email/{threadId}/modify` — 要求修改
- `GET /langgraph/email/{threadId}/state` — 查看状态

### 文档五
- `POST /langgraph/research/start` — 启动调研
- `POST /langgraph/research/{threadId}/approve` — 批准发布
- `POST /langgraph/research/{threadId}/revise` — 要求修改
- `POST /langgraph/research/{threadId}/reject` — 拒绝
- `GET /langgraph/research/{threadId}/state` — 查看状态

## 重要配置

模型配置在 `app/config.py` 中，从 `.env` 读取：
- `LANGGRAPH_MODEL` — 模型名称（默认 `qwen3.5:0.8b`）
- `OLLAMA_BASE_URL` — API 地址（默认 `http://localhost:11434/v1`）
- `OLLAMA_API_KEY` — API Key（默认 `ollama`）

本地开发需 Ollama：`ollama serve` + `ollama pull qwen3.5:0.8b`

## 技术栈

- **FastAPI** — Web 框架
- **LangGraph** — AI 工作流编排（Python）
- **langchain-ollama** — Ollama 集成
- **uvicorn** — ASGI 服务器