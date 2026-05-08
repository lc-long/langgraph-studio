import uuid
from typing import Any, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.services.llm_factory import get_llm


class RunState(TypedDict):
    input: str
    output: str
    logs: list[str]
    nodeResults: dict
    condBranch: str


class WorkflowStore:
    def __init__(self):
        self.store: dict[str, dict] = {}

    def create(self, dto: dict) -> dict:
        now = __import__("datetime").datetime.now().isoformat()
        wf = {
            "id": str(uuid.uuid4()),
            "name": dto["name"],
            "description": dto.get("description"),
            "nodes": dto["nodes"],
            "edges": dto["edges"],
            "createdAt": now,
            "updatedAt": now,
        }
        self.store[wf["id"]] = wf
        return wf

    def find_all(self) -> list[dict]:
        return sorted(self.store.values(), key=lambda x: x["updatedAt"], reverse=True)

    def find_one(self, id: str) -> dict:
        wf = self.store.get(id)
        if not wf:
            raise ValueError(f"工作流 {id} 不存在")
        return wf

    def update(self, id: str, dto: dict) -> dict:
        existing = self.find_one(id)
        now = __import__("datetime").datetime.now().isoformat()
        updated = {
            **existing,
            "name": dto["name"],
            "description": dto.get("description"),
            "nodes": dto["nodes"],
            "edges": dto["edges"],
            "updatedAt": now,
        }
        self.store[id] = updated
        return updated

    def remove(self, id: str) -> None:
        if id not in self.store:
            raise ValueError(f"工作流 {id} 不存在")
        del self.store[id]


store = WorkflowStore()


def build_adj(edges: list[dict]) -> dict[str, list[dict]]:
    adj: dict[str, list[dict]] = {}
    for e in edges:
        if e["source"] not in adj:
            adj[e["source"]] = []
        adj[e["source"]].append({"target": e["target"], "handle": e.get("sourceHandle", "default")})
    return adj


def make_handler(data: dict, llm_instance: ChatOllama):
    async def handler(state: dict) -> dict:
        label = data.get("label") or data.get("nodeType", "unknown")
        node_type = data.get("nodeType", "unknown")

        if node_type == "llm":
            messages = []
            if data.get("systemPrompt"):
                from langchain_core.messages import SystemMessage, HumanMessage
                messages.append(SystemMessage(content=data["systemPrompt"]))
            else:
                from langchain_core.messages import HumanMessage
            messages.append(HumanMessage(content=state.get("output") or state.get("input", "")))
            res = await llm_instance.ainvoke(messages)
            out = res.content if hasattr(res, "content") else str(res)
            return {
                "output": out,
                "logs": [f"✅ [{label}] LLM 响应完成"],
                "nodeResults": {label: {"output": out}},
            }

        elif node_type == "agent":
            max_iter = data.get("maxIter", 3)
            context = state.get("output") or state.get("input", "")
            agent_log = ""
            from langchain_core.messages import SystemMessage, HumanMessage
            for i in range(1, max_iter + 1):
                res = await llm_instance.ainvoke([
                    SystemMessage(content=f"你是一个自主规划的 Agent。目标：{data.get('goal', '完成用户请求')}"),
                    HumanMessage(content=f"当前上下文：{context}\n这是第 {i}/{max_iter} 次迭代，请推进目标。"),
                ])
                context = res.content if hasattr(res, "content") else str(res)
                agent_log += f"\n[迭代 {i}] {str(context)[:80]}..."
            return {
                "output": context,
                "logs": [f"✅ [{label}] Agent 完成，共 {max_iter} 次迭代"],
                "nodeResults": {label: {"goal": data.get("goal"), "iterations": max_iter, "log": agent_log}},
            }

        elif node_type == "knowledge":
            query = state.get("output") or state.get("input", "")
            top_k = data.get("topK", 2)
            docs = [f"【知识库片段 {i+1}】与'{query}'相关的文档内容摘要..." for i in range(top_k)]
            return {
                "output": f"已检索到以下内容：\n" + "\n".join(docs),
                "logs": [f"✅ [{label}] 知识库检索完成，topK={top_k}"],
                "nodeResults": {label: {"query": query, "docs": docs}},
            }

        elif node_type == "http":
            method = (data.get("method") or "GET").upper()
            url = data.get("url") or ""
            if not url:
                return {"logs": [f"⚠️ [{label}] HTTP 节点未配置 URL，跳过"]}
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    headers = {"Content-Type": "application/json", **(data.get("headers") or {})}
                    body = data.get("body") if method != "GET" else None
                    async with session.request(method, url, headers=headers, json=body if body else None) as resp:
                        text = await resp.text()
                        try:
                            parsed = __import__("json").loads(text)
                        except Exception:
                            parsed = text
                        output_str = __import__("json").dumps(parsed) if not isinstance(parsed, str) else text
                        return {
                            "output": output_str,
                            "logs": [f"✅ [{label}] HTTP {method} {url} → {resp.status}"],
                            "nodeResults": {label: {"status": resp.status, "body": parsed}},
                        }
            except Exception as e:
                return {"output": "", "logs": [f"❌ [{label}] HTTP 请求失败：{e}"]}

        elif node_type == "code":
            lang = data.get("lang", "javascript")
            code = data.get("code", "")
            out = ""
            if lang == "javascript":
                try:
                    fn = __import__("Function")("inputs", code)
                    ret = fn({"input": state.get("input", ""), "output": state.get("output", "")})
                    out = __import__("json").dumps(ret) if isinstance(ret, dict) else str(ret or "")
                except Exception as e:
                    out = f"执行错误：{e}"
            else:
                out = f"[Python 代码已提交，生产环境需接入 Python Sandbox]\n```python\n{code}\n```"
            return {"output": out, "logs": [f"✅ [{label}] {lang} 代码执行完成"], "nodeResults": {label: {"lang": lang, "output": out}}}

        elif node_type == "condition":
            expr = (data.get("condition", "false")
                .replace("===", "==")
                .replace("!==", "!="))
            try:
                output_val = state.get("output", "")
                input_val = state.get("input", "")
                _eval_globals = {"output": output_val, "input": input_val}
                exec("import json", _eval_globals)
                cond_bool = bool(eval(expr, _eval_globals))
            except Exception:
                cond_bool = False
            cond_branch = "true" if cond_bool else "false"
            return {"condBranch": cond_branch, "logs": [f"✅ [{label}] 条件判断：{expr} → {cond_branch}"], "nodeResults": {label: {"expr": expr, "branch": cond_branch}}}

        else:
            return {"logs": [f"⚠️ [{label}] 未知节点类型 {node_type}，跳过"]}

    return handler


async def execute_graph(nodes: list[dict], edges: list[dict], input_str: str, name: str = "工作流"):
    adj = build_adj(edges)
    node_map = {n["id"]: n for n in nodes}

    start_node = next((n for n in nodes if n["data"]["nodeType"] == "start"), None)
    if not start_node:
        raise ValueError("工作流缺少【开始】节点")

    graph = StateGraph(RunState)

    llm_instance = get_llm()

    for node in nodes:
        nt = node["data"]["nodeType"]
        if nt == "start" or nt == "end":
            continue
        handler = make_handler(node["data"], llm_instance)
        graph.add_node(node["id"], handler)

    first_edge = adj.get(start_node["id"])
    if not first_edge:
        raise ValueError("开始节点没有连接任何后续节点")

    first_target_id = first_edge[0]["target"]
    first_node = node_map.get(first_target_id)

    if first_node and first_node["data"]["nodeType"] == "end":
        async def passthrough(state: dict) -> dict:
            return {"output": state["input"], "logs": ["✅ 直通节点（start→end）"]}
        graph.add_node("__passthrough__", passthrough)
        graph.add_edge(START, "__passthrough__")
        graph.add_edge("__passthrough__", END)
    else:
        graph.add_edge(START, first_target_id)
        _connect_edges(graph, adj, node_map, first_target_id, set())

    compiled = graph.compile()
    return await compiled.ainvoke({"input": input_str, "logs": [f"▶ 开始执行工作流：{name}"], "nodeResults": {}})


def _connect_edges(graph: StateGraph, adj: dict, node_map: dict, current_id: str, visited: set):
    if current_id in visited:
        return
    visited.add(current_id)

    nexts = adj.get(current_id, [])
    node = node_map.get(current_id)

    if node and node["data"]["nodeType"] == "condition" and len(nexts) > 0:
        default_next = next((n for n in nexts if n["handle"] == "default"), None)
        else_next = next((n for n in nexts if n["handle"] == "else"), None)

        def resolve_target(next_opt: dict | None):
            if not next_opt:
                return END
            t = node_map.get(next_opt["target"])
            return END if not t or t["data"]["nodeType"] == "end" else next_opt["target"]

        def route_cond(state: dict) -> str:
            return state.get("condBranch", "false")

        graph.add_conditional_edges(current_id, route_cond, {"true": resolve_target(default_next), "false": resolve_target(else_next)})

        if default_next:
            t = node_map.get(default_next["target"])
            if t and t["data"]["nodeType"] != "end":
                _connect_edges(graph, adj, node_map, default_next["target"], visited)
        if else_next:
            t = node_map.get(else_next["target"])
            if t and t["data"]["nodeType"] != "end":
                _connect_edges(graph, adj, node_map, else_next["target"], visited)
    else:
        for next_edge in nexts:
            next_node = node_map.get(next_edge["target"])
            if next_node and next_node["data"]["nodeType"] == "end":
                graph.add_edge(current_id, END)
            else:
                graph.add_edge(current_id, next_edge["target"])
                _connect_edges(graph, adj, node_map, next_edge["target"], visited)
        if not nexts:
            graph.add_edge(current_id, END)


class WorkflowService:
    @staticmethod
    def create(dto: dict) -> dict:
        return store.create(dto)

    @staticmethod
    def find_all() -> list[dict]:
        return store.find_all()

    @staticmethod
    def find_one(id: str) -> dict:
        return store.find_one(id)

    @staticmethod
    def update(id: str, dto: dict) -> dict:
        return store.update(id, dto)

    @staticmethod
    def remove(id: str) -> None:
        store.remove(id)

    @staticmethod
    async def run(id: str, input_str: str) -> dict:
        import time
        t0 = time.time()
        wf = store.find_one(id)
        result = await execute_graph(wf["nodes"], wf["edges"], input_str, wf["name"])
        return {
            "workflowId": id,
            "input": input_str,
            "output": result.get("output", input_str),
            "logs": result.get("logs", []),
            "nodeResults": result.get("nodeResults", {}),
            "executionTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }

    @staticmethod
    async def run_direct(nodes: list[dict], edges: list[dict], input_str: str = "") -> dict:
        import time
        t0 = time.time()
        result = await execute_graph(nodes, edges, input_str, "临时工作流")
        return {
            "workflowId": "direct",
            "input": input_str,
            "output": result.get("output", input_str),
            "logs": result.get("logs", []),
            "nodeResults": result.get("nodeResults", {}),
            "executionTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }

    @staticmethod
    async def test_node(node_data: dict, input_str: str = "") -> dict:
        import time
        t0 = time.time()
        llm_instance = get_llm()
        handler = make_handler(node_data, llm_instance)
        fake_state = {"input": input_str, "output": input_str, "logs": [], "nodeResults": {}, "condBranch": "false"}
        patch = await handler(fake_state)
        http_status = None
        http_headers = None
        if node_data.get("nodeType") == "http" and node_data.get("url"):
            try:
                import aiohttp
                method = (node_data.get("method") or "GET").upper()
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, node_data["url"]) as resp:
                        http_status = resp.status
                        http_headers = {}
                        for k, v in resp.headers.items():
                            http_headers[k] = v
            except Exception:
                pass
        detail = None
        if patch.get("nodeResults"):
            vals = list(patch["nodeResults"].values())
            detail = vals[0] if vals else None
        result = {
            "input": input_str,
            "output": patch.get("output", input_str),
            "logs": patch.get("logs", []),
            "detail": detail,
            "executionTime": f"{(time.time() - t0) * 1000:.0f}ms",
        }
        if http_status is not None:
            result["httpStatus"] = http_status
        if http_headers is not None:
            result["httpHeaders"] = http_headers
        return result