"""仅监听本机的 Web Demo，与控制台共用 RetailRuntime。"""

from __future__ import annotations

import json
from pathlib import Path
import secrets
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .model import ModelConfig
from .budgets import DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOTAL_TOKENS, DEFAULT_MAX_REPAIRS
from .runtime import RetailRuntime
from .recovery import RUNTIME_REVISION


class WebSession:
    """一个浏览器会话拥有一个运行实例；请求并行时禁止重复写入。"""

    def __init__(self, runtime: RetailRuntime) -> None:
        self.runtime = runtime
        self.lock = threading.Lock()
        self.busy = False
        self.error: str | None = None
        self.updated = time.time()
        self.requests: dict[str, dict[str, Any]] = {}
        self.requests_lock = threading.Lock()
        self.stopping = threading.Event()

    def submit(self, content: str, request_id: str) -> dict[str, Any]:
        """在后台驱动完整工具循环；异常保留证据后停止会话。"""
        with self.requests_lock:
            existing = self.requests.get(request_id)
            if existing is not None:
                if existing["content"] != content:
                    raise ValueError("同一个 request_id 不能用于不同消息")
                return existing["receipt"]
            if len(self.requests) >= 128:
                raise ValueError("当前会话已达到请求上限，请新建会话")
            if self.runtime.state.status != "waiting_user" or not self.lock.acquire(blocking=False):
                raise ValueError("本轮正在运行或会话已停止，请等待完成或新建会话")
            self.busy = True
            receipt = {"accepted": True, "request_id": request_id, "run_id": self.runtime.state.run_id}
            self.requests[request_id] = {"content": content, "receipt": receipt}

        def work() -> None:
            try:
                self.runtime.receive(content)
                while self.runtime.state.status == "running":
                    if self.stopping.is_set():
                        self.runtime.close("web_server_shutdown")
                        break
                    message = self.runtime.decide()
                    if self.stopping.is_set():
                        self.runtime.close("web_server_shutdown")
                        break
                    if message.tool_calls:
                        self.runtime.execute(message)
            except Exception as error:
                self.error = self.runtime.model.redact(f"{type(error).__name__}: {error}")
                self.runtime.event("runtime_error", {"message": self.error})
                self.runtime.close("runtime_error")
            finally:
                self.updated = time.time()
                self.busy = False
                self.lock.release()

        threading.Thread(target=work, daemon=True, name="toolflow-turn").start()
        return receipt

    def snapshot(self) -> dict[str, Any]:
        """提供界面所需的消息与状态，不返回完整数据库或密钥。"""
        state = self.runtime.state
        messages = []
        for message in list(state.messages):
            if message.role in {"user", "assistant"} and not message.tool_calls:
                messages.append({"role": message.role, "content": message.content})
        # 逐行读取容忍写入中的最后一条 JSONL，最多展示最近 80 个事件。
        events = []
        event_path = self.runtime.output / "events.jsonl"
        if event_path.exists():
            for line in event_path.read_text(encoding="utf-8").splitlines()[-80:]:
                try:
                    record = json.loads(line)
                    data = record["data"]
                    events.append({"time": record["time"], "event": record["event"],
                                   "name": data.get("name") or data.get("call", {}).get("name"),
                                   "code": data.get("code"), "source": data.get("source")})
                except (ValueError, KeyError):
                    continue
        return {
            "run_id": state.run_id, "status": state.status, "busy": self.busy,
            "model_calls": state.model_calls, "total_tokens": state.total_tokens,
            "tool_calls": state.tool_calls, "blocked_proposals": state.blocked_proposals,
            "messages": messages, "events": events, "error": self.error,
            "backend": self.runtime.config.backend, "model": self.runtime.config.model,
            "retail_mode": self.runtime.retail_mode,
            "runtime_revision": RUNTIME_REVISION,
            "recovery_action": state.recovery_action,
        }


class DemoServer(ThreadingHTTPServer):
    """有容量限制和同源校验的本机演示服务，不是公共部署服务。"""

    daemon_threads = True

    def __init__(self, project_root: Path, configs: dict[str, ModelConfig], port: int,
                 max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
                 max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
                 max_repairs: int = DEFAULT_MAX_REPAIRS) -> None:
        super().__init__(("127.0.0.1", port), DemoHandler)
        self.project_root = project_root
        self.configs = configs
        self.token = secrets.token_urlsafe(32)
        self.sessions: dict[str, WebSession] = {}
        self.sessions_lock = threading.Lock()
        self.max_model_calls = max_model_calls
        self.max_total_tokens = max_total_tokens
        self.max_repairs = max_repairs


class DemoHandler(BaseHTTPRequestHandler):
    """固定路由，禁止任意文件下载和跨站修改会话。"""

    server: DemoServer

    def log_message(self, format: str, *args: Any) -> None:
        """不把包含会话凭证的 URL 写进终端日志。"""

    def _json(self, status: int, value: Any) -> None:
        """返回 UTF-8 JSON 并禁止缓存。"""
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _host_allowed(self) -> bool:
        """防止 DNS 重绑定，并拒绝带有异源 Origin 的访问。"""
        port = self.server.server_port
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin == "http://" + host

    def do_GET(self) -> None:
        """提供固定页面、启动元数据和当前会话快照。"""
        if not self._host_allowed():
            self._json(403, {"error": "仅允许本机同源请求"})
            return
        if self.path == "/":
            body = (Path(__file__).parent / "web_assets/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/bootstrap":
            self._json(200, {"token": self.server.token, "backends": list(self.server.configs),
                             "runtime_revision": RUNTIME_REVISION})
        elif self.path == "/api/session":
            if not self._authorized():
                return
            session_id = self.headers.get("X-Session-ID", "")
            with self.server.sessions_lock:
                session = self.server.sessions.get(session_id)
            if session is None:
                self._json(404, {"error": "会话不存在，请新建会话"})
            else:
                self._json(200, session.snapshot())
        else:
            self._json(404, {"error": "not found"})

    def _authorized(self) -> bool:
        """修改和读取私有会话均需随机令牌。"""
        supplied = self.headers.get("X-ToolFlow-Token", "")
        valid = self._host_allowed() and secrets.compare_digest(supplied, self.server.token)
        if not valid:
            self._json(403, {"error": "无效的本机请求令牌"})
        return valid

    def do_POST(self) -> None:
        """创建、提交和结束会话；不接受客户端提供模型端点或文件路径。"""
        if not self._authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("请求必须是 32KB 以内的 JSON")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求必须是 JSON 对象")
            if self.path == "/api/session":
                backend = data.get("backend", "local")
                mode = data.get("retail_mode", "official")
                if backend not in self.server.configs or mode not in {"official", "extended"}:
                    raise ValueError("无效的模型或零售模式")
                with self.server.sessions_lock:
                    if len(self.server.sessions) >= 16:
                        self._json(429, {"error": "最多保留 16 个会话，请先结束旧会话"})
                        return
                    runtime = RetailRuntime(
                        self.server.project_root, self.server.configs[backend],
                        self.server.max_model_calls, self.server.max_total_tokens,
                        retail_mode=mode,
                        store_path=self.server.project_root / "state/retail_extended.sqlite3" if mode == "extended" else None,
                        output_group="web",
                        max_repairs=self.server.max_repairs,
                    )
                    session_id = secrets.token_urlsafe(32)
                    self.server.sessions[session_id] = WebSession(runtime)
                self._json(201, {"session_id": session_id, **self.server.sessions[session_id].snapshot()})
                return
            session_id = self.headers.get("X-Session-ID", "")
            with self.server.sessions_lock:
                session = self.server.sessions.get(session_id)
            if session is None:
                self._json(404, {"error": "会话不存在"})
                return
            if self.path == "/api/message":
                content = data.get("content")
                if not isinstance(content, str) or not content.strip() or len(content) > 8000:
                    raise ValueError("消息必须是 1–8000 个字符")
                request_id = data.get("request_id")
                if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", request_id):
                    raise ValueError("需要 16–80 位字母、数字或下划线组成的 request_id，可使用 UUID")
                try:
                    receipt = session.submit(content, request_id)
                except ValueError as error:
                    self._json(409, {"error": str(error)})
                    return
                self._json(202, receipt)
            elif self.path == "/api/close":
                if not session.lock.acquire(blocking=False):
                    self._json(409, {"error": "请求执行中，请等待完成后结束"})
                    return
                try:
                    session.runtime.close("web_user_finish")
                    with self.server.sessions_lock:
                        self.server.sessions.pop(session_id, None)
                finally:
                    session.lock.release()
                self._json(200, {"closed": True})
            else:
                self._json(404, {"error": "not found"})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            # 只返回错误类型；提供方凭据和内部路径不进入浏览器错误响应。
            self._json(500, {"error": f"服务错误：{type(error).__name__}，请查看本地日志"})
