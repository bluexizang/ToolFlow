"""按需管理本机已有 Ollama 安装的生命周期。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

import psutil

from .local_config import DEFAULT_LOCAL_MODEL


class LocalServer:
    """只启动已有的资产，且只停止由本进程拉起的服务。"""

    def __init__(self, project: Path) -> None:
        self.project = project
        self.process: subprocess.Popen[bytes] | None = None
        self.log: Any = None
        self.opener = build_opener(ProxyHandler({}))
        self.model = DEFAULT_LOCAL_MODEL

    def request(self, path: str, body: dict[str, Any] | None = None) -> Any:
        """发起本地元数据/控制请求，不经过代理转发。"""
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            "http://127.0.0.1:11434" + path, data=data,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=5) as response:
            return json.load(response)

    def ensure_ready(self, start: bool = False, model: str = DEFAULT_LOCAL_MODEL) -> None:
        """检查已有服务，或仅在显式要求时以后台隐藏方式启动它。"""
        if not re.fullmatch(r"[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?", model):
            raise ValueError("Only an installed local model name:tag is supported")
        self.model = model
        try:
            self.request("/api/version")
        except (OSError, ValueError):
            if not start:
                raise RuntimeError("Local Ollama is stopped; use --start-server") from None
            candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
                          Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"]
            configured = os.environ.get("TOOLFLOW_OLLAMA_EXE")
            executable = Path(configured) if configured else next((p for p in candidates if p.is_file()), candidates[0])
            models = Path(os.environ.get("TOOLFLOW_OLLAMA_MODELS", r"D:\OllamaLocal\models"))
            name, separator, tag = model.partition(":")
            manifest = models / "manifests/registry.ollama.ai/library" / name / (tag if separator else "latest")
            if not executable.is_file() or not manifest.is_file():
                raise RuntimeError(f"Existing Ollama/{model} assets not found; configure TOOLFLOW_OLLAMA_EXE and TOOLFLOW_OLLAMA_MODELS; no download attempted")
            environment = os.environ.copy()
            environment.update({
                "OLLAMA_MODELS": str(models), "OLLAMA_HOST": "127.0.0.1:11434",
                "OLLAMA_NUM_PARALLEL": "1",
                "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0",
                "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NO_CLOUD": "1",
            })
            folder = self.project / "runs/p1"
            folder.mkdir(parents=True, exist_ok=True)
            self.log = (folder / "ollama_server.log").open("ab")
            self.process = subprocess.Popen(
                [str(executable), "serve"], env=environment,
                stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Ollama exited; inspect runs/p1/ollama_server.log")
                try:
                    self.request("/api/tags")
                    break
                except (OSError, ValueError):
                    time.sleep(0.5)
            else:
                raise RuntimeError("Ollama startup exceeded 45 seconds")
        models = self.request("/api/tags").get("models", [])
        if not any(item["name"] == model and not item.get("remote_host") for item in models):
            raise RuntimeError(f"The required local {model} model is unavailable")

    def close(self) -> None:
        """释放由本类启动的模型/服务；预先存在的服务保持不动。"""
        if self.process:
            try:
                children = psutil.Process(self.process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            try:
                self.request("/api/generate", {"model": self.model, "keep_alive": 0})
            except (OSError, ValueError):
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            # Windows 上终止父进程并不会终止 llama-server。
            # 只回收从本进程活动服务中捕获到的子进程。
            _, remaining = psutil.wait_procs(children, timeout=3)
            for child in remaining:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, remaining = psutil.wait_procs(remaining, timeout=3)
            for child in remaining:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(remaining, timeout=3)
            cleanup = {
                "owned_server_pid": self.process.pid,
                "owned_child_pids": [child.pid for child in children],
                "remaining_owned_child_pids": [
                    child.pid for child in children if child.is_running()
                ],
            }
            (self.project / "runs/p1/last_server_cleanup.json").write_text(
                json.dumps(cleanup, indent=2), encoding="utf-8",
            )
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
