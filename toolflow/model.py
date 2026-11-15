"""本地模型与云端 Chat Completions 适配器，共享同一协议。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import time
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from tau2.data_model.message import AssistantMessage, ToolCall

from .local_config import DEFAULT_LOCAL_MODEL
from .budgets import DEFAULT_MAX_OUTPUT_TOKENS


class ModelProtocolError(RuntimeError):
    """提供方返回了不可用或非原生的工具调用响应。"""


@dataclass(frozen=True)
class ModelConfig:
    """可序列化的模型配置，仅包含密钥的环境变量名。"""

    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = DEFAULT_LOCAL_MODEL
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout_seconds: float = 180.0
    reasoning_effort: str | None = None
    backend: Literal["local", "cloud"] = "local"
    api_key_env: str = "TOOLFLOW_API_KEY"

    def __post_init__(self) -> None:
        """校验端点与默认值，不读取或存储任何密钥。"""
        if self.backend not in {"local", "cloud"}:
            raise ValueError("backend must be local or cloud")
        parsed = urlparse(self.base_url)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("base URL must have a host and no credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base URL must not contain a query or fragment")
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if self.backend == "local":
            if parsed.scheme != "http" or not local or ":cloud" in self.model:
                raise ValueError("local backend requires local HTTP and a local model")
            for name, value in (
                ("temperature", 0.0), ("seed", 42), ("reasoning_effort", "none"),
            ):
                if getattr(self, name) is None:
                    object.__setattr__(self, name, value)
        elif parsed.scheme != "https" or local:
            raise ValueError("cloud backend requires a remote HTTPS base URL")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        if not self.model.strip() or self.max_tokens < 1:
            raise ValueError("model and positive max_tokens are required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if self.temperature is not None and not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")


def parse_response(
    response: dict[str, Any], *, api_fee: float | None = None,
) -> AssistantMessage:
    """校验并转换一条真实的提供方响应。

    Parameters
    ----------
    response : dict[str, Any]
        来自 chat-completions 端点的原始 JSON 响应。
    api_fee : float | None
        已知的请求费用；None 表示尚未估算。

    Returns
    -------
    AssistantMessage
        官方 tau2 消息，保留提供方的调用 ID。

    Raises
    ------
    ModelProtocolError
        当结果被截断、混杂、为空或参数非法时抛出。
    """
    try:
        choice = response["choices"][0]
        raw = choice["message"]
        content = raw.get("content") or None
        raw_calls = raw.get("tool_calls") or []
        if choice.get("finish_reason") == "length":
            raise ValueError("provider output reached max_tokens")
        if raw.get("role") != "assistant":
            raise ValueError("expected assistant role")
        if len(raw_calls) > 1:
            raise ValueError("Retail permits one tool call per model turn")
        if raw_calls and content and content.strip():
            raise ValueError("Retail disallows simultaneous text and tool call")
        calls = []
        for call in raw_calls:
            if call.get("type") != "function" or not call.get("id"):
                raise ValueError("native function call with provider ID required")
            function = call["function"]
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            calls.append(ToolCall(
                id=call["id"], name=function["name"], arguments=arguments,
            ))
        if not calls and (not content or not content.strip()):
            raise ValueError("empty assistant response")
        return AssistantMessage(
            role="assistant", content=content, tool_calls=calls or None,
            usage=response.get("usage"), cost=api_fee, raw_data=response,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ModelProtocolError(str(error)) from error


class NoRedirect(HTTPRedirectHandler):
    """防止授权头被转发到重定向后的主机。"""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str,
        headers: Any, newurl: str,
    ) -> None:
        """拒绝重定向，而不是继续转发模型请求。"""
        return None


class ChatCompletionsModel:
    """共享的 JSON 传输层；密钥绝不会写入请求体或配置。"""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._api_key: str | None = None
        proxy = ProxyHandler({}) if config.backend == "local" else ProxyHandler()
        self.opener = build_opener(proxy, NoRedirect())

    def build_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """构造请求，便于运行时在发送前保存。"""
        body: dict[str, Any] = {
            "model": self.config.model, "messages": messages, "tools": tools,
            "tool_choice": "auto", "stream": False,
            "max_tokens": self.config.max_tokens,
        }
        for name in ("temperature", "seed", "reasoning_effort"):
            value = getattr(self.config, name)
            if value is not None:
                body[name] = value
        return body

    def complete(
        self, body: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        """仅发起一次模型请求，不做自动重试。

        Parameters
        ----------
        body : dict[str, Any]
            已保存的请求，包含可见的观测与 schema。

        Returns
        -------
        tuple[dict[str, Any], float]
            未经修改的提供方 JSON 与耗时秒数。
        """
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )
        start = time.perf_counter()
        try:
            with self.opener.open(
                request, timeout=self.config.timeout_seconds,
            ) as result:
                response = json.load(result)
        except HTTPError as error:
            detail = self.redact(error.read().decode("utf-8", errors="replace"))
            raise RuntimeError(
                f"{self.config.backend} model HTTP {error.code}: {detail[:2000]}"
            ) from None
        except URLError as error:
            raise RuntimeError(
                f"{self.config.backend} connection failed: {self.redact(str(error.reason))}"
            ) from None
        return response, time.perf_counter() - start

    def redact(self, text: str) -> str:
        """当 HTTP 错误回显已配置的凭据时将其移除。"""
        return text.replace(self._api_key, "[REDACTED]") if self._api_key else text


class LocalChatModel(ChatCompletionsModel):
    """使用现有的本地模型服务，不读取云端密钥。"""

    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "local":
            raise ValueError("LocalChatModel requires backend=local")
        super().__init__(config)


class CloudChatModel(ChatCompletionsModel):
    """使用 HTTPS Chat Completions 端点，密钥由环境变量保存。"""

    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "cloud":
            raise ValueError("CloudChatModel requires backend=cloud")
        key = os.environ.get(config.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Cloud key is not configured in {config.api_key_env}")
        if "\r" in key or "\n" in key:
            raise ValueError("Cloud key contains an invalid line break")
        super().__init__(config)
        self._api_key = key


def create_model(config: ModelConfig) -> ChatCompletionsModel:
    """选择本地或云端适配器，不发起模型请求。"""
    if config.backend == "local":
        return LocalChatModel(config)
    return CloudChatModel(config)
