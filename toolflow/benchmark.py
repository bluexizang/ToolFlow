"""官方 Benchmark 适配：只提出动作，由官方编排器执行工具。"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from tau2.agent.base import LocalAgent, is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage, Message, MultiToolMessage, ToolCall, ToolMessage, UserMessage,
)
from tau2.user.user_simulator import UserSimulator

from .model import ModelConfig
from .budgets import DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOTAL_TOKENS, DEFAULT_MAX_REPAIRS
from .runtime import RetailRuntime, provider_message, save_json


class BenchmarkAgentAdapter(LocalAgent[RetailRuntime]):
    """共享交互 Runtime，但把数据库执行权交给官方 Orchestrator。

    Parameters
    ----------
    project_root : Path
        项目目录。
    config : ModelConfig
        真实模型配置，不包含密钥内容。
    environment : Any
        当前试验独享的官方环境，与编排器持有同一个实例。
    max_model_calls : int
        整次试验的模型调用上限。
    max_total_tokens : int
        根据服务端 usage 累计的 token 预算。
    """

    def __init__(
        self, project_root: Path, config: ModelConfig, environment: Any,
        max_model_calls: int = DEFAULT_MAX_MODEL_CALLS, max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        *, enhanced_validation: bool = True, recovery: bool = True,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
    ) -> None:
        super().__init__(environment.get_tools(), environment.get_policy())
        self.project_root = project_root
        self.config = config
        self.environment = environment
        self.max_model_calls = max_model_calls
        self.max_total_tokens = max_total_tokens
        self.enhanced_validation = enhanced_validation
        self.recovery = recovery
        self.max_repairs = max_repairs
        self.runtime: RetailRuntime | None = None
        self.pending_call: ToolCall | None = None

    def get_init_state(self, message_history: list[Message] | None = None) -> RetailRuntime:
        """从可见历史恢复证据；旧确认无条件失效，绝不重放数据库写入。"""
        history = message_history or []
        if not all(is_valid_agent_history_message(m) for m in history):
            raise ValueError("invalid agent history")
        runtime = RetailRuntime(
            self.project_root, self.config, self.max_model_calls,
            self.max_total_tokens, environment=self.environment,
            output_group="benchmark_runtime",
            enhanced_validation=self.enhanced_validation, recovery=self.recovery,
            max_repairs=self.max_repairs,
        )
        self.runtime = runtime
        self.pending_call = None
        calls: dict[str, ToolCall] = {}
        for message in history:
            runtime.state.messages.append(message)
            runtime.provider_history.append(provider_message(message))
            if isinstance(message, UserMessage):
                runtime.guard.receive(message.content or "")
            elif isinstance(message, AssistantMessage):
                for call in message.tool_calls or []:
                    calls[call.id] = call
                    runtime.seen_call_ids.add(call.id)
            elif isinstance(message, ToolMessage):
                call = calls.pop(message.id, None)
                if call is None:
                    raise ValueError("orphan tool observation in history")
                runtime.guard.observe(call, message)
                if runtime.guard.is_write(call):
                    runtime.guard.attempted_writes.add(self._fingerprint(call))
        if len(calls) > 1:
            raise ValueError("only one outstanding call is supported")
        self.pending_call = next(iter(calls.values()), None)
        if self.pending_call is not None and runtime.guard.is_write(self.pending_call):
            runtime.close("unsafe_history_error")
            raise ValueError("cannot resume an unexecuted historical write without a fresh confirmation")
        runtime.guard.pending = None
        runtime.guard.confirmed = None
        runtime.state.status = "waiting_tool" if self.pending_call else "waiting_user"
        runtime.event("history_restored", {"messages": len(history), "confirmation_invalidated": True})
        return runtime

    @staticmethod
    def _fingerprint(call: ToolCall) -> str:
        """复用交互模式的参数摘要算法。"""
        from .validation import fingerprint
        return fingerprint(call)

    def set_seed(self, seed: int) -> None:
        """在初始化之前固定采样种子。"""
        self.config = replace(self.config, seed=seed)

    def generate_next_message(
        self, message: UserMessage | ToolMessage | MultiToolMessage,
        state: RetailRuntime,
    ) -> tuple[AssistantMessage, RetailRuntime]:
        """接收官方观测，返回决策；这里不调用 environment.get_response。"""
        if state is not self.runtime:
            raise ValueError("state belongs to another agent session")
        if isinstance(message, MultiToolMessage):
            if len(message.tool_messages) != 1:
                raise ValueError("Retail runtime supports one pending tool")
            message = message.tool_messages[0]
        flow = state._get_flow()
        if isinstance(message, UserMessage):
            if self.pending_call:
                raise ValueError("tool result required before another user turn")
            candidate = flow.run_user(message.content or "", external=True)
        else:
            candidate = flow.run_tool(message, pending=self.pending_call)
        if candidate is None:
            raise RuntimeError("agent graph stopped without a response")
        self.pending_call = candidate.tool_calls[0] if candidate.tool_calls else None
        return candidate, state

    def stop(self, message: Any = None, state: RetailRuntime | None = None) -> None:
        """只保存最终状态，不把停止当作任务成功。"""
        if state is not None:
            state.close("official_orchestrator_stop")


def litellm_settings(config: ModelConfig) -> tuple[str, dict[str, Any]]:
    """把配置转给官方传输层；返回值含凭据，不允许写入文件或输出。"""
    import os
    key = os.environ.get(config.api_key_env, "").strip() if config.backend == "cloud" else "local-not-a-secret"
    if not key:
        raise ValueError(f"missing credential environment variable: {config.api_key_env}")
    args: dict[str, Any] = {
        "api_base": config.base_url, "api_key": key, "max_tokens": config.max_tokens,
        "timeout": config.timeout_seconds, "num_retries": 0,
    }
    for name in ("temperature", "seed", "reasoning_effort"):
        value = getattr(config, name)
        if value is not None:
            args[name] = value
    return "openai/" + config.model, args


class BudgetedOfficialAgent(LLMAgent):
    """官方原版决策与提示词，加调用预算和离线审计保存。"""

    def __init__(self, tools: list[Any], domain_policy: str, config: ModelConfig,
                 output: Path, max_calls: int, max_total_tokens: int) -> None:
        model, args = litellm_settings(config)
        super().__init__(tools, domain_policy, model, args)
        self.backend = config.backend
        self.audit = ModelAudit(output, max_calls, max_total_tokens, config.max_tokens)

    def generate_next_message(self, message: Any, state: Any) -> tuple[AssistantMessage, Any]:
        """调用官方实现，保留原始 Baseline 的工具可见性与行为。"""
        incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        self.audit.before({
            "messages": [provider_message(m) for m in state.system_messages + state.messages + incoming],
            "tools": [tool.openai_schema for tool in self.tools],
        })
        result, state = super().generate_next_message(message, state)
        result.cost = 0.0 if self.backend == "local" else None
        self.audit.after(state.system_messages + state.messages, result)
        return result, state


class BudgetedOfficialUser(UserSimulator):
    """官方用户模拟器，单独配置模型与预算。"""

    def __init__(self, instructions: str, config: ModelConfig, output: Path,
                 max_calls: int, max_total_tokens: int) -> None:
        model, args = litellm_settings(config)
        super().__init__(tools=None, instructions=instructions, llm=model, llm_args=args)
        self.backend = config.backend
        self.audit = ModelAudit(output, max_calls, max_total_tokens, config.max_tokens)

    def generate_next_message(self, message: Any, state: Any) -> tuple[UserMessage, Any]:
        """仅把用户场景传给用户模拟器，不传给客服 Agent。"""
        visible = state.model_copy(deep=True)
        incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        visible.messages.extend(incoming)
        self.audit.before({"messages": [provider_message(m) for m in visible.system_messages + visible.flip_roles()]})
        result, state = super().generate_next_message(message, state)
        result.cost = 0.0 if self.backend == "local" else None
        self.audit.after(state.system_messages + state.messages, result)
        return result, state


class ModelAudit:
    """记录官方模型可见历史与响应；usage 预算是检查阈值而非费用估算。"""

    def __init__(self, output: Path, max_calls: int, max_tokens: int, output_tokens: int) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.calls = 0
        self.tokens = 0

    def before(self, request: dict[str, Any] | None = None) -> None:
        """耗尽预算后停止，不自动重试。"""
        if self.calls >= self.max_calls or self.tokens + self.output_tokens > self.max_tokens:
            raise RuntimeError("official model budget exhausted")
        self.calls += 1
        if request is not None:
            save_json(self.output / f"{self.calls:03}.request.json", request)

    def after(self, history: list[Any], response: Any) -> None:
        """记录已完成调用；未知费用不伪装成零费用。"""
        usage = response.usage or {}
        self.tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        save_json(self.output / f"{self.calls:03}.json", {
            "history": [m.model_dump(mode="json") for m in history],
            "response": response.model_dump(mode="json"), "total_tokens": self.tokens,
        })
        if self.tokens > self.max_tokens:
            raise RuntimeError("official provider usage exceeded token budget")
