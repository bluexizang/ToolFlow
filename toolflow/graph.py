"""ToolFlow 的唯一编排图：节点负责做事，条件边决定下一步。

阅读顺序：build_graph → model → validate → recover / confirm / authorize。
本机工具在 execute 执行；Benchmark 在 wait_tool 暂停，交给官方执行。
"""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any, Callable, TYPE_CHECKING
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

from .budgets import DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_REPAIRS
from .graph_state import FlowState
from .model import ModelProtocolError
from .recovery import IDENTITY_QUESTION, classify_guard_error
from .validation import GuardError, fingerprint

if TYPE_CHECKING:
    from .runtime import RetailRuntime


class ToolFlowGraph:
    """把业务服务连接为图，不在节点内部再嵌套 Agent while 循环。

    Parameters
    ----------
    runtime : RetailRuntime
        当前会话的模型、守卫、环境和日志服务。
    """

    def __init__(self, runtime: RetailRuntime) -> None:
        self.runtime = runtime
        self.builder = self.build_graph()
        self.snapshot: dict[str, Any] = {}
        self.next_nodes: tuple[str, ...] = ()
        self.mode: str | None = None
        topology = self.builder.compile().get_graph().draw_mermaid()
        (runtime.output / "graph.mmd").write_text(topology, encoding="utf-8")

    def build_graph(self) -> StateGraph:
        """声明真实执行的节点和分支，图中的循环就是 Agent Loop。"""
        graph = StateGraph(FlowState)
        for name in (
            "receive",
            "model",
            "validate",
            "recover",
            "confirm",
            "reply",
            "authorize",
            "execute",
            "wait_tool",
            "observe",
            "prepare_readback",
            "verify_readback",
            "wait_user",
        ):
            graph.add_node(name, getattr(self, name))
        graph.add_conditional_edges(
            START,
            lambda s: s["entry"],
            {name: name for name in ("receive", "model", "authorize", "observe")},
        )
        graph.add_edge("receive", "model")
        graph.add_conditional_edges(
            "model",
            lambda s: "recover" if s["error"] else "validate",
            {"recover": "recover", "validate": "validate"},
        )
        graph.add_conditional_edges(
            "validate",
            lambda s: s["route"],
            {
                "recover": "recover",
                "confirm": "confirm",
                "reply": "reply",
                "authorize": "authorize",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "recover",
            lambda s: s["route"],
            {
                "model": "model",
                "wait_user": "wait_user",
                "end": END,
            },
        )
        for node in ("confirm", "reply", "verify_readback"):
            graph.add_conditional_edges(
                node,
                self.after_reply,
                {
                    "wait_user": "wait_user",
                    "end": END,
                },
            )
        graph.add_conditional_edges(
            "authorize",
            lambda s: s["route"],
            {
                "execute": "execute",
                "wait_tool": "wait_tool",
                "recover": "recover",
                "confirm": "confirm",
            },
        )
        graph.add_edge("execute", "observe")
        graph.add_edge("wait_tool", "observe")
        graph.add_conditional_edges(
            "observe",
            lambda s: s["route"],
            {
                "model": "model",
                "prepare_readback": "prepare_readback",
                "verify_readback": "verify_readback",
                "wait_user": "wait_user",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "prepare_readback",
            lambda s: s["route"],
            {
                "execute": "execute",
                "wait_tool": "wait_tool",
            },
        )
        graph.add_edge("wait_user", "model")
        return graph

    def _run(
        self,
        value: FlowState | Command,
        mode: str,
        on_node: Callable[[str, dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """驱动框架并保存节点轨迹；此处遍历事件，不自行决定下一个动作。

        SQLite 连接只在本次调用内打开，Web 线程结束后不会遗留文件句柄。
        图检查点可用于检查暂停位置；尚不支持进程重启后重建整套业务会话。
        """
        from .runtime import save_json

        config = {
            "configurable": {"thread_id": self.runtime.state.run_id + ":" + mode},
            "recursion_limit": max(
                100,
                getattr(self.runtime, "max_model_calls", DEFAULT_MAX_MODEL_CALLS) * 12
                + 100,
            ),
        }
        self.mode = mode
        with SqliteSaver.from_conn_string(
            str(self.runtime.output / "graph.sqlite3")
        ) as saver:
            graph = self.builder.compile(checkpointer=saver)
            stream = graph.stream(value, config=config, stream_mode="updates")
            try:
                for update in stream:
                    for node, data in update.items():
                        if node == "__interrupt__":
                            continue
                        self.runtime.event("graph_node_completed", {"node": node})
                        if on_node is not None:
                            on_node(node, data or {})
                    if should_stop is not None and should_stop():
                        self.runtime.close("web_server_shutdown")
                        break
            finally:
                stream.close()
                state = graph.get_state(config)
                self.snapshot = dict(state.values)
                self.next_nodes = state.next
                save_json(
                    self.runtime.output / "graph_state.json",
                    {
                        "thread_id": config["configurable"]["thread_id"],
                        "next_nodes": list(state.next),
                        "values": self.snapshot,
                        "restart_resume_supported": False,
                    },
                )
        return self.snapshot

    def run_user(
        self,
        content: str,
        *,
        external: bool = False,
        on_node: Callable[[str, dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> AssistantMessage | None:
        """接收一轮真实用户输入，直到回复用户或等待官方工具结果。"""
        mode = "external" if external else "local"
        if self.runtime.state.status != "waiting_user":
            raise ValueError("expected a user message while waiting")
        from .console import clean_input

        content = clean_input(content)
        if self.mode == mode and self.next_nodes == ("wait_user",):
            value = Command(resume={"kind": "user", "content": content})
        else:
            value = FlowState(
                entry="receive",
                mode=mode,
                user_input=content,
                candidate=None,
                error=None,
                is_readback=False,
            )
        self._run(value, mode, on_node, should_stop)
        if self.runtime.state.status in {"stopped", "failed"}:
            return None
        return self.last_assistant()

    def run_tool(
        self, response: ToolMessage, pending: ToolCall | None = None
    ) -> AssistantMessage:
        """只接收官方 Orchestrator 已执行的结果，不在这里调用数据库。"""
        data = self.snapshot.get("call")
        call = ToolCall.model_validate(data) if data else pending
        if call is None or response.id != call.id or response.requestor != "assistant":
            raise ValueError("tool result correlation mismatch")
        if self.runtime.state.status != "waiting_tool":
            raise ValueError("tool result correlation mismatch: no outstanding tool")
        if self.mode == "external" and self.next_nodes == ("wait_tool",):
            value = Command(
                resume={"kind": "tool", "response": response.model_dump(mode="json")}
            )
        else:
            # 仅恢复官方传入的可见历史中的待返回只读调用，不重放执行。
            value = FlowState(
                entry="observe",
                mode="external",
                call=call.model_dump(mode="json"),
                response=response.model_dump(mode="json"),
                is_readback=False,
                before_write=None,
            )
        self._run(value, "external")
        return self.last_assistant()

    def last_assistant(self) -> AssistantMessage:
        """返回本次暂停时给前端或编排器的消息。"""
        return next(
            m
            for m in reversed(self.runtime.state.messages)
            if isinstance(m, AssistantMessage)
        )

    def receive(self, state: FlowState) -> dict[str, Any]:
        """把用户消息写入会话和下一次模型请求历史。"""
        self.runtime.receive(state["user_input"])
        return {"error": None, "candidate": None, "is_readback": False}

    def model(self, state: FlowState) -> dict[str, Any]:
        """进行一次原生 Tool Calling 请求，格式错误交给恢复节点。"""
        try:
            candidate = self.runtime._request()
            return {"candidate": candidate.model_dump(mode="json"), "error": None}
        except ModelProtocolError as error:
            return {
                "candidate": None,
                "error": {"code": "MODEL_PROTOCOL", "detail": str(error)},
            }

    def validate(self, state: FlowState) -> dict[str, Any]:
        """判断模型提议是否合法；确认预览从不写入官方执行轨迹。"""
        from .runtime import PREPARATIONS, provider_message

        runtime = self.runtime
        candidate = AssistantMessage.model_validate(state["candidate"])
        try:
            if candidate.tool_calls and runtime.guard.needs_identity_input():
                raise GuardError(
                    "USER_INPUT_REQUIRED",
                    "A failed identity lookup requires corrected user information before any further tool call.",
                )
            if not candidate.tool_calls:
                runtime.guard.check_text(candidate.content)
                return {"route": "reply"}
            call = candidate.tool_calls[0]
            preparations = getattr(runtime, "preparations", PREPARATIONS)
            if call.name not in runtime.tool_names and call.name not in preparations:
                raise GuardError(
                    "UNKNOWN_OR_DISABLED_TOOL",
                    "Use only provided tools. For variants use get_product_details.",
                )
            if call.id in runtime.seen_call_ids:
                raise ModelProtocolError("duplicate provider call ID")
            runtime.seen_call_ids.add(call.id)
            runtime.provider_history.append(provider_message(candidate))
            prepared = call.name in preparations
            checked_call = (
                call.model_copy(update={"name": preparations[call.name]})
                if prepared
                else call
            )
            if hasattr(runtime.environment, "refresh"):
                runtime.environment.refresh()
            proposal = runtime.guard.check(checked_call)
            if prepared and proposal is None:
                raise GuardError(
                    "WRITE_ALREADY_CONFIRMED",
                    f"The preview is already confirmed. Call {checked_call.name} with the exact approved parameters now.",
                )
            if proposal is not None:
                runtime.provider_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {"code": "CONFIRMATION_REQUIRED", "executed": False}
                        ),
                    }
                )
                runtime.event(
                    "confirmation_required",
                    {"proposal": asdict(proposal), "provider_call_id": call.id},
                )
                return {"route": "confirm", "review": proposal.review}
            if state["mode"] == "decision":
                runtime.state.messages.append(candidate)
                runtime.state.status = "waiting_tool"
                runtime.event(
                    "assistant_message",
                    {
                        "source": "model",
                        "message": state["candidate"],
                    },
                )
            return {
                "route": "end" if state["mode"] == "decision" else "authorize",
                "call": call.model_dump(mode="json"),
                "is_readback": False,
            }
        except (GuardError, ModelProtocolError) as error:
            code = error.code if isinstance(error, GuardError) else "MODEL_PROTOCOL"
            return {"route": "recover", "error": {"code": code, "detail": str(error)}}

    def recover(self, state: FlowState) -> dict[str, Any]:
        """分类错误：能修正就回模型，缺信息就补问，达到预算才暂停。"""
        runtime = self.runtime
        code, detail = state["error"]["code"], state["error"]["detail"]
        runtime.state.blocked_proposals += 1
        decision = classify_guard_error(code, detail, runtime.guard)
        if decision.action == "repair":
            runtime.state.turn_repairs += 1
        runtime.state.recovery_action = decision.action
        feedback = {
            "code": code,
            "executed": False,
            "detail": detail,
            "next_action": decision.action,
            "recovery": decision.instruction,
        }
        candidate = state.get("candidate") or {}
        calls = candidate.get("tool_calls") or []
        call_id = calls[0]["id"] if calls else None
        history = runtime.provider_history
        if (
            calls
            and history
            and history[-1].get("tool_calls")
            and history[-1]["tool_calls"][0]["id"] == call_id
        ):
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(feedback, ensure_ascii=False),
                }
            )
        else:
            history.append(
                {
                    "role": "system",
                    "content": (
                        "Runtime rejected your last output before execution. "
                        + json.dumps(feedback, ensure_ascii=False)
                        + " Correct the format/parameters, or ask the user for missing information."
                    ),
                }
            )
        runtime.event("proposal_blocked", {**feedback, "provider_call_id": call_id})
        runtime.event(
            "recovery_decision", {"code": code, "next_action": decision.action}
        )
        limit = getattr(runtime, "repair_limit", DEFAULT_MAX_REPAIRS)
        if decision.action == "ask_user" or runtime.state.turn_repairs >= limit:
            runtime.guard.pending = runtime.guard.confirmed = None
            if decision.action == "ask_user":
                runtime.state.turn_repairs = 0
                message = runtime._say(
                    decision.question or IDENTITY_QUESTION,
                    "runtime_information_request",
                )
            else:
                message = runtime._say(
                    f"当前步骤已达到 {limit} 次连续修正上限，已暂停；未执行被拦截的工具提议。"
                    f"原因：{code}，{detail}。请根据原因补充信息或调整需求后继续。",
                    "runtime_recovery_pause",
                )
            return {
                "candidate": message.model_dump(mode="json"),
                "route": self.after_reply(state),
            }
        return {"route": "model", "error": None}

    def confirm(self, state: FlowState) -> dict[str, Any]:
        """输出由数据库证据生成的精确确认单，然后等待用户。"""
        message = self.runtime._say(state["review"], "runtime_confirmation")
        return {"candidate": message.model_dump(mode="json")}

    def reply(self, state: FlowState) -> dict[str, Any]:
        """把校验后的模型文本交给用户，过期的确认同时失效。"""
        from .runtime import provider_message

        runtime = self.runtime
        message = AssistantMessage.model_validate(state["candidate"])
        runtime.state.messages.append(message)
        runtime.provider_history.append(provider_message(message))
        runtime.guard.pending = runtime.guard.confirmed = None
        runtime.state.status = "waiting_user"
        runtime.event(
            "assistant_message", {"source": "model", "message": state["candidate"]}
        )
        return {}

    @staticmethod
    def after_reply(state: FlowState) -> str:
        """逐步调试在本步结束，交互和 Benchmark 保留可恢复的暂停点。"""
        return "end" if state["mode"] in {"decision", "execution"} else "wait_user"

    def authorize(self, state: FlowState) -> dict[str, Any]:
        """执行边界再次校验并消耗一次性确认；写操作不会自动重试。"""
        runtime = self.runtime
        call = ToolCall.model_validate(state["call"])
        try:
            if hasattr(runtime.environment, "refresh"):
                runtime.environment.refresh()
            proposal = runtime.guard.check(call)
            if proposal is not None:
                if state["mode"] == "execution":
                    raise RuntimeError("write requires fresh user confirmation")
                runtime.provider_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {"code": "CONFIRMATION_REQUIRED", "executed": False}
                        ),
                    }
                )
                runtime.event(
                    "confirmation_required",
                    {
                        "proposal": asdict(proposal),
                        "provider_call_id": call.id,
                        "source": "execution_recheck",
                    },
                )
                return {"route": "confirm", "review": proposal.review}
        except GuardError as error:
            # 调试接口没有再次请求模型的权限；完整图则交给统一恢复分支。
            if state["mode"] == "execution":
                raise RuntimeError(str(error)) from error
            from .runtime import provider_message

            runtime.provider_history.append(
                provider_message(
                    ToolMessage(
                        role="tool",
                        id=call.id,
                        name=call.name,
                        requestor="assistant",
                        error=True,
                        content=json.dumps(
                            {
                                "code": error.code,
                                "executed": False,
                                "detail": str(error),
                            }
                        ),
                    )
                )
            )
            runtime.state.status = "running"
            return {
                "route": "recover",
                "error": {"code": error.code, "detail": str(error)},
            }
        if state["mode"] != "execution":
            candidate = AssistantMessage.model_validate(state["candidate"])
            runtime.state.messages.append(candidate)
            runtime.state.status = "waiting_tool"
            runtime.event(
                "assistant_message",
                {
                    "source": "model",
                    "message": state["candidate"],
                },
            )
        write = runtime.guard.is_write(call)
        before = runtime.environment.get_db_hash() if write else None
        if write:
            if hasattr(runtime.environment, "expected_write_hash"):
                runtime.environment.expected_write_hash = before
            runtime.guard.consume_confirmation(call)
            event = (
                "external_write_authorized"
                if state["mode"] == "external"
                else "write_attempt"
            )
            runtime.event(
                event,
                {
                    "call_id": call.id,
                    "fingerprint": fingerprint(call),
                    "db_hash": before,
                },
            )
        return {
            "before_write": before,
            "route": "wait_tool" if state["mode"] == "external" else "execute",
        }

    def execute(self, state: FlowState) -> dict[str, Any]:
        """只有本机模式会调用环境，外部模式不会经过此节点。"""
        call = ToolCall.model_validate(state["call"])
        started = time.perf_counter()
        self.runtime.event("tool_started", call.model_dump(mode="json"))
        response = self.runtime.environment.get_response(call)
        if response.id != call.id:
            raise RuntimeError("official tool correlation changed")
        return {"response": response.model_dump(mode="json"), "started_at": started}

    def wait_tool(self, state: FlowState) -> dict[str, Any]:
        """暂停交出 ToolCall，恢复时只接收结果；此前节点不会重复执行。"""
        value = interrupt({"kind": "tool", "call": state["call"]})
        if value.get("kind") != "tool":
            raise ValueError("tool result required")
        response = ToolMessage.model_validate(value["response"])
        if response.id != state["call"]["id"] or response.requestor != "assistant":
            raise ValueError("tool result correlation mismatch")
        return {"response": response.model_dump(mode="json")}

    def observe(self, state: FlowState) -> dict[str, Any]:
        """唯一的工具结果入账处：日志、身份/订单证据、修正计数和错误分类。"""
        from .runtime import provider_message

        runtime = self.runtime
        call = ToolCall.model_validate(state["call"])
        response = ToolMessage.model_validate(state["response"])
        runtime.state.tool_calls += 1
        runtime.state.messages.append(response)
        runtime.provider_history.append(provider_message(response))
        runtime.guard.observe(call, response)
        if not response.error:
            runtime.state.turn_repairs = 0
        runtime.state.status = "running"
        external = state["mode"] == "external"
        runtime.event(
            "external_tool_finished" if external else "tool_finished",
            {
                "name": call.name,
                "call": state["call"],
                "response": state["response"],
                "seconds": (
                    None if external else time.perf_counter() - state["started_at"]
                ),
            },
        )
        if state.get("is_readback"):
            return {"route": "verify_readback"}
        if runtime.guard.is_write(call):
            if (
                response.error
                and runtime.environment.get_db_hash() != state["before_write"]
            ):
                runtime.state.status = "failed"
                runtime.state.stop_reason = "write_outcome_error"
                runtime.state.recovery_action = "stop"
                runtime.event(
                    "recovery_decision",
                    {"code": "WRITE_OUTCOME_UNCERTAIN", "next_action": "stop"},
                )
                runtime.event("write_outcome_uncertain", {"call_id": call.id})
                raise RuntimeError(
                    "write failed after state changed; no automatic retry"
                )
            if not response.error:
                return {
                    "route": "prepare_readback",
                    "write_call": state["call"],
                    "write_result": response.content,
                    "execution_result": state["response"],
                }
        if response.error:
            runtime.record_tool_error(call, response)
        if runtime.state.status == "waiting_user":
            return {
                "route": self.after_reply(state),
                "candidate": self.last_assistant().model_dump(mode="json"),
                "execution_result": state["response"],
            }
        return {
            "route": "end" if state["mode"] == "execution" else "model",
            "execution_result": state["response"],
        }

    def prepare_readback(self, state: FlowState) -> dict[str, Any]:
        """用确定的代码构造写后查询；不要求模型记住这一步。"""
        from .runtime import provider_message

        write = ToolCall.model_validate(state["write_call"])
        result = json.loads(state["write_result"])
        profile = write.name == "register_account" or (
            "order_id" not in write.arguments and write.name != "place_order"
        )
        key = "user_id" if profile else "order_id"
        call = ToolCall(
            id="tf_readback_" + uuid.uuid4().hex,
            name="get_user_details" if profile else "get_order_details",
            arguments={key: write.arguments.get(key, result[key])},
        )
        message = AssistantMessage(role="assistant", tool_calls=[call], cost=0.0)
        runtime = self.runtime
        runtime.state.messages.append(message)
        runtime.provider_history.append(provider_message(message))
        runtime.state.status = "waiting_tool"
        runtime.event(
            (
                "external_readback_started"
                if state["mode"] == "external"
                else "readback_started"
            ),
            {"source": "runtime", "call": call.model_dump(mode="json")},
        )
        return {
            "candidate": message.model_dump(mode="json"),
            "call": call.model_dump(mode="json"),
            "is_readback": True,
            "route": "wait_tool" if state["mode"] == "external" else "execute",
        }

    def verify_readback(self, state: FlowState) -> dict[str, Any]:
        """回查一致才发回执；不一致停止，避免重复写入。"""
        runtime = self.runtime
        checked = ToolMessage.model_validate(state["response"])
        try:
            match = not checked.error and json.loads(checked.content) == json.loads(
                state["write_result"]
            )
        except (ValueError, TypeError):
            match = False
        runtime.event(
            "readback_finished",
            {
                "source": "runtime",
                "matches_write_result": match,
                "response": state["response"],
            },
        )
        if not match:
            runtime.state.status = "failed"
            runtime.state.stop_reason = "readback_error"
            runtime.state.recovery_action = "stop"
            runtime.event(
                "recovery_decision",
                {"code": "READBACK_MISMATCH", "next_action": "stop"},
            )
            runtime.event(
                "integrity_failure",
                {"call_id": state["write_call"]["id"], "kind": "readback_mismatch"},
            )
            raise RuntimeError(
                "write readback mismatch; inspect state before any retry"
            )
        message = runtime._verified_receipt(
            ToolCall.model_validate(state["write_call"]), json.loads(checked.content)
        )
        return {"candidate": message.model_dump(mode="json"), "is_readback": False}

    def wait_user(self, state: FlowState) -> dict[str, Any]:
        """暂停点自身没有副作用：恢复之后才追加新的用户消息。"""
        value = interrupt({"kind": "user", "message": state.get("candidate")})
        if value.get("kind") != "user":
            raise ValueError("user input required")
        self.runtime.receive(value["content"])
        return {"candidate": None, "error": None, "is_readback": False}
