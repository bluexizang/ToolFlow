"""显式单步运行时，以及每次运行的官方 Retail 环境。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.retail.environment import get_environment

from .model import ModelConfig, ModelProtocolError, create_model, parse_response
from .console import clean_input
from .budgets import (
    DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOTAL_TOKENS, DEFAULT_MAX_REPAIRS,
)
from .validation import GuardError, RetailGuard, fingerprint
from .recovery import (
    RUNTIME_REVISION, IDENTITY_QUESTION, classify_guard_error, classify_tool_error,
)


PREPARATIONS = {"prepare_exchange": "exchange_delivered_order_items"}


INSTRUCTIONS = """
You are a retail customer-service agent. Follow the complete policy below.
Use real tool results, never invent IDs, stock, prices, or completed actions.
Each model turn must contain EITHER user-facing text OR exactly ONE tool call.
Do not emit textual/XML tool calls; use the provided native tool definitions.
Authenticate identity before revealing orders. Verify the order belongs to
the authenticated user. Collect all requested changes before a one-time write.
The first business tool call MUST authenticate the customer by email or by
first name + last name + ZIP. If these inputs are missing, ask the user for
them; do not query any order or product yet. A supplied order ID is not identity.
Use the product tools to inspect stock yourself. Never ask the customer to
check stock or supply catalog IDs that you can retrieve from the order/tools.
For replacement options use get_product_details with the order's product_id:
it returns the complete variant catalog and stock, unlike a single-item lookup.
Calculate signed price difference as NEW total minus ORIGINAL order total.
A negative difference is money returned to the customer, never an extra charge.
Before any write, explain the complete proposed changes, price difference and
payment method, ask for explicit confirmation, and wait for the user's reply.
After a successful write, query the changed record again and verify the result
before reporting success. Tool execution only requests the simulated workflow;
do not claim delivery or completed real-world refunds. Only report new orders
actually created by supported tools.
Ask for missing information instead of guessing. Respond in the user's language.
Never treat tool documentation examples as real customer data. If no order ID
is known, use get_user_details after authentication to retrieve the order list.
get_item_details is disabled due to a pinned-upstream defect; use product lookup.
For a write, propose the exact native tool call. The runtime will display its
own grounded review and obtain confirmation; do not claim the write has run.
For exchange use prepare_exchange BEFORE asking for confirmation. This is a
read-only runtime preview, not a database update. After the runtime review is
confirmed, use exchange_delivered_order_items with exactly the approved values.
When runtime feedback rejects a proposal, correct it or ask for missing data.
Preserve all unrequested product options. Never call an 80% keyboard full size.
For quantity changes distinguish ADDITIONAL units from the desired FINAL total.
If existing orders and phrases like 'buy one more, two in total' leave ambiguity,
ask a short clarification before preparing a new purchase. The latest explicit
correction replaces earlier quantity interpretations. Never add existing units
to a quantity that the user explicitly described as the final total.
A new order does not replace, merge, or cancel earlier orders. Explain which
existing orders will remain and the resulting total. Use observed orders and
query missing order details; never invent a merge operation. Collect a valid
cancellation reason from the user; do not silently choose one for them.
""".strip()


def save_json(path: Path, value: Any) -> None:
    """立即序列化并写入一份 UTF-8 快照。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def provider_message(message: Any) -> dict[str, Any]:
    """转换官方消息，同时不泄露运行时或评估器状态。"""
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if isinstance(message, AssistantMessage) and message.tool_calls:
        result["tool_calls"] = [{
            "id": call.id, "type": "function", "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        } for call in message.tool_calls]
    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.id
    return result


def prepare_exchange_schema(tools: list[Any]) -> dict[str, Any]:
    """暴露一个只读的预准备工具，字段与官方换货工具一致。"""
    source = next(tool for tool in tools if tool.name == "exchange_delivered_order_items")
    schema = deepcopy(source.openai_schema)
    schema["function"]["name"] = "prepare_exchange"
    schema["function"]["description"] = (
        "Read-only ToolFlow exchange preview. Use AFTER reading the order, "
        "profile/payment methods and required product variants, BEFORE asking "
        "the user to confirm. Validates selected variants and computes the exact "
        "signed price difference. Runtime displays a grounded confirmation form. "
        "This tool NEVER updates the database. After user confirmation submit "
        "exchange_delivered_order_items with the approved arguments."
    )
    return schema


@dataclass
class SessionState:
    """会话状态；业务成功与否由独立验证判定。"""

    run_id: str
    status: str = "waiting_user"
    messages: list[Any] = field(default_factory=list)
    model_calls: int = 0
    model_seconds: float = 0.0
    total_tokens: int = 0
    tool_calls: int = 0
    blocked_proposals: int = 0
    turn_repairs: int = 0  # 连续失败次数，成功执行工具后清零。
    stop_reason: str | None = None
    recovery_action: str | None = None


class RetailRuntime:
    """将模型决策与官方工具执行相隔离。

    Parameters
    ----------
    project_root : Path
        包含锁定上游代码与本地运行输出的项目根目录。
    config : ModelConfig
        现有本地模型的显式配置。
    max_model_calls : int
        整次运行的模型调用预算，涵盖所有用户轮次。
    max_total_tokens : int
        所有调用上报的输入/输出 token 之和的上限。
    """

    def __init__(
        self, project_root: Path, config: ModelConfig,
        max_model_calls: int = DEFAULT_MAX_MODEL_CALLS, max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        retail_mode: str = "official", store_path: Path | None = None,
        *, environment: Any | None = None, output_group: str = "p1",
        enhanced_validation: bool = True, recovery: bool = True,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
    ) -> None:
        if max_model_calls < 1 or max_total_tokens < config.max_tokens or max_repairs < 1:
            raise ValueError("invalid run budget")
        self.project = project_root.resolve()
        self.model = create_model(config)
        if retail_mode not in {"official", "extended"}:
            raise ValueError("retail_mode must be official or extended")
        self.retail_mode = retail_mode
        self.preparations = dict(PREPARATIONS)
        if retail_mode == "extended":
            from .commerce import CommerceEnvironment, CommerceGuard
            self.environment = CommerceEnvironment(store_path)
            self.guard = CommerceGuard(self.environment)
            self.preparations.update({
                "prepare_registration": "register_account",
                "prepare_purchase": "place_order",
            })
        else:
            self.environment = environment if environment is not None else get_environment()
            self.guard = RetailGuard(self.environment)
        self.config = config
        self.guard.enhanced_validation = enhanced_validation
        self.repair_limit = max_repairs if recovery else 1
        self.max_model_calls = max_model_calls
        self.max_total_tokens = max_total_tokens
        self.state = SessionState(uuid.uuid4().hex)
        self.provider_history: list[dict[str, Any]] = []
        self.output = self.project / "runs" / output_group / self.state.run_id
        (self.output / "provider").mkdir(parents=True)
        self.tools = [tool for tool in self.environment.get_tools()
                      if tool.name != "get_item_details"]
        self.tool_names = {tool.name for tool in self.tools} | set(self.preparations)
        self.initial_db_hash = self.environment.get_db_hash()
        self.seen_call_ids: set[str] = set()
        upstream = self.project / "upstream/tau2-bench-verified"
        commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
        ).strip()
        if commit != "864350a8971a8f8ee9e7b8472e2edc380a806b0c":
            raise ValueError("upstream differs from the audited P0 commit")
        scope = (
            "Account registration and new purchases are unsupported in official "
            "mode; follow the policy transfer rule."
        )
        if retail_mode == "extended":
            scope = (
                "EXTENDED MODE EXCEPTIONS: registration and purchasing are supported. "
                "Public catalog browsing and registration do not require authentication. "
                "For registration collect all fields at once: first/last name, email, "
                "street address, city, state/province, country, postal code; address2 is optional. "
                "Use prepare_registration, then after runtime confirmation register_account. "
                "The backend assigns IDs and a simulation test payment method automatically. "
                "After registration the new customer is authenticated; use their returned IDs. "
                "For purchase browse list_all_product_types then get_product_details. "
                "Use prepare_purchase, then after runtime confirmation place_order. "
                "Do not claim either operation is unsupported or transfer these requests."
            )
        self.system_prompt = (
            INSTRUCTIONS + "\n\n" + scope + "\n\n<policy>\n"
            + self.environment.get_policy() + "\n</policy>"
        )
        save_json(self.output / "manifest.json", {
            "phase": output_group, "run_id": self.state.run_id, "commit": commit,
            "model": asdict(config), "max_model_calls": max_model_calls,
            "max_total_tokens": max_total_tokens,
            "initial_db_hash": self.initial_db_hash,
            "policy_sha256": hashlib.sha256(
                self.environment.get_policy().encode("utf-8"),
            ).hexdigest(),
            "protocol": ("official_orchestrator_toolflow" if output_group == "benchmark_runtime"
                         else "interactive_cli_not_official_benchmark" if retail_mode == "official"
                         else "toolflow_extended_retail_not_official_benchmark"),
            "retail_mode": retail_mode,
            "store_path": str(store_path) if store_path is not None else None,
            "runtime_revision": RUNTIME_REVISION,
            "disabled_tools": {"get_item_details": "pinned upstream _get_item defect"},
            "internal_tools": list(self.preparations),
            "enhanced_validation": enhanced_validation, "recovery": recovery,
            "max_repairs": self.repair_limit,
            "system_prompt": self.system_prompt,
        })
        self.event("run_started", {"run_id": self.state.run_id})

    def event(self, kind: str, data: dict[str, Any]) -> None:
        """追加一条不可变的事件快照，并持久化当前状态。"""
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": kind, "data": data,
        }
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.checkpoint()

    def checkpoint(self) -> None:
        """即使在出错或中断之前，也保存官方消息与状态。"""
        save_json(self.output / "messages.json", [
            message.model_dump(mode="json") for message in self.state.messages
        ])
        save_json(self.output / "model_history.json", self.provider_history)
        summary = asdict(self.state)
        summary.pop("messages")
        summary["identity_state"] = self.guard.identity_summary()
        backend = self.config.backend
        summary["model_backend"] = backend
        summary["api_fee"] = 0.0 if backend == "local" else None
        summary["api_fee_status"] = (
            "local_no_api_charge" if backend == "local" else "not_estimated"
        )
        summary["business_success"] = "requires_independent_verification"
        save_json(self.output / "summary.json", summary)

    def receive(self, content: str) -> None:
        """向当前活动会话追加一条真实的操作员/用户消息。"""
        if self.state.status != "waiting_user":
            raise ValueError("expected a non-empty user message while waiting")
        content = clean_input(content)
        confirmed = self.guard.receive(content)
        message = UserMessage(role="user", content=content)
        self.state.messages.append(message)
        self.provider_history.append(provider_message(message))
        if confirmed:
            proposal = self.guard.pending
            self.provider_history.append({"role": "system", "content": (
                "Runtime: the user explicitly confirmed the immediately preceding "
                "review. Submit exactly this native tool call, without changing "
                f"parameters: {proposal.name} "
                + json.dumps(proposal.arguments, ensure_ascii=False)
            )})
        self.state.turn_repairs = 0
        self.state.recovery_action = None
        self.state.status = "running"
        self.event("user_message", message.model_dump(mode="json"))

    def _request(self) -> AssistantMessage:
        """记录一次真实的 provider 请求及其未经修改的响应。"""
        if self.state.status != "running":
            raise RuntimeError("decide requires a running session")
        if self.state.model_calls >= self.max_model_calls:
            raise RuntimeError("run model-call budget exhausted")
        if self.state.total_tokens + self.config.max_tokens > self.max_total_tokens:
            raise RuntimeError("run token budget exhausted")
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.provider_history)
        messages.append({"role": "system", "content": (
            "Trusted runtime state (observed evidence only): "
            + json.dumps({
                "authenticated_user_id": self.guard.user_id,
                "identity_state": self.guard.identity_summary(),
                "identity_input_from_user": self.guard.identity_input() if not self.guard.user_id else None,
                "profile_read": self.guard.profile is not None,
                "orders_read": list(self.guard.orders),
                "observed_orders": [
                    {"order_id": order_id, "status": order["status"],
                     "item_quantities": dict(Counter(item["item_id"] for item in order["items"]))}
                    for order_id, order in self.guard.orders.items()
                ],
                "product_catalogs_read": list(self.guard.products),
                "named_exchange_items_from_user": {
                    order_id: sorted(self.guard.requested_items(order))
                    for order_id, order in self.guard.orders.items()
                },
                "exact_write_confirmed": self.guard.confirmed is not None,
            })
            + ". Only an untried identity candidate is actionable. If identity_state "
            "is needs_user_input, ask for corrected identity information and do not call tools. "
            "If unauthenticated and identity_input_from_user is present, call "
            "that native authentication tool now. Otherwise ask for email or first "
            "name + last name + ZIP for retail requests. After authentication, use get_user_details to "
            "retrieve orders and payment methods. Inspect required product catalogs "
            "before choosing variants. Do not end with 'please wait' or a promise "
            "to check tools. For exchange use the read-only prepare_exchange "
            "tool when ready: runtime will show its complete review and collect "
            "confirmation. Do not ask "
            "for a separate model-authored confirmation first."
        )})
        self.state.model_calls += 1
        number = self.state.model_calls
        self.event("model_started", {"number": number})
        extended = getattr(self, "retail_mode", "official") == "extended"
        visible_tools = self.tools if self.guard.user_id else [
            tool for tool in self.tools if tool.name.startswith("find_user_id_by_")
            or tool.name == "transfer_to_human_agents"
            or (extended and tool.name in {"register_account", "list_all_product_types", "get_product_details"})
        ]
        schemas = [tool.openai_schema for tool in visible_tools]
        if self.guard.user_id and not self.guard.confirmed:
            schemas.append(prepare_exchange_schema(self.tools))
        if extended:
            for preview, name in (("prepare_registration", "register_account"),
                                  ("prepare_purchase", "place_order")):
                if self.guard.confirmed or (name == "place_order" and not self.guard.user_id):
                    continue
                if name == "register_account" and self.guard.user_id:
                    continue
                source = next(tool for tool in self.tools if tool.name == name)
                schema = deepcopy(source.openai_schema)
                schema["function"]["name"] = preview
                schema["function"]["description"] = (
                    f"Read-only runtime preview for {name}. Use BEFORE asking for "
                    "confirmation. Produces a grounded review without changing data. "
                    f"After the runtime review is confirmed, call {name} with exact approved arguments."
                )
                schemas.append(schema)
            messages.append({"role": "system", "content": (
                "Extended mode: registration is supported before authentication. "
                "Ask for missing registration fields, or call prepare_registration "
                "when complete. Never demand an existing account to register. "
                "For explicit registration requests do not call authentication "
                "first, even if identity fields are present. "
                "A newly registered user is already authenticated. Purchase uses "
                "prepare_purchase for confirmation and place_order to commit."
            )})
        if self.guard.needs_identity_input():
            messages.append({"role": "system", "content": (
                "The identity lookup returned no matching account. The failed input is not usable "
                "evidence for another query. Ask the user to correct their email or provide real "
                "name and ZIP, in their language. Do not invent identity fields or report a system "
                "failure. Wait for the user's answer."
            )})
        body = self.model.build_request(messages, schemas)
        if self.guard.needs_identity_input():
            body["tool_choice"] = "none"
        save_json(self.output / "provider" / f"{number:03}.request.json", body)
        raw, elapsed = self.model.complete(body)
        save_json(self.output / "provider" / f"{number:03}.response.json", raw)
        self.state.model_seconds += elapsed
        usage = raw.get("usage") or {}
        self.state.total_tokens += usage.get("total_tokens", 0)
        self.event("model_finished", {
            "number": number, "seconds": elapsed, "usage": usage,
        })
        if self.state.total_tokens > self.max_total_tokens:
            raise RuntimeError("provider usage exceeded run token budget")
        return parse_response(
            raw, api_fee=0.0 if self.config.backend == "local" else None,
        )

    def _say(self, content: str, source: str) -> AssistantMessage:
        """记录一条由运行时撰写的消息，并标注明确来源。"""
        message = AssistantMessage(role="assistant", content=content)
        self.state.messages.append(message)
        self.provider_history.append(provider_message(message))
        self.state.status = "waiting_user"
        self.event("assistant_message", {
            "source": source, "message": message.model_dump(mode="json"),
        })
        return message

    def decide(self) -> AssistantMessage:
        """校验原生工具提议，在可配置的连续失败预算内给出修正反馈。"""
        while True:
            candidate = None
            try:
                candidate = self._request()
                if candidate.tool_calls and self.guard.needs_identity_input():
                    raise GuardError("USER_INPUT_REQUIRED", "A failed identity lookup requires corrected user information before any further tool call.")
                if not candidate.tool_calls:
                    self.guard.check_text(candidate.content)
                if candidate.tool_calls:
                    call = candidate.tool_calls[0]
                    preparations = getattr(self, "preparations", PREPARATIONS)
                    if call.name not in self.tool_names and call.name not in preparations:
                        raise GuardError("UNKNOWN_OR_DISABLED_TOOL", "Use only provided tools. For variants use get_product_details.")
                    if call.id in self.seen_call_ids:
                        raise ModelProtocolError("duplicate provider call ID")
                    self.seen_call_ids.add(call.id)
                    self.provider_history.append(provider_message(candidate))
                    prepared = call.name in preparations
                    checked_call = call.model_copy(update={"name": preparations[call.name]}) if prepared else call
                    if hasattr(self.environment, "refresh"):
                        self.environment.refresh()
                    proposal = self.guard.check(checked_call)
                    if prepared and proposal is None:
                        raise GuardError("WRITE_ALREADY_CONFIRMED", f"The preview is already confirmed. Call {checked_call.name} with the exact approved parameters now.")
                    if proposal is not None:
                        # 这是一个延迟确认的提议，而非已执行的官方调用。
                        self.provider_history.append({
                            "role": "tool", "tool_call_id": call.id,
                            "content": json.dumps({"code": "CONFIRMATION_REQUIRED", "executed": False}),
                        })
                        self.event("confirmation_required", {
                            "proposal": asdict(proposal), "provider_call_id": call.id,
                        })
                        return self._say(proposal.review, "runtime_confirmation")
                self.state.messages.append(candidate)
                if not candidate.tool_calls:
                    self.provider_history.append(provider_message(candidate))
                    # 之后的“同意”绝不能批准此前已被取代的审核。
                    self.guard.pending = None
                    self.guard.confirmed = None
                self.state.status = "waiting_tool" if candidate.tool_calls else "waiting_user"
                self.event("assistant_message", {
                    "source": "model", "message": candidate.model_dump(mode="json"),
                })
                return candidate
            except (GuardError, ModelProtocolError) as error:
                code = error.code if isinstance(error, GuardError) else "MODEL_PROTOCOL"
                self.state.blocked_proposals += 1
                decision = classify_guard_error(code, str(error), self.guard)
                if decision.action == "repair":
                    self.state.turn_repairs += 1
                self.state.recovery_action = decision.action
                feedback = {"code": code, "executed": False, "detail": str(error),
                            "next_action": decision.action, "recovery": decision.instruction}
                if (candidate and candidate.tool_calls
                        and self.provider_history
                        and self.provider_history[-1].get("tool_calls")
                        and self.provider_history[-1]["tool_calls"][0]["id"] == candidate.tool_calls[0].id):
                    self.provider_history.append({
                        "role": "tool", "tool_call_id": candidate.tool_calls[0].id,
                        "content": json.dumps(feedback, ensure_ascii=False),
                    })
                else:
                    self.provider_history.append({"role": "system", "content": (
                        "Runtime rejected your last output before execution. "
                        + json.dumps(feedback, ensure_ascii=False)
                        + " Correct the format/parameters, or ask the user for missing information."
                    )})
                self.event("proposal_blocked", {
                    **feedback,
                    "provider_call_id": (candidate.tool_calls[0].id
                                         if candidate and candidate.tool_calls else None),
                })
                self.event("recovery_decision", {"code": code, "next_action": decision.action})
                if decision.action == "ask_user":
                    self.state.turn_repairs = 0
                    self.guard.pending = None
                    self.guard.confirmed = None
                    return self._say(decision.question or IDENTITY_QUESTION, "runtime_information_request")
                if self.state.turn_repairs >= getattr(self, "repair_limit", DEFAULT_MAX_REPAIRS):
                    self.guard.pending = None
                    self.guard.confirmed = None
                    return self._say(
                        f"当前步骤已达到 {getattr(self, 'repair_limit', DEFAULT_MAX_REPAIRS)} 次连续修正上限，已暂停；未执行被拦截的工具提议。"
                        f"原因：{code}，{error}。请根据原因补充信息或调整需求后继续。",
                        "runtime_recovery_pause",
                    )

    def execute(self, message: AssistantMessage) -> ToolMessage:
        """通过官方环境执行且仅执行一个待处理的工具调用。"""
        if self.state.status != "waiting_tool" or not message.tool_calls:
            raise RuntimeError("there is no pending tool call")
        call = message.tool_calls[0]
        if message is not self.state.messages[-1]:
            raise RuntimeError("tool request is not the latest accepted decision")
        # 在执行边界处再次校验，使延迟/过期的决策
        # 无法绕过确认或时效性检查。
        if hasattr(self.environment, "refresh"):
            self.environment.refresh()
        proposal = self.guard.check(call)
        if proposal is not None:
            raise RuntimeError("write requires fresh user confirmation")
        write = self.guard.is_write(call)
        before = self.environment.get_db_hash() if write else None
        if write:
            if hasattr(self.environment, "expected_write_hash"):
                self.environment.expected_write_hash = before
            self.guard.consume_confirmation(call)
            self.event("write_attempt", {"fingerprint": fingerprint(call), "db_hash": before})
        started = time.perf_counter()
        self.event("tool_started", call.model_dump(mode="json"))
        response = self.environment.get_response(call)
        assert response.id == call.id, "official tool correlation changed"
        self.state.tool_calls += 1
        self.state.messages.append(response)
        self.provider_history.append(provider_message(response))
        self.guard.observe(call, response)
        if not response.error:
            # 当前工具已成功完成，后续步骤重新计算连续失败次数。
            self.state.turn_repairs = 0
        self.state.status = "running"
        self.event("tool_finished", {
            "name": call.name, "seconds": time.perf_counter() - started,
            "response": response.model_dump(mode="json"),
        })
        if write:
            if response.error:
                if self.environment.get_db_hash() != before:
                    self.state.status = "failed"
                    self.state.stop_reason = "write_outcome_error"
                    self.state.recovery_action = "stop"
                    self.event("recovery_decision", {"code": "WRITE_OUTCOME_UNCERTAIN", "next_action": "stop"})
                    self.event("write_outcome_uncertain", {"call_id": call.id})
                    raise RuntimeError("write failed after state changed; no automatic retry")
            else:
                self._read_back(call, response)
        if response.error:
            self.record_tool_error(call, response)
        return response

    def record_tool_error(self, call: ToolCall, response: ToolMessage) -> None:
        """共享执行后错误分流，用户信息缺失不计为模型连续修正失败。

        Parameters
        ----------
        call : ToolCall
            实际执行过的调用。
        response : ToolMessage
            已记录并传给守卫观察的官方结果。
        """
        decision = classify_tool_error(call, response, self.guard)
        self.state.recovery_action = decision.action
        self.state.turn_repairs = self.state.turn_repairs + 1 if decision.action == "repair" else 0
        feedback = {"code": decision.code, "call_id": call.id, "detail": response.content,
                    "next_action": decision.action, "recovery": decision.instruction}
        self.provider_history.append({"role": "system", "content": json.dumps(feedback)})
        self.event("tool_error_classified", feedback)
        self.event("recovery_decision", {"code": decision.code, "next_action": decision.action})
        if self.state.turn_repairs >= getattr(self, "repair_limit", DEFAULT_MAX_REPAIRS):
            self._say("当前步骤多次修正仍未成功，请补充信息或调整需求；失败记录已保存。", "runtime_recovery_pause")

    def _read_back(self, write: ToolCall, response: ToolMessage) -> None:
        """通过官方工具回读被修改的实体并验证结果。"""
        result = json.loads(response.content)
        profile = write.name == "register_account" or (
            "order_id" not in write.arguments and write.name != "place_order"
        )
        key = "user_id" if profile else "order_id"
        call = ToolCall(
            id="tf_readback_" + uuid.uuid4().hex,
            name="get_user_details" if profile else "get_order_details",
            arguments={key: write.arguments.get(key, result[key])},
        )
        message = AssistantMessage(role="assistant", tool_calls=[call])
        self.state.messages.append(message)
        self.provider_history.append(provider_message(message))
        self.event("readback_started", {"source": "runtime", "call": call.model_dump(mode="json")})
        checked = self.environment.get_response(call)
        self.state.tool_calls += 1
        self.state.messages.append(checked)
        self.provider_history.append(provider_message(checked))
        self.guard.observe(call, checked)
        match = not checked.error and json.loads(checked.content) == json.loads(response.content)
        self.event("readback_finished", {
            "source": "runtime", "matches_write_result": match,
            "response": checked.model_dump(mode="json"),
        })
        if not match:
            self.state.status = "failed"
            self.state.stop_reason = "readback_error"
            self.state.recovery_action = "stop"
            self.event("recovery_decision", {"code": "READBACK_MISMATCH", "next_action": "stop"})
            self.event("integrity_failure", {"call_id": write.id, "kind": "readback_mismatch"})
            raise RuntimeError("write readback mismatch; inspect state before any retry")
        data = json.loads(checked.content)
        content = f"模拟数据库操作已提交并回查一致：{write.name}。"
        if write.name == "register_account":
            content += f"账号 {data['user_id']}，邮箱 {data['email']}；已自动进入该账号会话，测试支付方式已创建。"
        if not profile:
            content += f"订单 {data['order_id']}，当前状态：{data['status']}。"
        if write.name == "place_order":
            total = data["payment_history"][0]["amount"]
            content += f"共 {len(data['items'])} 件，总价 USD ${total:.2f}；模拟支付，不实际扣款。"
        difference = data.get("exchange_price_difference")
        if difference is not None:
            direction = "退回" if difference < 0 else "补付" if difference > 0 else "无需补退"
            content += f"差价 ${difference:.2f}，{direction} ${abs(difference):.2f}；方式 {data['exchange_payment_method_id']}。"
        content += "这是模拟系统中的申请/更新结果。"
        self._say(content, "runtime_verified_receipt")

    def close(self, reason: str) -> None:
        """持久化最终数据库与停止原因，但不声称任务成功。"""
        self.state.status = "failed" if reason.endswith("error") else "stopped"
        self.state.stop_reason = reason
        save_json(
            self.output / "final_db.json",
            self.environment.tools.db.model_dump(mode="json"),
        )
        self.event("run_stopped", {"reason": reason})
