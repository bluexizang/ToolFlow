"""按下一步所需条件分流错误，而不是统一消耗模型修正次数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tau2.data_model.message import ToolCall, ToolMessage

RUNTIME_REVISION = "identity_recovery_v2"
IDENTITY_QUESTION = "未能核实您的账户。请核对邮箱，或提供您的姓名和邮政编码，以便继续查询。"


@dataclass(frozen=True)
class RecoveryDecision:
    """区分模型可修正、需要用户输入以及可以继续的情况。"""

    code: str
    action: Literal["repair", "ask_user", "continue", "stop"]
    instruction: str
    question: str | None = None


def classify_guard_error(code: str, detail: str, guard: Any) -> RecoveryDecision:
    """结合当前证据确定拒绝后的下一步。

    Parameters
    ----------
    code : str
        程序守卫的结构化错误码。
    detail : str
        原始错误说明。
    guard : Any
        当前会话的证据与身份状态。

    Returns
    -------
    RecoveryDecision
        模型修正或用户补充信息的决策。
    """
    if guard.needs_identity_input():
        return RecoveryDecision(code, "ask_user", "Identity information is missing after a failed lookup. Ask the user instead of attempting more autonomous repairs.", IDENTITY_QUESTION)
    identity_errors = {"UNGROUNDED_IDENTITY", "IDENTITY_INPUT_REJECTED", "AUTH_REQUIRED",
                       "AUTH_QUESTION_REQUIRED", "USER_INPUT_REQUIRED"}
    if code in identity_errors and guard.identity_input() is None:
        return RecoveryDecision(code, "ask_user", "Ask for corrected identity information; do not invent fields or repeat a failed lookup.", IDENTITY_QUESTION)
    if code in {"WRONG_OWNER", "IDENTITY_LOCKED"}:
        return RecoveryDecision(code, "ask_user", "Ask which order belongs to the authenticated customer; do not guess another customer's identity.",
                                "请提供当前账户本人的订单信息；如需切换账户，请新建对话后重新核实身份。")
    return RecoveryDecision(code, "repair", detail)


def classify_tool_error(call: ToolCall, result: ToolMessage, guard: Any) -> RecoveryDecision:
    """把官方工具的既定错误转换为明确的恢复动作。

    Parameters
    ----------
    call : ToolCall
        实际执行的调用。
    result : ToolMessage
        官方返回的错误结果；此处不猜测任意自然语言错误。
    guard : Any
        已经观察本次结果的会话守卫。

    Returns
    -------
    RecoveryDecision
        已有其他身份输入则继续，否则正常请求用户补充。
    """
    if call.name.startswith("find_user_id_by_") and result.content.strip() == "Error: User not found":
        if guard.identity_input() is not None:
            return RecoveryDecision("IDENTITY_NOT_FOUND", "continue", "Use a different, untried identity candidate already provided by the user. Do not repeat the failed input.")
        return RecoveryDecision("IDENTITY_NOT_FOUND", "ask_user", "Ask the user to correct the email or provide their real name and ZIP. No further lookup is possible without new information.", IDENTITY_QUESTION)
    return RecoveryDecision("OFFICIAL_TOOL_ERROR", "repair", "Inspect the error and observed state, correct arguments or ask for missing information. Never blindly repeat a write. " + result.content)
