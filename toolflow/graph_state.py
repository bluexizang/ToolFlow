"""图中传递的可序列化数据；不把模型连接、守卫或数据库放进检查点。"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class FlowState(TypedDict, total=False):
    """每个节点只返回需要更新的字段，其他字段由 LangGraph 保留。

    `mode` 区分本机执行、官方编排器执行与逐步调试；它们使用同一张图。
    `candidate` 是模型提议，`response` 是已经执行的结果，两者不能混淆。
    字段均为普通字典、字符串或数值，可以保存到 SQLite 检查点。
    """

    entry: Literal["receive", "model", "authorize", "observe"]
    mode: Literal["local", "external", "decision", "execution"]
    user_input: str
    candidate: dict[str, Any] | None
    error: dict[str, str] | None
    route: str
    review: str | None
    call: dict[str, Any] | None
    response: dict[str, Any] | None
    execution_result: dict[str, Any] | None
    before_write: str | None
    write_call: dict[str, Any] | None
    write_result: str | None
    is_readback: bool
    started_at: float
