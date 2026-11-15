"""P1 协议/执行检查；这些 fixture 不会生成 Agent 动作。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from run_p1 import bootstrap

bootstrap()

from loguru import logger
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.domains.retail.environment import get_environment
from toolflow.model import LocalChatModel, ModelConfig, ModelProtocolError, parse_response
from toolflow.runtime import RetailRuntime, SessionState, provider_message
from toolflow.validation import RetailGuard

logger.remove()


class ProtocolTests(unittest.TestCase):
    """在任何推理之前检查协议转换和仅限本地的传输。"""

    def setUp(self) -> None:
        """创建仅用于解析器的协议 fixture，绝不是替代模型。"""
        self.response = {
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "unit-call-id", "type": "function", "function": {
                        "name": "get_order_details",
                        "arguments": '{"order_id":"#W2378156"}',
                    },
                }],
            }}],
            "usage": {"total_tokens": 100},
        }

    def test_preserve_provider_ids_and_arguments(self) -> None:
        """在 API 边界两侧保留 provider ID。"""
        message = parse_response(self.response)
        self.assertEqual(message.tool_calls[0].id, "unit-call-id")
        self.assertEqual(message.tool_calls[0].arguments, {"order_id": "#W2378156"})
        converted = provider_message(message)
        self.assertEqual(converted["tool_calls"][0]["id"], "unit-call-id")
        result = provider_message(ToolMessage(
            role="tool", id="unit-call-id", content="{}", requestor="assistant",
        ))
        self.assertEqual(result["tool_call_id"], "unit-call-id")
        self.assertNotIn("raw_data", converted)

    def test_reject_mixed_and_parallel_calls(self) -> None:
        """在工具能够执行之前拒绝违反 Retail 协议的请求。"""
        mixed = deepcopy(self.response)
        mixed["choices"][0]["message"]["content"] = "Executing now"
        with self.assertRaises(ModelProtocolError):
            parse_response(mixed)
        parallel = deepcopy(self.response)
        calls = parallel["choices"][0]["message"]["tool_calls"]
        calls.append(deepcopy(calls[0]))
        with self.assertRaises(ModelProtocolError):
            parse_response(parallel)

    def test_reject_truncated_and_invalid_arguments(self) -> None:
        """不执行部分生成的或非对象类型的参数。"""
        truncated = deepcopy(self.response)
        truncated["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ModelProtocolError):
            parse_response(truncated)
        invalid = deepcopy(self.response)
        invalid["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "[]"
        with self.assertRaises(ModelProtocolError):
            parse_response(invalid)

    def test_reject_remote_and_cloud_transport(self) -> None:
        """P1 不能悄悄地把此对话发送到计费端点。"""
        with self.assertRaises(ValueError):
            LocalChatModel(ModelConfig(base_url="https://example.org/v1"))
        with self.assertRaises(ValueError):
            LocalChatModel(ModelConfig(model="example:cloud"))


class ExecutionTests(unittest.TestCase):
    """使用真实的 Retail 实例和工具，不使用模型/网络替代品。"""

    def test_fresh_databases_are_isolated(self) -> None:
        """一个被修改的运行不能改动另一个运行的数据库。"""
        first, second = get_environment(), get_environment()
        first.tools.db.orders["#W2378156"].status = "exchange requested"
        self.assertEqual(second.tools.db.orders["#W2378156"].status, "delivered")

    def test_execute_once_and_preserve_snapshot(self) -> None:
        """一个待执行的真实工具只执行一次，且记录的输出不可变。"""
        with tempfile.TemporaryDirectory() as folder:
            runtime = RetailRuntime.__new__(RetailRuntime)
            runtime.config = ModelConfig()
            runtime.environment = get_environment()
            runtime.guard = RetailGuard(runtime.environment)
            runtime.guard.user_id = "yusuf_rossi_9620"
            runtime.guard.user_texts = ["查询 #W2378156"]
            runtime.provider_history = []
            runtime.output = Path(folder)
            call = ToolCall(
                id="unit-read", name="get_order_details",
                arguments={"order_id": "#W2378156"},
            )
            message = AssistantMessage(role="assistant", tool_calls=[call])
            runtime.state = SessionState(
                "unit", status="waiting_tool", messages=[message],
            )
            response = runtime.execute(message)
            self.assertFalse(response.error)
            self.assertEqual(response.id, "unit-read")
            with self.assertRaises(RuntimeError):
                runtime.execute(message)
            runtime.environment.tools.db.orders["#W2378156"].status = "cancelled"
            recorded = json.loads((Path(folder) / "messages.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(recorded[-1]["content"])["status"], "delivered")

    def test_budget_stops_before_model_request(self) -> None:
        """预算耗尽时，在访问任何模型对象之前就抛出异常。"""
        runtime = RetailRuntime.__new__(RetailRuntime)
        runtime.state = SessionState("unit", status="running", model_calls=2)
        runtime.max_model_calls = 2
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            runtime.decide()


if __name__ == "__main__":
    unittest.main()
