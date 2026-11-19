"""LangGraph 重构的行为验收：真实 Retail 工具，模型边界使用离线响应。"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from run_cli import ROOT, bootstrap

bootstrap()

from tau2.data_model.message import ToolMessage
from tau2.domains.retail.environment import get_environment, get_tasks

from toolflow.benchmark import BenchmarkAgentAdapter
from toolflow.model import ModelConfig
from toolflow.runtime import RetailRuntime


def raw_text(content: str) -> dict:
    """构造测试用原生文本协议，生产路径不引用此函数。"""
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
        "usage": {"total_tokens": 10},
    }


def raw_call(name: str, arguments: dict, call_id: str) -> dict:
    """构造单工具协议，参数仍由真实业务守卫验证。"""
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"total_tokens": 10},
    }


class LangGraphTests(unittest.TestCase):
    """验证图的跳转、真实暂停恢复和本机/官方执行权边界。"""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = RetailRuntime(
            ROOT,
            ModelConfig(),
            output_group=self.temp.name,
        )

    def records(self) -> list[dict]:
        """读取实际运行事件，检查走过的节点而非模拟路由函数。"""
        return [
            json.loads(line)
            for line in (self.runtime.output / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def test_multi_tool_loop_then_user_interrupt_and_resume(self) -> None:
        """一个图内完成多个查询；新消息恢复暂停点，不重复旧工具。"""
        identity = {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}
        fixtures = [
            raw_call("find_user_id_by_name_zip", identity, "identity"),
            raw_call("get_user_details", {"user_id": "yusuf_rossi_9620"}, "profile"),
            raw_text("您想处理哪个订单？"),
            raw_text("请告诉我您的换货需求。"),
        ]
        with patch.object(
            self.runtime.model,
            "complete",
            side_effect=[(item, 0.01) for item in fixtures],
        ) as complete, patch.object(
            self.runtime.environment,
            "get_response",
            wraps=self.runtime.environment.get_response,
        ) as execute:
            self.runtime.run_turn("我是 Yusuf Rossi，邮编 19122，想换货。")
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(self.runtime.flow.next_nodes, ("wait_user",))
            self.runtime.run_turn("我还没想好。")
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(complete.call_count, 4)
        self.assertEqual(self.runtime.state.status, "waiting_user")
        users = [m for m in self.runtime.state.messages if m.role == "user"]
        self.assertEqual(len(users), 2)
        nodes = [
            r["data"]["node"]
            for r in self.records()
            if r["event"] == "graph_node_completed"
        ]
        self.assertEqual(nodes.count("execute"), 2)
        self.assertIn("wait_user", nodes)
        request = json.loads(
            (self.runtime.output / "provider/004.request.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            len([m for m in request["messages"] if m["role"] == "user"]), 2
        )

    def test_protocol_failure_routes_through_recovery_node(self) -> None:
        """格式失败反馈回模型，由边形成循环，不靠节点内部 while。"""
        fixtures = [{"choices": []}, raw_text("请提供您的邮箱或姓名和邮编。")]
        with patch.object(
            self.runtime.model,
            "complete",
            side_effect=[(item, 0.01) for item in fixtures],
        ):
            self.runtime.run_turn("我要换货。")
        nodes = [
            r["data"]["node"]
            for r in self.records()
            if r["event"] == "graph_node_completed"
        ]
        self.assertEqual(nodes.count("model"), 2)
        self.assertIn("recover", nodes)
        self.assertEqual(self.runtime.state.blocked_proposals, 1)
        self.assertEqual(self.runtime.flow.next_nodes, ("wait_user",))

    def test_bad_identity_asks_user_without_retrying_lookup(self) -> None:
        """真实查询失败后，模型坚持重复查也只会补问，不执行第二次。"""
        fixtures = [
            raw_call(
                "find_user_id_by_email", {"email": "missing@example.invalid"}, "first"
            ),
            raw_call(
                "find_user_id_by_email", {"email": "missing@example.invalid"}, "repeat"
            ),
        ]
        with patch.object(
            self.runtime.model,
            "complete",
            side_effect=[(item, 0.01) for item in fixtures],
        ), patch.object(
            self.runtime.environment,
            "get_response",
            wraps=self.runtime.environment.get_response,
        ) as execute:
            message = self.runtime.run_turn(
                "我的邮箱 missing@example.invalid，我要退货。"
            )
            self.assertEqual(execute.call_count, 1)
        self.assertIn("核", message.content)
        self.assertEqual(self.runtime.state.recovery_action, "ask_user")
        self.assertEqual(self.runtime.state.turn_repairs, 0)

    def test_confirmation_resume_single_write_and_readback(self) -> None:
        """离线参考动作仅在测试出现，验证完整图的一次写入和回查。"""
        task = next(t for t in get_tasks() if t.id == "0")
        actions = task.evaluation_criteria.actions
        fixtures = [
            raw_call(actions[0].name, actions[0].arguments, "identity"),
            raw_call("get_user_details", {"user_id": "yusuf_rossi_9620"}, "profile"),
        ]
        fixtures += [
            raw_call(a.name, a.arguments, f"read{i}")
            for i, a in enumerate(actions[1:4])
        ]
        fixtures += [
            raw_call("prepare_exchange", actions[4].arguments, "preview"),
            raw_call(actions[4].name, actions[4].arguments, "commit"),
        ]
        with patch.object(
            self.runtime.model,
            "complete",
            side_effect=[(item, 0.01) for item in fixtures],
        ) as model, patch.object(
            self.runtime.environment,
            "get_response",
            wraps=self.runtime.environment.get_response,
        ) as execute:
            self.runtime.run_turn(
                "Yusuf Rossi 19122。订单 #W2378156：键盘换 clicky 青轴，全尺寸不变；"
                "RGB 没货可以无背光；黑色恒温器换 Google Assistant，黑色不变。用原信用卡退差价。"
            )
            self.assertEqual(
                self.runtime.environment.tools.db.orders["#W2378156"].status,
                "delivered",
            )
            self.assertIsNotNone(self.runtime.guard.pending)
            self.assertEqual(self.runtime.flow.next_nodes, ("wait_user",))
            receipt = self.runtime.run_turn("yes")
            writes = [
                c
                for c in execute.call_args_list
                if c.args[0].name == "exchange_delivered_order_items"
            ]
            self.assertEqual(len(writes), 1)
            self.assertTrue(
                execute.call_args_list[-1].args[0].id.startswith("tf_readback_")
            )
            self.assertEqual(model.call_count, len(fixtures))
        self.assertIn("回查一致", receipt.content)
        self.assertEqual(self.runtime.flow.next_nodes, ("wait_user",))
        self.assertEqual(
            self.runtime.environment.tools.db.orders["#W2378156"].status,
            "exchange requested",
        )

    def test_external_interrupt_has_no_local_execution_and_checks_id(self) -> None:
        """官方模式输出 ToolCall 后暂停，错误 ID 不消耗该暂停点。"""
        environment = get_environment()
        agent = BenchmarkAgentAdapter(ROOT, ModelConfig(), environment)
        with patch("toolflow.benchmark.RetailRuntime", return_value=self.runtime):
            runtime = agent.get_init_state()
        from tau2.data_model.message import UserMessage

        fixture = raw_call(
            "find_user_id_by_email", {"email": "missing@example.invalid"}, "identity"
        )
        with patch.object(
            runtime.model, "complete", return_value=(fixture, 0.01)
        ), patch.object(
            runtime.environment,
            "get_response",
            wraps=runtime.environment.get_response,
        ) as execute:
            candidate, _ = agent.generate_next_message(
                UserMessage(
                    role="user",
                    content="我的邮箱 missing@example.invalid，我要退货。",
                ),
                runtime,
            )
            execute.assert_not_called()
            self.assertEqual(runtime.flow.next_nodes, ("wait_tool",))
            wrong = ToolMessage(
                role="tool",
                id="wrong",
                name="find_user_id_by_email",
                requestor="assistant",
                content="Error: not found",
                error=True,
            )
            with self.assertRaisesRegex(ValueError, "correlation"):
                agent.generate_next_message(wrong, runtime)
            self.assertEqual(runtime.flow.next_nodes, ("wait_tool",))
            response = runtime.environment.get_response(candidate.tool_calls[0])
        with patch.object(
            runtime.model, "complete", return_value=(raw_text("请核对邮箱。"), 0.01)
        ):
            agent.generate_next_message(response, runtime)
        self.assertEqual(runtime.flow.next_nodes, ("wait_user",))

    def test_shutdown_between_model_and_execution_never_calls_tool(self) -> None:
        """模型返回后收到停止信号，尚未授权的工具不会开始执行。"""
        fixture = raw_call(
            "find_user_id_by_email", {"email": "missing@example.invalid"}, "identity"
        )
        stop = {"requested": False}

        def on_node(node: str, data: dict) -> None:
            if node == "model":
                stop["requested"] = True

        with patch.object(
            self.runtime.model, "complete", return_value=(fixture, 0.01)
        ), patch.object(
            self.runtime.environment,
            "get_response",
        ) as execute:
            # 停止发生在第一次回复之前，运行方法应允许没有 assistant 输出。
            self.runtime.run_turn(
                "我的邮箱 missing@example.invalid，我要退货。",
                on_node=on_node,
                should_stop=lambda: stop["requested"],
            )
            execute.assert_not_called()
        self.assertEqual(self.runtime.state.status, "stopped")

    def test_checkpoint_is_disk_backed_serializable_and_session_scoped(self) -> None:
        """检查点实际落入独立 SQLite，而不是仅给日志起一个框架名字。"""
        with patch.object(
            self.runtime.model,
            "complete",
            return_value=(raw_text("请提供邮箱。"), 0.01),
        ):
            self.runtime.run_turn("我要退货。")
        output = self.runtime.output
        snapshot = json.loads((output / "graph_state.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["next_nodes"], ["wait_user"])
        self.assertIn(self.runtime.state.run_id, snapshot["thread_id"])
        self.assertFalse(snapshot["restart_resume_supported"])
        with sqlite3.connect(output / "graph.sqlite3") as connection:
            rows = connection.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ).fetchall()
            self.assertEqual(rows, [(self.runtime.state.run_id + ":local",)])
        topology = (output / "graph.mmd").read_text(encoding="utf-8")
        for name in ("model", "recover", "wait_tool", "wait_user", "verify_readback"):
            self.assertIn(name, topology)
