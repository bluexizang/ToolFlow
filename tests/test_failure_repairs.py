"""使用官方工具和离线 HTTP fixture 复现已观察到的失败。

这些 fixture 测试的是执行契约；它们不是 LLM 可靠性度量。
运行时或守卫代码不会导入任何参考动作。
"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from run_p1 import bootstrap

bootstrap()

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from toolflow.console import clean_input, read_user_input
from toolflow.model import ModelConfig, create_model
from toolflow.runtime import RetailRuntime, SessionState
from toolflow.validation import GuardError, RetailGuard, fingerprint


IDENTITY = {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}
USER = "yusuf_rossi_9620"
ORDER = "#W2378156"
EXCHANGE = {
    "order_id": ORDER, "item_ids": ["1151293680", "4983901480"],
    "new_item_ids": ["7706410293", "7747408585"],
    "payment_method_id": "credit_card_9513926",
}
REQUEST = (
    "订单 #W2378156：键盘换 clicky 青轴，全尺寸不变；RGB 没货可以无背光；"
    "黑色恒温器换 Google Assistant，黑色不变。用原信用卡退差价。"
)


def call(name: str, arguments: dict, identity: str = "fixture") -> ToolCall:
    """构造一个显式的测试提议，绝不是推理实现。"""
    return ToolCall(id=identity, name=name, arguments=arguments)


def observe(guard: RetailGuard, name: str, arguments: dict) -> None:
    """通过未改动的官方执行器传递一个已批准的读取。"""
    proposal = call(name, arguments)
    assert guard.check(proposal) is None
    result = guard.environment.get_response(proposal)
    assert not result.error, result.content
    guard.observe(proposal, result)


def prepared_guard() -> RetailGuard:
    """使用真实的认证/资料/订单工具构建观察到的证据。"""
    guard = RetailGuard(get_environment())
    guard.receive("Yusuf Rossi，ZIP 19122")
    observe(guard, "find_user_id_by_name_zip", IDENTITY)
    observe(guard, "get_user_details", {"user_id": USER})
    observe(guard, "get_order_details", {"order_id": ORDER})
    for product in ("1656367028", "4896585277"):
        observe(guard, "get_product_details", {"product_id": product})
    guard.receive(REQUEST)
    return guard


class GuardRegressionTests(unittest.TestCase):
    """在任何数据库变更之前拒绝已知的不安全提议。"""

    def assert_blocked(self, guard: RetailGuard, proposal: ToolCall, code: str) -> None:
        """同时检查错误分类和不变的数据库哈希。"""
        before = guard.environment.get_db_hash()
        with self.assertRaises(GuardError) as caught:
            guard.check(proposal)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(before, guard.environment.get_db_hash())

    def test_failed_identity_changes_state_and_allows_clarification(self) -> None:
        """真实查不到之后，输入不再可推荐，姓名不能从邮箱拆出。"""
        guard = RetailGuard(get_environment())
        guard.receive("我是1234567890@example.invalid")
        lookup = call("find_user_id_by_email", {"email": "1234567890@example.invalid"})
        self.assertIsNone(guard.check(lookup))
        response = guard.environment.get_response(lookup)
        self.assertTrue(response.error)
        guard.observe(lookup, response)
        self.assertIsNone(guard.identity_input())
        self.assertTrue(guard.needs_identity_input())
        guard.check_text("请核对邮箱，或者提供您的姓名和邮政编码。")
        self.assert_blocked(guard, call("find_user_id_by_name_zip", {
            "first_name": "1234567890", "last_name": "qq", "zip": "com",
        }), "UNGROUNDED_IDENTITY")
        self.assert_blocked(guard, lookup, "IDENTITY_INPUT_REJECTED")
        self.assert_blocked(guard, call("find_user_id_by_email", {"email": "1234567890@EXAMPLE.INVALID"}), "IDENTITY_INPUT_REJECTED")
        guard.receive("谢谢")
        self.assertIsNone(guard.identity_input())
        guard.receive("请再核对一下1234567890@example.invalid")
        self.assertIsNotNone(guard.identity_input())
        self.assertIsNone(guard.check(lookup))

    def test_identity_fields_can_be_completed_across_user_turns(self) -> None:
        """用户分两次提供姓名和邮编时，可以组成真实身份字段。"""
        guard = RetailGuard(get_environment())
        guard.receive("我叫 Yusuf Rossi")
        self.assertIsNone(guard.identity_input())
        guard.receive("邮编19122")
        self.assertEqual(guard.identity_input()["arguments"], IDENTITY)
        self.assertIsNone(guard.check(call("find_user_id_by_name_zip", IDENTITY)))

    def test_authentication_and_identity_provenance(self) -> None:
        guard = RetailGuard(get_environment())
        guard.receive("我要换货")
        self.assert_blocked(guard, call("get_order_details", {"order_id": ORDER}), "AUTH_REQUIRED")
        self.assert_blocked(guard, call("find_user_id_by_name_zip", {
            "first_name": "用户提供的姓名", "last_name": "姓氏", "zip": "19122",
        }), "UNGROUNDED_IDENTITY")
        self.assert_blocked(guard, call("find_user_id_by_email", {"email": "madeup@example.com"}), "UNGROUNDED_IDENTITY")

    def test_example_order_rejected_real_order_discovered_and_owner_checked(self) -> None:
        guard = RetailGuard(get_environment())
        guard.receive("Yusuf Rossi 19122，我想换个键盘")
        observe(guard, "find_user_id_by_name_zip", IDENTITY)
        self.assert_blocked(guard, call("get_order_details", {"order_id": "#W0000000"}), "UNKNOWN_ORDER")
        observe(guard, "get_user_details", {"user_id": USER})
        self.assertIn(ORDER, guard.profile["orders"])
        observe(guard, "get_order_details", {"order_id": ORDER})
        other = next(order for order in guard.environment.tools.db.orders.values()
                     if order.user_id != USER)
        guard.receive(f"查一下 {other.order_id}")
        self.assert_blocked(guard, call("get_order_details", {"order_id": other.order_id}), "WRONG_OWNER")
        self.assert_blocked(guard, call("get_user_details", {"user_id": other.user_id}), "WRONG_OWNER")
        self.assert_blocked(guard, call("find_user_id_by_name_zip", IDENTITY), "IDENTITY_LOCKED")

    def test_strict_schema_and_item_lookup_workaround(self) -> None:
        guard = prepared_guard()
        self.assert_blocked(guard, call("get_order_details", {"order_id": ORDER, "extra": 1}), "ARGUMENT_SCHEMA")
        self.assert_blocked(guard, call("get_order_details", {"order_id": 123}), "ARGUMENT_SCHEMA")
        self.assert_blocked(guard, call("get_item_details", {"item_id": "1151293680"}), "UPSTREAM_ITEM_LOOKUP")
        self.assertIn("7706410293", guard.products["1656367028"]["variants"])

    def test_wrong_size_stock_same_item_empty_list_and_payment(self) -> None:
        guard = prepared_guard()
        for new, code in (("9665000388", "UNREQUESTED_OPTION"),
                          ("9025753381", "VARIANT_UNAVAILABLE"),
                          ("1151293680", "VARIANT_UNAVAILABLE")):
            args = deepcopy(EXCHANGE)
            args["new_item_ids"][0] = new
            self.assert_blocked(guard, call("exchange_delivered_order_items", args), code)
        args = deepcopy(EXCHANGE)
        args["item_ids"], args["new_item_ids"] = [], []
        self.assert_blocked(guard, call("exchange_delivered_order_items", args), "EXCHANGE_ITEMS")
        args = deepcopy(EXCHANGE)
        args["payment_method_id"] = "credit_card_0000000"
        self.assert_blocked(guard, call("exchange_delivered_order_items", args), "PAYMENT_NOT_OBSERVED")

    def test_exact_confirmation_refund_direction_and_invalidation(self) -> None:
        guard = prepared_guard()
        proposal = call("exchange_delivered_order_items", EXCHANGE)
        review = guard.check(proposal)
        self.assertIn("退回 $16.63", review.review)
        self.assertIn("full size", review.review)
        self.assertIn(EXCHANGE["payment_method_id"], review.review)
        self.assertFalse(guard.receive("确认，但改成另一张卡"))
        self.assertIsNotNone(guard.check(proposal))
        self.assertTrue(guard.receive("确认"))
        self.assertIsNone(guard.check(proposal))
        changed = deepcopy(EXCHANGE)
        changed["item_ids"] = list(reversed(EXCHANGE["item_ids"]))
        changed["new_item_ids"] = list(reversed(EXCHANGE["new_item_ids"]))
        self.assertIsNotNone(guard.check(call("exchange_delivered_order_items", changed)))
        self.assertIsNone(guard.confirmed)

    def test_stale_confirmation_and_duplicate_write(self) -> None:
        guard = prepared_guard()
        proposal = call("exchange_delivered_order_items", EXCHANGE)
        review = guard.check(proposal)
        self.assertTrue(guard.receive(f"/confirm {review.fingerprint[:12]}"))
        guard.environment.tools.db.orders[ORDER].address.city = "changed"
        self.assert_blocked(guard, proposal, "STALE_CONFIRMATION")
        guard.check(proposal)
        guard.receive("yes")
        guard.consume_confirmation(proposal)
        self.assert_blocked(guard, proposal, "WRITE_ALREADY_ATTEMPTED")

    def test_all_named_exchange_items_must_be_collected(self) -> None:
        guard = prepared_guard()
        args = deepcopy(EXCHANGE)
        args["item_ids"] = args["item_ids"][:1]
        args["new_item_ids"] = args["new_item_ids"][:1]
        self.assert_blocked(guard, call("exchange_delivered_order_items", args), "INCOMPLETE_EXCHANGE")
        guard.receive("这次只换键盘，恒温器暂时不换。")
        self.assertIsNotNone(guard.check(call("exchange_delivered_order_items", args)))

    def test_negated_option_is_not_permission(self) -> None:
        guard = prepared_guard()
        guard.receive("尺寸保持全尺寸，不要80%。")
        args = deepcopy(EXCHANGE)
        args["new_item_ids"][0] = "9665000388"
        self.assert_blocked(guard, call("exchange_delivered_order_items", args), "UNREQUESTED_OPTION")

    def test_narration_stall_and_wrong_initial_question(self) -> None:
        guard = RetailGuard(get_environment())
        guard.receive("我想换个键盘")
        with self.assertRaises(GuardError) as caught:
            guard.check_text("请提供键盘型号、颜色和付款方式。")
        self.assertEqual(caught.exception.code, "AUTH_QUESTION_REQUIRED")
        guard.check_text("请提供邮箱，或者姓名和邮编。")
        guard.receive("请讲解 LangChain")
        guard.check_text("抱歉，我只负责零售账户和订单业务。")
        guard = prepared_guard()
        with self.assertRaises(GuardError) as caught:
            guard.check_text("现在我将核实库存和差价。请稍等。")
        self.assertEqual(caught.exception.code, "TOOL_REQUIRED_FOR_PROGRESS")


class InputRegressionTests(unittest.TestCase):
    """不要把模型请求浪费在控制字符或每一行粘贴内容上。"""

    def test_identity_slots_come_from_user_text_not_database(self) -> None:
        guard = RetailGuard(get_environment())
        guard.receive("我要换货，忘记邮箱了")
        self.assertIsNone(guard.identity_input())
        guard.receive("我叫 Yusuf Rossi，邮编19122。我要换货。")
        self.assertEqual(guard.identity_input(), {"name": "find_user_id_by_name_zip", "arguments": IDENTITY})
        guard.check_text("请提供姓名和邮箱。")
        self.assertEqual(guard.identity_input()["arguments"], IDENTITY)

    def test_control_v_empty_and_oversized(self) -> None:
        for content in ("\x16", "Yusuf\x16", "  ", "a" * 16_001):
            with self.subTest(content=content[:10]), self.assertRaises(ValueError):
                clean_input(content)

    def test_multiline_is_submitted_once(self) -> None:
        lines = iter(["第一行", "", "第二行", "/send"])
        self.assertEqual(read_user_input(read_line=lambda _: next(lines)), "第一行\n\n第二行")
        self.assertEqual(read_user_input(read_line=lambda _: "/status"), "/status")


class RuntimeRegressionTests(unittest.TestCase):
    """使用真实的 Runtime、模型适配器和 Retail 工具，HTTP 回复为离线。"""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        runtime = RetailRuntime.__new__(RetailRuntime)
        runtime.config = ModelConfig()
        runtime.environment = get_environment()
        runtime.guard = RetailGuard(runtime.environment)
        runtime.model = create_model(runtime.config)
        runtime.output = Path(self.folder.name)
        (runtime.output / "provider").mkdir()
        runtime.tools = [tool for tool in runtime.environment.get_tools()
                         if tool.name != "get_item_details"]
        runtime.tool_names = {tool.name for tool in runtime.tools}
        runtime.provider_history = []
        runtime.seen_call_ids = set()
        runtime.max_model_calls = 80
        runtime.max_total_tokens = 80_000
        runtime.repair_limit = 3  # 显式检验低预算分支，不依赖正式默认值。
        runtime.system_prompt = "Offline regression contract, not real inference."
        runtime.state = SessionState("fixture")
        self.runtime = runtime
        self.serial = 0

    def reply(self, proposals: list[ToolCall | str | dict]) -> None:
        """在真实传输层上安装线上格式的 HTTP 响应 fixture。"""
        streams = []
        for proposal in proposals:
            if isinstance(proposal, dict):
                raw = proposal
            else:
                message = {"role": "assistant", "content": None}
                if isinstance(proposal, str):
                    message["content"] = proposal
                else:
                    message["tool_calls"] = [{
                        "type": "function", "id": proposal.id,
                        "function": {"name": proposal.name,
                                     "arguments": json.dumps(proposal.arguments)},
                    }]
                raw = {"choices": [{"finish_reason": "stop", "message": message}],
                       "usage": {"total_tokens": 10}}
            stream = Mock()
            stream.__enter__ = Mock(return_value=BytesIO(json.dumps(raw).encode()))
            stream.__exit__ = Mock(return_value=False)
            streams.append(stream)
        self.runtime.model.opener.open = Mock(side_effect=streams)

    def execute_read(self, name: str, arguments: dict) -> None:
        self.serial += 1
        self.reply([call(name, arguments, str(self.serial))])
        result = self.runtime.execute(self.runtime.decide())
        self.assertFalse(result.error, result.content)

    def authenticate_and_read(self) -> None:
        self.runtime.receive("Yusuf Rossi 19122。" + REQUEST)
        self.execute_read("find_user_id_by_name_zip", IDENTITY)
        self.execute_read("get_user_details", {"user_id": USER})
        self.execute_read("get_order_details", {"order_id": ORDER})
        for product in ("1656367028", "4896585277"):
            self.execute_read("get_product_details", {"product_id": product})

    def test_rejected_proposal_feedback_and_three_repair_limit(self) -> None:
        runtime = self.runtime
        runtime.receive("我想换个键盘")
        before = runtime.environment.get_db_hash()
        self.reply([call("find_user_id_by_email", {"email": 123}, f"bad{i}") for i in range(3)])
        message = runtime.decide()
        self.assertFalse(message.tool_calls)
        self.assertEqual(runtime.state.blocked_proposals, 3)
        self.assertEqual(runtime.state.status, "waiting_user")
        self.assertEqual(runtime.state.tool_calls, 0)
        self.assertEqual(runtime.environment.get_db_hash(), before)
        self.assertFalse(any(item.tool_calls for item in runtime.state.messages
                             if item.role == "assistant"))
        feedback = [item for item in runtime.provider_history if item["role"] == "tool"]
        self.assertEqual([item["tool_call_id"] for item in feedback], ["bad0", "bad1", "bad2"])
        self.assertIn("ARGUMENT_SCHEMA", feedback[0]["content"])

    def test_protocol_repair_counts_usage_and_keeps_raw_response(self) -> None:
        self.runtime.receive("我需要换货")
        raw = {"choices": [{"finish_reason": "length", "message": {
            "role": "assistant", "content": "truncated",
        }}], "usage": {"total_tokens": 10}}
        self.reply([raw, "请提供邮箱或姓名与邮编。"])
        answer = self.runtime.decide()
        self.assertIn("邮编", answer.content)
        self.assertEqual(self.runtime.state.model_calls, 2)
        self.assertEqual(self.runtime.state.total_tokens, 20)
        self.assertEqual(self.runtime.state.blocked_proposals, 1)
        self.assertEqual(json.loads((self.runtime.output / "provider/001.response.json").read_text()), raw)

    def test_recovery_can_continue_after_more_than_three_rejections(self) -> None:
        """较大的修正预算允许第四次失败后继续到合法回复，而非机械暂停。"""
        runtime = self.runtime
        runtime.repair_limit = 10
        runtime.receive("我想换货")
        self.reply([call("find_user_id_by_email", {"email": 123}, f"retry{i}") for i in range(4)]
                   + ["请提供您的邮箱，或姓名与邮编。"])
        message = runtime.decide()
        self.assertEqual(runtime.state.model_calls, 5)
        self.assertEqual(runtime.state.blocked_proposals, 4)
        self.assertEqual(runtime.state.tool_calls, 0)
        self.assertIn("邮编", message.content)
        self.assertNotIn("已暂停", message.content)

    def test_successful_tool_resets_consecutive_repair_count(self) -> None:
        """成功查询后，较早的拒绝不再占用下一步骤的修复额度。"""
        runtime = self.runtime
        runtime.receive("Yusuf Rossi 19122。" + REQUEST)
        self.reply([
            call("get_order_details", {"order_id": ORDER}, "rejected-before-auth"),
            call("find_user_id_by_name_zip", IDENTITY, "accepted-auth"),
        ])
        runtime.execute(runtime.decide())
        self.assertEqual(runtime.state.blocked_proposals, 1)
        self.assertEqual(runtime.state.turn_repairs, 0)
        self.reply([
            call("not_a_tool", {}, "rejected-after-read-1"),
            call("not_a_tool", {}, "rejected-after-read-2"),
            "已确认您的身份，请告诉我希望更换的规格。",
        ])
        answer = runtime.decide()
        self.assertIn("已确认您的身份", answer.content)
        self.assertNotIn("已暂停", answer.content)
        self.assertEqual(runtime.state.blocked_proposals, 3)
        self.assertEqual(runtime.state.status, "waiting_user")

    def test_successful_tool_resets_previous_tool_error(self) -> None:
        """真实工具错误与随后成功的工具执行按连续失败计数。"""
        runtime = self.runtime
        runtime.receive("先查 missing@example.invalid，也可以用 Yusuf Rossi 19122 查。")
        self.reply([call(
            "find_user_id_by_email", {"email": "missing@example.invalid"}, "missing"
        )])
        self.assertTrue(runtime.execute(runtime.decide()).error)
        self.assertEqual(runtime.state.turn_repairs, 0)
        self.execute_read("find_user_id_by_name_zip", IDENTITY)
        self.assertEqual(runtime.state.turn_repairs, 0)

    def test_exchange_identity_summary_is_not_write_confirmation(self) -> None:
        """换货上下文中也应允许说明身份并询问商品规格。"""
        self.authenticate_and_read()
        self.reply(["已确认您的身份，请告诉我还需要哪种键盘规格。"])
        answer = self.runtime.decide()
        self.assertIn("已确认您的身份", answer.content)
        self.assertEqual(self.runtime.state.blocked_proposals, 0)

    def test_confirm_write_auto_readback_and_official_evaluator(self) -> None:
        self.authenticate_and_read()
        runtime = self.runtime
        before = runtime.environment.get_db_hash()
        self.reply([call("prepare_exchange", EXCHANGE, "proposal")])
        review = runtime.decide()
        self.assertFalse(review.tool_calls)
        self.assertEqual(runtime.environment.get_db_hash(), before)
        self.assertEqual(runtime.state.status, "waiting_user")
        runtime.receive("确认")
        self.reply([call("exchange_delivered_order_items", EXCHANGE, "confirmed")])
        runtime.execute(runtime.decide())
        self.assertEqual(runtime.state.status, "waiting_user")
        self.assertEqual(runtime.state.messages[-2].role, "tool")
        self.assertTrue(runtime.state.messages[-2].id.startswith("tf_readback_"))
        self.assertIn("退回 $16.63", runtime.state.messages[-1].content)
        runtime.close("user_stop")
        task = next(task for task in get_tasks(None) if task.id == "0")
        reward = EnvironmentEvaluator.calculate_reward(get_environment, task, runtime.state.messages)
        self.assertEqual(reward.reward, 1.0)
        self.assertTrue(reward.db_check.db_match)
        executed = [c for msg in runtime.state.messages if msg.role == "assistant"
                    for c in msg.tool_calls or []]
        self.assertNotIn("proposal", [item.id for item in executed])
        self.assertEqual(sum(item.name == "exchange_delivered_order_items" for item in executed), 1)
        self.assertIn(fingerprint(call("exchange_delivered_order_items", EXCHANGE)), runtime.guard.attempted_writes)

    def test_receive_control_character_never_reaches_model(self) -> None:
        self.runtime.model.opener.open = Mock()
        with self.assertRaises(ValueError):
            self.runtime.receive("\x16")
        self.runtime.model.opener.open.assert_not_called()
        self.assertEqual(self.runtime.state.messages, [])

    def test_identity_not_found_is_normal_clarification_not_repair_failure(self) -> None:
        """错误邮箱只查询一次，模型正常补问，不能被当作三次修正失败。"""
        self.runtime.receive("我记得邮箱是 missing@example.invalid")
        self.reply([call("find_user_id_by_email", {"email": "missing@example.invalid"}, "missing")])
        result = self.runtime.execute(self.runtime.decide())
        self.assertTrue(result.error)
        self.assertEqual(self.runtime.state.turn_repairs, 0)
        self.assertEqual(self.runtime.state.recovery_action, "ask_user")
        self.reply(["没有找到这个邮箱，请核对邮箱或提供姓名和邮编。"])
        answer = self.runtime.decide()
        self.assertIn("邮编", answer.content)
        self.assertNotIn("已暂停", answer.content)
        self.assertEqual(self.runtime.state.status, "waiting_user")
        self.assertEqual(self.runtime.state.tool_calls, 1)
        request = self.runtime.model.opener.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["tool_choice"], "none")
        trusted = next(m["content"] for m in body["messages"] if (m.get("content") or "").startswith("Trusted runtime state"))
        self.assertIn('"identity_input_from_user": null', trusted)
        self.assertIn('"status": "needs_user_input"', trusted)

    def test_model_ignoring_information_request_cannot_repeat_failed_lookup(self) -> None:
        """模型不服从补问提示时，程序直接补问，不再执行已失败查询。"""
        runtime = self.runtime
        runtime.receive("missing@example.invalid")
        self.reply([call("find_user_id_by_email", {"email": "missing@example.invalid"}, "first")])
        runtime.execute(runtime.decide())
        self.reply([call("find_user_id_by_email", {"email": "missing@example.invalid"}, "repeat")])
        answer = runtime.decide()
        self.assertIn("邮政编码", answer.content)
        self.assertEqual(runtime.state.status, "waiting_user")
        self.assertEqual(runtime.state.tool_calls, 1)
        self.assertEqual(runtime.state.turn_repairs, 0)
        self.assertFalse(any(m.role == "assistant" and m.tool_calls and m.tool_calls[0].id == "repeat" for m in runtime.state.messages))

    def test_malformed_clarification_uses_program_fallback_without_retry_loop(self) -> None:
        """已确定缺用户信息时，模型补问输出截断也不反复调用模型。"""
        runtime = self.runtime
        runtime.receive("missing@example.invalid")
        self.reply([call("find_user_id_by_email", {"email": "missing@example.invalid"}, "missing")])
        runtime.execute(runtime.decide())
        self.reply([{"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "请核对"}}], "usage": {"total_tokens": 10}}])
        answer = runtime.decide()
        self.assertIn("邮政编码", answer.content)
        self.assertEqual(runtime.state.model_calls, 2)
        self.assertEqual(runtime.state.turn_repairs, 0)
        self.assertEqual(runtime.state.status, "waiting_user")

    def test_new_user_information_can_recover_after_identity_not_found(self) -> None:
        """用户补充真实姓名和邮编后，重新开放查询并核实身份。"""
        runtime = self.runtime
        runtime.receive("missing@example.invalid")
        self.reply([call("find_user_id_by_email", {"email": "missing@example.invalid"}, "missing")])
        runtime.execute(runtime.decide())
        self.reply(["请核对邮箱，或提供姓名和邮编。"])
        runtime.decide()
        runtime.receive("我叫 Yusuf Rossi，邮编19122。")
        self.execute_read("find_user_id_by_name_zip", IDENTITY)
        self.assertEqual(runtime.guard.user_id, USER)
        self.assertEqual(runtime.guard.identity_summary()["status"], "authenticated")
        self.assertEqual(runtime.state.turn_repairs, 0)

    def test_extended_environment_uses_same_identity_recovery(self) -> None:
        """扩展业务同样只查一次错误邮箱，正常返回补问而不是暂停。"""
        from toolflow.commerce import CommerceEnvironment, CommerceGuard
        runtime = self.runtime
        runtime.environment = CommerceEnvironment(None)
        runtime.guard = CommerceGuard(runtime.environment)
        runtime.retail_mode = "extended"
        runtime.tools = [tool for tool in runtime.environment.get_tools() if tool.name != "get_item_details"]
        runtime.tool_names = {tool.name for tool in runtime.tools}
        before = runtime.environment.get_db_hash()
        runtime.receive("我的邮箱是 missing@example.invalid")
        self.reply([call("find_user_id_by_email", {"email": "missing@example.invalid"}, "extended-missing")])
        runtime.execute(runtime.decide())
        self.reply(["没查到该邮箱，请提供正确邮箱或姓名和邮编。"])
        answer = runtime.decide()
        self.assertEqual(runtime.state.status, "waiting_user")
        self.assertEqual(runtime.state.tool_calls, 1)
        self.assertEqual(runtime.state.turn_repairs, 0)
        self.assertEqual(before, runtime.environment.get_db_hash())
        self.assertNotIn("已暂停", answer.content)

    def test_missing_identity_is_requested_without_repair_loop(self) -> None:
        """完全没提供身份时，越过认证的工具提案变成正常补问，不反复重试。"""
        runtime = self.runtime
        runtime.receive("我想查询订单")
        self.reply([call("get_order_details", {"order_id": ORDER}, "premature")])
        answer = runtime.decide()
        self.assertIn("邮箱", answer.content)
        self.assertEqual(runtime.state.tool_calls, 0)
        self.assertEqual(runtime.state.model_calls, 1)
        self.assertEqual(runtime.state.turn_repairs, 0)
        self.assertEqual(runtime.state.status, "waiting_user")

    def confirmed_write(self) -> object:
        """通过只读的准备工具到达一个已批准的真实写入。"""
        self.authenticate_and_read()
        self.reply([call("prepare_exchange", EXCHANGE, "proposal")])
        self.runtime.decide()
        self.runtime.receive("确认")
        self.reply([call("exchange_delivered_order_items", EXCHANGE, "confirmed")])
        return self.runtime.decide()

    def test_error_after_mutation_stops_without_retry(self) -> None:
        message = self.confirmed_write()
        original = self.runtime.environment.get_response

        def uncertain(proposal: ToolCall) -> ToolMessage:
            original(proposal)
            return ToolMessage(role="tool", id=proposal.id, error=True,
                               content="Error: simulated response loss after write")

        self.runtime.environment.get_response = Mock(side_effect=uncertain)
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            self.runtime.execute(message)
        self.runtime.environment.get_response.assert_called_once()
        self.assertIsNone(self.runtime.guard.confirmed)
        self.assertEqual(self.runtime.environment.tools.db.orders[ORDER].status, "exchange requested")

    def test_failed_readback_never_emits_success_receipt(self) -> None:
        message = self.confirmed_write()
        original = self.runtime.environment.get_response

        def broken_read(proposal: ToolCall) -> ToolMessage:
            if proposal.name == "exchange_delivered_order_items":
                return original(proposal)
            return ToolMessage(role="tool", id=proposal.id, error=True,
                               content="Error: simulated read failure")

        self.runtime.environment.get_response = Mock(side_effect=broken_read)
        with self.assertRaisesRegex(RuntimeError, "readback mismatch"):
            self.runtime.execute(message)
        self.assertEqual(self.runtime.environment.get_response.call_count, 2)
        self.assertEqual(self.runtime.state.messages[-1].role, "tool")


if __name__ == "__main__":
    unittest.main()
