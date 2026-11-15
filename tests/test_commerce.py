"""针对真实工具和隔离 SQLite 存储的注册/购买契约测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_p1

run_p1.bootstrap()

from tau2.data_model.message import ToolCall
from tau2.domains.retail.environment import get_environment
from tau2.domains.retail.data_model import GiftCard
from toolflow.commerce import CommerceEnvironment, CommerceGuard
from toolflow.model import ModelConfig, create_model
from toolflow.runtime import RetailRuntime, SessionState
from toolflow.validation import GuardError

REGISTRATION = {
    "first_name": "Flow",
    "last_name": "Buyer",
    "email": "flow.buyer@example.invalid",
    "address1": "12 Demo Street",
    "address2": "",
    "city": "Boston",
    "state": "MA",
    "country": "USA",
    "zip": "02108",
}
PRODUCT = "1656367028"
ITEM = "7706410293"


def proposal(name: str, args: dict, identity: str = "contract") -> ToolCall:
    """创建显式的协议测试输入，绝不是替代 LLM。"""
    return ToolCall(id=identity, name=name, arguments=args)


class CommerceFixture(unittest.TestCase):
    """让所有测试写入都落在用户持久化数据库之外。"""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.environment = CommerceEnvironment(self.path / "retail.sqlite3")

    def register(self, identity: str = "register", email: str | None = None) -> dict:
        """使用真实的事务性写入来准备后端测试状态。"""
        args = dict(REGISTRATION)
        if email:
            args["email"] = email
        response = self.environment.get_response(
            proposal("register_account", args, identity)
        )
        self.assertFalse(response.error, response.content)
        return json.loads(response.content)

    def cart(self, user: dict) -> dict:
        """使用已注册客户返回的 ID 和观察到的目录。"""
        return {
            "user_id": user["user_id"],
            "item_ids": [ITEM, ITEM],
            "payment_method_id": next(iter(user["payment_methods"])),
        }

    def guard_for(self, user: dict) -> CommerceGuard:
        """在购买前观察真实的用户资料、列表和商品结果。"""
        guard = CommerceGuard(self.environment)
        guard.user_id = user["user_id"]
        for name, args in (
            ("get_user_details", {"user_id": user["user_id"]}),
            ("list_all_product_types", {}),
            ("get_product_details", {"product_id": PRODUCT}),
        ):
            call = proposal(name, args, name)
            self.assertIsNone(guard.check(call))
            guard.observe(call, self.environment.get_response(call))
        guard.receive("买两个全尺寸 clicky 无背光键盘，用当前地址和测试卡。")
        return guard


class CommerceTests(CommerceFixture):
    """在无模型的情况下检查事务、持久化和确认规则。"""

    def test_identity_and_variant_questions_allow_normal_replies(self) -> None:
        """已确认身份和询问规格时允许正常结束本轮回复。"""
        guard = self.guard_for(self.register())
        replies = (
            "已确认您的身份。目前有货的键盘如下，请告诉我轴体、背光、尺寸和数量。",
            "已完成身份确认。请告诉我您想要哪种规格，确认后我会生成订单报价供您核对。",
            "请确认所需的轴体、背光和尺寸。",
            "I confirmed your identity. Which keyboard size would you like?",
        )
        for reply in replies:
            with self.subTest(reply=reply):
                guard.check_text(reply)

    def test_free_text_purchase_approval_still_requires_preview(self) -> None:
        """文本确认可以正常表达，但不能替代真正写入所需的程序确认单。"""
        user = self.register()
        replies = (
            "请确认购买。",
            "请您确认下单。",
            "总价 USD 538.32，请回复“确认”后提交。",
            "Please confirm this purchase.",
            "Confirm the order.",
        )
        for reply in replies:
            with self.subTest(reply=reply):
                guard = self.guard_for(user)
                guard.check_text(reply)
                guard.receive("确认")
                self.assertIsNotNone(guard.check(proposal("place_order", self.cart(guard.profile))))

    def test_purchase_history_does_not_block_cancellation_clarification(self) -> None:
        """购买之后的取消原因提问与确认，不再被误判为购买报价。"""
        guard = self.guard_for(self.register())
        guard.receive("一共买两件，取消多出来的一件。")
        for reply in (
            "保留两件那单，取消一件那单。请确认取消原因是下错了单还是不再需要？",
            "请回复确认是否取消最早那单，另一个订单不变。",
            "Please confirm which order to cancel and provide the reason.",
        ):
            with self.subTest(reply=reply):
                guard.check_text(reply)
        self.assertIsNone(guard.confirmed)

    def test_purchase_review_shows_existing_added_and_resulting_quantities(self) -> None:
        """已有一件时，新购两件的核对单必须明确最终会是三件。"""
        user = self.register()
        first = self.environment.get_response(proposal("place_order", self.cart(user) | {"item_ids": [ITEM]}, "existing-one"))
        self.assertFalse(first.error)
        guard = self.guard_for(user)
        review = guard.check(proposal("place_order", self.cart(user), "new-two"))
        self.assertIn("已有待处理订单 1 件，本次新增 2 件，提交后待处理合计 3 件", review.review)
        self.assertIn("不会自动取消或合并", review.review)
        self.assertEqual(len(self.environment.tools.db.users[user["user_id"]].orders), 1)

    def test_missing_registration_fields_can_be_clarified(self) -> None:
        """已提供邮箱后仍可询问注册所需的具体地址字段。"""
        guard = CommerceGuard(self.environment)
        guard.receive("我要注册，邮箱 flow.buyer@example.invalid。")
        guard.check_text("请确认您的城市和国家。")

    def test_registration_persists_and_duplicate_email_is_atomic(self) -> None:
        user = self.register()
        self.assertTrue(user["user_id"].startswith("tf_customer_"))
        self.assertEqual(len(user["payment_methods"]), 1)
        reopened = CommerceEnvironment(self.environment.store_path)
        response = reopened.get_response(
            proposal(
                "find_user_id_by_email",
                {"email": REGISTRATION["email"]},
                "lookup",
            )
        )
        self.assertEqual(response.content, user["user_id"])
        before = reopened.get_db_hash()
        args = dict(REGISTRATION, email=REGISTRATION["email"].upper())
        failed = reopened.get_response(proposal("register_account", args, "duplicate"))
        self.assertTrue(failed.error)
        self.assertEqual(before, reopened.get_db_hash())

    def test_registration_needs_real_fields_and_current_confirmation(self) -> None:
        guard = CommerceGuard(self.environment)
        guard.receive("我要注册")
        call = proposal("register_account", REGISTRATION)
        with self.assertRaises(GuardError) as caught:
            guard.check(call)
        self.assertEqual(caught.exception.code, "UNGROUNDED_REGISTRATION")
        guard.receive(json.dumps(REGISTRATION))
        before = self.environment.get_db_hash()
        review = guard.check(call)
        self.assertIsNotNone(review)
        self.assertIn("测试支付", review.review)
        self.assertEqual(before, self.environment.get_db_hash())
        guard.receive("确认")
        self.assertIsNone(guard.check(call))

    def test_purchase_total_quantity_address_and_reopen(self) -> None:
        user = self.register()
        args = self.cart(user)
        guard = self.guard_for(user)
        before = len(self.environment.tools.db.orders)
        review = guard.check(proposal("place_order", args))
        self.assertIn("$538.32", review.review)
        self.assertIn("12 Demo Street", review.review)
        self.assertEqual(before, len(self.environment.tools.db.orders))
        guard.receive("确认")
        self.assertIsNone(guard.check(proposal("place_order", args)))
        response = self.environment.get_response(proposal("place_order", args))
        self.assertFalse(response.error, response.content)
        order = json.loads(response.content)
        self.assertEqual(order["status"], "pending")
        self.assertEqual(len(order["items"]), 2)
        self.assertEqual(order["payment_history"][0]["amount"], 538.32)
        reopened = CommerceEnvironment(self.environment.store_path)
        self.assertIn(
            order["order_id"], reopened.tools.db.users[user["user_id"]].orders
        )
        self.assertEqual(
            reopened.tools.db.orders[order["order_id"]].address.address1,
            "12 Demo Street",
        )
        cancel = reopened.get_response(
            proposal(
                "cancel_pending_order",
                {
                    "order_id": order["order_id"],
                    "reason": "no longer needed",
                },
                "cancel",
            )
        )
        self.assertFalse(cancel.error, cancel.content)
        self.assertEqual(json.loads(cancel.content)["status"], "cancelled")

    def test_no_auth_wrong_owner_unknown_item_and_wrong_size(self) -> None:
        user = self.register()
        guard = CommerceGuard(self.environment)
        with self.assertRaises(GuardError) as caught:
            guard.check(proposal("place_order", self.cart(user)))
        self.assertEqual(caught.exception.code, "AUTH_REQUIRED")
        guard = self.guard_for(user)
        for args, code in (
            (dict(self.cart(user), user_id="someone_else"), "AUTH_REQUIRED"),
            (dict(self.cart(user), item_ids=["0000000000"]), "CATALOG_REQUIRED"),
            (dict(self.cart(user), item_ids=["9665000388"]), "OPTION_MISMATCH"),
            (dict(self.cart(user), item_ids=[]), "PURCHASE_ITEMS"),
            (dict(self.cart(user), item_ids=["9025753381"]), "PURCHASE_ITEMS"),
        ):
            before = self.environment.get_db_hash()
            with self.subTest(code=code), self.assertRaises(GuardError) as caught:
                guard.check(proposal("place_order", args))
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(before, self.environment.get_db_hash())

    def test_call_id_replay_does_not_create_second_order(self) -> None:
        user = self.register()
        call = proposal("place_order", self.cart(user), "one-purchase")
        first = self.environment.get_response(call)
        count = len(self.environment.tools.db.orders)
        second = self.environment.get_response(call)
        self.assertEqual(first, second)
        self.assertEqual(count, len(self.environment.tools.db.orders))
        changed = call.model_copy(
            update={"arguments": dict(self.cart(user), item_ids=[ITEM])}
        )
        with self.assertRaisesRegex(ValueError, "Idempotency"):
            self.environment.get_response(changed)

    def test_partial_failure_rolls_back_database(self) -> None:
        user = self.register()
        before = self.environment.get_db_hash()

        def broken(name: str, **arguments: object) -> None:
            self.environment.tools.db.users[user["user_id"]].orders.append(
                "#not-committed"
            )
            raise ValueError("forced failure after mutation")

        with patch.object(self.environment, "make_tool_call", side_effect=broken):
            result = self.environment.get_response(
                proposal("place_order", self.cart(user))
            )
        self.assertTrue(result.error)
        self.assertEqual(before, self.environment.get_db_hash())
        self.assertEqual(
            before, CommerceEnvironment(self.environment.store_path).get_db_hash()
        )

    def test_concurrent_duplicate_registration_creates_one_customer(self) -> None:
        peer = CommerceEnvironment(self.environment.store_path)
        before = len(peer.tools.db.users)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda env: env.get_response(
                        proposal("register_account", REGISTRATION)
                    ),
                    (self.environment, peer),
                )
            )
        self.assertEqual(sum(not result.error for result in results), 1)
        peer.refresh()
        self.assertEqual(len(peer.tools.db.users), before + 1)

    def test_changed_database_invalidates_confirmation(self) -> None:
        user = self.register()
        guard = self.guard_for(user)
        call = proposal("place_order", self.cart(user))
        guard.check(call)
        guard.receive("确认")
        peer = CommerceEnvironment(self.environment.store_path)
        args = dict(REGISTRATION, email="other@example.invalid")
        peer.get_response(proposal("register_account", args, "other"))
        self.environment.refresh()
        with self.assertRaises(GuardError) as caught:
            guard.check(call)
        self.assertEqual(caught.exception.code, "STALE_CONFIRMATION")

    def test_official_environment_still_has_no_extension_tools(self) -> None:
        tools = {tool.name for tool in get_environment().get_tools()}
        self.assertNotIn("register_account", tools)
        self.assertNotIn("place_order", tools)

    def test_gift_card_debit_once_and_insufficient_balance_rollback(self) -> None:
        user = self.register()
        stored = self.environment.tools.db.users[user["user_id"]]
        stored.payment_methods["test_gift_card"] = GiftCard(
            id="test_gift_card",
            source="gift_card",
            balance=600.0,
        )
        with self.environment.connect() as connection:
            connection.execute(
                "UPDATE retail_state SET db_json=? WHERE id=1",
                (self.environment.tools.db.model_dump_json(),),
            )
        args = dict(self.cart(user), payment_method_id="test_gift_card")
        call = proposal("place_order", args, "gift-purchase")
        first = self.environment.get_response(call)
        self.assertFalse(first.error, first.content)
        self.environment.get_response(call)
        current = self.environment.tools.db.users[user["user_id"]]
        self.assertEqual(current.payment_methods["test_gift_card"].balance, 61.68)
        before = self.environment.get_db_hash()
        failed = self.environment.get_response(
            proposal(
                "place_order",
                dict(args, item_ids=[ITEM]),
                "gift-insufficient",
            )
        )
        self.assertTrue(failed.error)
        self.assertEqual(before, self.environment.get_db_hash())

    def test_transaction_rejects_change_after_confirmation_check(self) -> None:
        user = self.register()
        self.environment.expected_write_hash = self.environment.get_db_hash()
        peer = CommerceEnvironment(self.environment.store_path)
        peer.get_response(
            proposal(
                "register_account",
                dict(
                    REGISTRATION,
                    email="concurrent@example.invalid",
                ),
                "peer-write",
            )
        )
        count = len(peer.tools.db.orders)
        failed = self.environment.get_response(proposal("place_order", self.cart(user)))
        self.assertTrue(failed.error)
        self.assertIn("state changed", failed.content)
        self.assertEqual(count, len(self.environment.tools.db.orders))


class CommerceRuntimeTests(CommerceFixture):
    """针对离线原生 HTTP 调用 fixture 检验真实的 Runtime/适配器。"""

    def test_native_registration_purchase_and_readbacks(self) -> None:
        runtime = RetailRuntime.__new__(RetailRuntime)
        runtime.environment = self.environment
        runtime.guard = CommerceGuard(self.environment)
        runtime.retail_mode = "extended"
        runtime.preparations = {
            "prepare_exchange": "exchange_delivered_order_items",
            "prepare_registration": "register_account",
            "prepare_purchase": "place_order",
        }
        runtime.config = ModelConfig()
        runtime.model = create_model(runtime.config)
        runtime.tools = [
            tool
            for tool in self.environment.get_tools()
            if tool.name != "get_item_details"
        ]
        runtime.tool_names = {tool.name for tool in runtime.tools} | set(
            runtime.preparations
        )
        runtime.output = self.path / "runtime"
        (runtime.output / "provider").mkdir(parents=True)
        runtime.state = SessionState("commerce-contract")
        runtime.provider_history = []
        runtime.seen_call_ids = set()
        runtime.max_model_calls = 30
        runtime.max_total_tokens = 120_000
        runtime.system_prompt = "Offline contract fixture, no real inference."
        number = 0

        def decide(name: str, args: dict) -> object:
            nonlocal number
            number += 1
            raw = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"native{number}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"total_tokens": 10},
            }
            stream = Mock()
            stream.__enter__ = Mock(return_value=BytesIO(json.dumps(raw).encode()))
            stream.__exit__ = Mock(return_value=False)
            runtime.model.opener.open = Mock(return_value=stream)
            return runtime.decide()

        runtime.receive("请注册 " + json.dumps(REGISTRATION))
        review = decide("prepare_registration", REGISTRATION)
        self.assertIsNone(review.tool_calls)
        self.assertIsNone(runtime.guard.user_id)
        runtime.receive("确认")
        runtime.execute(decide("register_account", REGISTRATION))
        self.assertEqual(runtime.state.status, "waiting_user")
        self.assertIn("测试支付方式已创建", runtime.state.messages[-1].content)
        args = self.cart(runtime.guard.profile)
        runtime.receive("买两个全尺寸 clicky 无背光键盘，用当前地址和测试卡。")
        runtime.execute(decide("list_all_product_types", {}))
        runtime.execute(decide("get_product_details", {"product_id": PRODUCT}))
        review = decide("prepare_purchase", args)
        self.assertIn("538.32", review.content)
        runtime.receive("确认")
        runtime.execute(decide("place_order", args))
        self.assertIn("pending", runtime.state.messages[-1].content)
        self.assertIn("$538.32", runtime.state.messages[-1].content)
        events = [
            json.loads(line)
            for line in (runtime.output / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(sum(e["event"] == "readback_finished" for e in events), 2)
        executed = [
            call.name
            for message in runtime.state.messages
            if message.role == "assistant"
            for call in message.tool_calls or []
        ]
        self.assertNotIn("prepare_purchase", executed)
        self.assertNotIn("prepare_registration", executed)

    def test_cli_configuration_exposes_mode_without_connecting(self) -> None:
        for mode in ("official", "extended"):
            with patch(
                "sys.argv", ["run_p1.py", "--check-config", "--retail-mode", mode]
            ):
                output = StringIO()
                with patch("sys.stdout", output):
                    self.assertEqual(run_p1.main(), 0)
                value = json.loads(output.getvalue())
                self.assertEqual(value["retail_mode"], mode)
                self.assertEqual(value["network_requests"], 0)


if __name__ == "__main__":
    unittest.main()
