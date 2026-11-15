"""ToolFlow 注册/购买扩展，独立于基准测试。"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator, List
import uuid

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.domains.retail.data_model import (
    CreditCard,
    GiftCard,
    Order,
    OrderItem,
    OrderPayment,
    RetailDB,
    User,
    UserAddress,
    UserName,
)
from tau2.domains.retail.environment import get_environment
from tau2.domains.retail.tools import RetailTools
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType, is_tool

from .validation import (
    GuardError,
    Proposal,
    RetailGuard,
    requests_write_confirmation,
)

EXTENSION_POLICY = """
ToolFlow extended Retail policy (simulation):
Public product browsing and account registration do not require authentication.
Registration requires the user's name, email and complete shipping address.
Use prepare_registration to obtain a runtime review, then register_account only
after explicit confirmation. Registration creates a test credit-card payment
method automatically; do not ask for passwords, bank card numbers or API keys.
A successfully registered new customer becomes this session's authenticated user.
Purchasing requires authentication, an observed customer profile and catalog
variants. Use prepare_purchase for the grounded quote. After explicit confirmation
call place_order with the exact approved arguments. The shipping address is the
profile's current default address. Repeated item IDs represent quantity.
Prices are USD. This is simulated payment; no real charge or fulfillment occurs.
Catalog availability is a boolean, not a numeric stock count. Check availability
but do not claim stock quantities or decrement an invented inventory count.
New orders have pending status and can use existing official pending-order tools.
Other account/order operations follow the official rules below. Account creation
and purchases are additional supported actions, not reasons to transfer to human.
""".strip()


def cart_records(db: RetailDB, item_ids: List[str]) -> tuple[list[OrderItem], float]:
    """解析可用的目录变体并计算精确到分的总额。

    Parameters
    ----------
    db : RetailDB
        当前事务快照。
    item_ids : List[str]
        每件商品一个 ID，重复 ID 表示数量；最多 100 件。

    Returns
    -------
    tuple[list[OrderItem], float]
        基于目录的订单行及美元总额。
    """
    if not item_ids or len(item_ids) > 100:
        raise ValueError("A purchase must contain 1–100 units")
    index = {
        item_id: (product, variant)
        for product in db.products.values()
        for item_id, variant in product.variants.items()
    }
    lines: list[OrderItem] = []
    total = Decimal("0")
    for item_id in item_ids:
        if item_id not in index:
            raise ValueError(f"Unknown catalog item: {item_id}")
        product, variant = index[item_id]
        if not variant.available:
            raise ValueError(f"Item unavailable: {item_id}")
        lines.append(
            OrderItem(
                name=product.name,
                product_id=product.product_id,
                item_id=item_id,
                price=variant.price,
                options=dict(variant.options),
            )
        )
        total += Decimal(str(variant.price))
    return lines, float(total.quantize(Decimal("0.01")))


class CommerceTools(RetailTools):
    """复用官方工具，并恰好新增两个模拟写操作。"""

    @is_tool(ToolType.WRITE)
    def register_account(
        self,
        first_name: str,
        last_name: str,
        email: str,
        address1: str,
        city: str,
        state: str,
        country: str,
        zip: str,
        address2: str = "",
    ) -> User:
        """在运行时核对并确认后创建模拟客户。

        Parameters
        ----------
        first_name : str
            用户提供的名字；绝不使用占位符。
        last_name : str
            用户提供的姓氏。
        email : str
            用户提供的唯一邮箱，用于后续认证。
        address1 : str
            用户提供的收货街道地址。
        city : str
            收货城市。
        state : str
            收货州或省份。
        country : str
            用户提供的收货国家。
        zip : str
            收货邮政编码。
        address2 : str
            可选的公寓或套房号，省略时为空。

        Returns
        -------
        User
            创建的客户，带有生成的 ID 和模拟测试卡。

        Raises
        ------
        ValueError
            必填字段为空、邮箱无效或已存在时抛出。
        """
        fields = (first_name, last_name, email, address1, city, state, country, zip)
        if any(not field.strip() for field in fields):
            raise ValueError("Name, email and complete shipping address are required")
        email = email.strip().casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            raise ValueError("Invalid email address")
        if any(user.email.casefold() == email for user in self.db.users.values()):
            raise ValueError("Email already registered; authenticate instead")
        user_id = "tf_customer_" + uuid.uuid4().hex
        payment_id = "tf_test_card_" + uuid.uuid4().hex
        user = User(
            user_id=user_id,
            name=UserName(first_name=first_name, last_name=last_name),
            email=email,
            address=UserAddress(
                address1=address1,
                address2=address2,
                city=city,
                state=state,
                country=country,
                zip=zip,
            ),
            payment_methods={
                payment_id: CreditCard(
                    id=payment_id,
                    source="credit_card",
                    brand="ToolFlow simulation",
                    last_four="0000",
                )
            },
            orders=[],
        )
        self.db.users[user_id] = user
        return user

    @is_tool(ToolType.WRITE)
    def place_order(
        self,
        user_id: str,
        item_ids: List[str],
        payment_method_id: str,
    ) -> Order:
        """在已批准的购买报价之后创建待处理的模拟订单。

        Parameters
        ----------
        user_id : str
            由工具返回的已认证客户 ID。
        item_ids : List[str]
            已观察到的可用变体 ID，每件一个，最多 100 件。
        payment_method_id : str
            来自客户已观察档案的支付方式 ID。

        Returns
        -------
        Order
            带有目录价格和支付记录的新待处理订单。

        Raises
        ------
        ValueError
            商品、地址、支付方式或礼品卡余额无效时抛出。
        """
        user = self._get_user(user_id)
        if any(
            not getattr(user.address, key).strip()
            for key in ("address1", "city", "state", "country", "zip")
        ):
            raise ValueError("A complete default shipping address is required")
        lines, total = cart_records(self.db, item_ids)
        payment = self._get_payment_method(user_id, payment_method_id)
        if isinstance(payment, GiftCard) and payment.balance < total:
            raise ValueError("Insufficient gift card balance")
        order_id = "#TF" + uuid.uuid4().hex
        order = Order(
            order_id=order_id,
            user_id=user_id,
            address=user.address.model_copy(deep=True),
            items=lines,
            status="pending",
            fulfillments=[],
            payment_history=[
                OrderPayment(
                    transaction_type="payment",
                    amount=total,
                    payment_method_id=payment_method_id,
                )
            ],
        )
        # 在首次修改之前完成所有校验；存储层也会回滚。
        if isinstance(payment, GiftCard):
            payment.balance = round(payment.balance - total, 2)
        self.db.orders[order_id] = order
        user.orders.append(order_id)
        return order


class CommerceEnvironment(Environment):
    """串行化本地事务并持久化仅限扩展的 Retail 快照。

    Parameters
    ----------
    store_path : Path | None
        独立的 SQLite 存储，或为 None 表示隔离的内存测试会话。
    """

    def __init__(self, store_path: Path | None = None) -> None:
        official = get_environment()
        policy = official.get_policy().replace(
            "At the beginning of the conversation, you have to authenticate the user identity",
            "Before accessing an existing customer's private data, authenticate the user identity",
        )
        super().__init__(
            "retail_extended",
            EXTENSION_POLICY + "\n\n" + policy,
            CommerceTools(official.tools.db),
        )
        self.store_path = store_path.resolve() if store_path is not None else None
        self.scope = uuid.uuid4().hex
        self.expected_write_hash: str | None = None
        if store_path is not None:
            store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS retail_state (id INTEGER PRIMARY KEY CHECK(id=1), db_json TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS tool_receipts (key TEXT PRIMARY KEY, args_hash TEXT NOT NULL, response TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT OR IGNORE INTO retail_state VALUES (1, ?)",
                    (self.tools.db.model_dump_json(),),
                )
            self.refresh()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """打开一个带有限等待时间的本地连接；不使用网络数据库。"""
        connection = sqlite3.connect(self.store_path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def refresh(self) -> None:
        """在检查确认时效性之前刷新可信执行器状态。"""
        if self.store_path is not None:
            with self.connect() as connection:
                payload = connection.execute(
                    "SELECT db_json FROM retail_state WHERE id=1"
                ).fetchone()[0]
            self.tools.db = RetailDB.model_validate_json(payload)

    def get_response(self, message: ToolCall) -> ToolMessage:
        """原子化执行并持久化；重复的调用 ID 返回已保存的回执。"""
        if self.store_path is None:
            return super().get_response(message)
        key = f"{self.scope}:{message.id}"
        digest = hashlib.sha256(
            json.dumps(
                [message.name, message.arguments],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        write = self.tools.tool_type(message.name) == ToolType.WRITE
        expected_hash = self.expected_write_hash if write else None
        if write:
            self.expected_write_hash = None
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            payload = connection.execute(
                "SELECT db_json FROM retail_state WHERE id=1"
            ).fetchone()[0]
            self.tools.db = RetailDB.model_validate_json(payload)
            if write:
                prior = connection.execute(
                    "SELECT args_hash, response FROM tool_receipts WHERE key=?",
                    (key,),
                ).fetchone()
                if prior:
                    if prior[0] != digest:
                        raise ValueError(
                            "Idempotency key reused with different arguments"
                        )
                    return ToolMessage.model_validate_json(prior[1])
                if expected_hash is not None and self.get_db_hash() != expected_hash:
                    return ToolMessage(
                        role="tool",
                        id=message.id,
                        error=True,
                        content="Error: confirmed state changed before commit; review again",
                    )
            try:
                response = super().get_response(message)
                if response.error:
                    self.tools.db = RetailDB.model_validate_json(payload)
                    return response
                if write:
                    RetailDB.model_validate_json(self.tools.db.model_dump_json())
                    connection.execute(
                        "UPDATE retail_state SET db_json=? WHERE id=1",
                        (self.tools.db.model_dump_json(),),
                    )
                    connection.execute(
                        "INSERT INTO tool_receipts VALUES (?, ?, ?)",
                        (key, digest, response.model_dump_json()),
                    )
                return response
            except Exception:
                self.tools.db = RetailDB.model_validate_json(payload)
                raise


class CommerceGuard(RetailGuard):
    """在官方防护措施之上增加注册/公开浏览/购买守卫。"""

    def check(self, call: ToolCall) -> Proposal | None:
        """校验新操作，且不削弱私有数据归属保护。"""
        self.validate_arguments(call)
        if call.name == "register_account":
            if self.user_id:
                raise GuardError(
                    "IDENTITY_LOCKED",
                    "Use /new before registering a different customer.",
                )
            for key, value in call.arguments.items():
                if value and not self.mentioned(value):
                    raise GuardError(
                        "UNGROUNDED_REGISTRATION",
                        f"Ask for {key}; do not invent customer information.",
                    )
            required = (
                "first_name",
                "last_name",
                "email",
                "address1",
                "city",
                "state",
                "country",
                "zip",
            )
            if any(not call.arguments[key].strip() for key in required):
                raise GuardError(
                    "REGISTRATION_FIELDS",
                    "Collect a name, valid email and complete shipping address before previewing registration.",
                )
            if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", call.arguments["email"]):
                raise GuardError(
                    "REGISTRATION_FIELDS", "Ask for a valid email address."
                )
            # 官方 Pydantic 类型构造和邮箱唯一性会在数据库事务内
            # 由写工具再次检查。
            if any(
                user.email.casefold() == call.arguments["email"].strip().casefold()
                for user in self.environment.tools.db.users.values()
            ):
                raise GuardError(
                    "EMAIL_EXISTS",
                    "Email already registered. Authenticate by email instead.",
                )
            return self.review(
                call,
                [
                    "注册模拟客户账号，收货地址如上；系统自动生成测试支付方式。",
                    "不会创建真实网站登录，也不会收集密码或真实银行卡号。",
                ],
            )
        if call.name == "list_all_product_types":
            return None
        if call.name == "get_product_details" and not self.user_id:
            if call.arguments["product_id"] not in self.catalog_ids:
                raise GuardError("UNKNOWN_PRODUCT", "Use list_all_product_types first.")
            return None
        if call.name == "place_order":
            if not self.user_id or call.arguments["user_id"] != self.user_id:
                raise GuardError(
                    "AUTH_REQUIRED", "Authenticate or register before purchasing."
                )
            if not self.profile:
                raise GuardError(
                    "PROFILE_REQUIRED", "Read get_user_details before purchase."
                )
            observed = {
                item_id: (product, variant)
                for product in self.products.values()
                for item_id, variant in product["variants"].items()
            }
            for item_id in call.arguments["item_ids"]:
                if item_id not in observed:
                    raise GuardError(
                        "CATALOG_REQUIRED",
                        "Choose item IDs from get_product_details, not examples.",
                    )
            payment_id = call.arguments["payment_method_id"]
            if payment_id not in self.profile["payment_methods"]:
                raise GuardError(
                    "PAYMENT_NOT_OBSERVED",
                    "Choose a payment method from the observed profile.",
                )
            current_user = self.environment.tools.db.users[self.user_id]
            if current_user.address.model_dump() != self.profile["address"]:
                raise GuardError(
                    "PROFILE_STALE",
                    "Shipping address changed; read get_user_details again.",
                )
            current_variants = {
                item_id: variant.model_dump()
                for product in self.environment.tools.db.products.values()
                for item_id, variant in product.variants.items()
            }
            if any(
                observed[item_id][1] != current_variants.get(item_id)
                for item_id in call.arguments["item_ids"]
            ):
                raise GuardError(
                    "CATALOG_STALE",
                    "Catalog changed; read get_product_details again before confirming.",
                )
            try:
                lines, total = cart_records(
                    self.environment.tools.db, call.arguments["item_ids"]
                )
            except ValueError as error:
                raise GuardError("PURCHASE_ITEMS", str(error)) from error
            details = []
            for item_id, quantity in Counter(call.arguments["item_ids"]).items():
                product, variant = observed[item_id]
                for option, value in variant["options"].items():
                    requested = {
                        candidate["options"][option]
                        for candidate in product["variants"].values()
                        if option in candidate["options"]
                        and self.option_requested(candidate["options"][option])
                    }
                    if requested and value not in requested:
                        raise GuardError(
                            "OPTION_MISMATCH",
                            f"Selected {option}={value} conflicts with user-provided options {sorted(requested)}.",
                        )
                details.append(
                    f"{product['name']} {item_id} {variant['options']} × {quantity}；单价 ${variant['price']:.2f}"
                )
                existing = {
                    order.order_id: sum(item.item_id == item_id for item in order.items)
                    for order in self.environment.tools.db.orders.values()
                    if order.user_id == self.user_id and order.status == "pending"
                    and any(item.item_id == item_id for item in order.items)
                }
                previous = sum(existing.values())
                details.append(
                    f"同款数量核对：已有待处理订单 {previous} 件，本次新增 {quantity} 件，"
                    f"提交后待处理合计 {previous + quantity} 件。"
                )
                if existing:
                    details.append("已有订单不会自动取消或合并：" + json.dumps(existing, ensure_ascii=False))
            payment = self.environment.tools.db.users[self.user_id].payment_methods[
                payment_id
            ]
            if isinstance(payment, GiftCard) and payment.balance < total:
                raise GuardError(
                    "INSUFFICIENT_BALANCE", "Gift card cannot cover the purchase total."
                )
            details.extend(
                [
                    f"总价：USD ${total:.2f}（{len(lines)} 件）。",
                    "收货地址："
                    + json.dumps(self.profile["address"], ensure_ascii=False),
                    f"支付方式：{payment_id}；仅模拟支付，不实际扣款。",
                ]
            )
            return self.review(call, details)
        return self.check_business(call)

    def check_text(self, content: str) -> None:
        """允许方案解释和确认提问；真正写入仍须通过精确参数确认。"""
        if not self.user_id:
            if re.search(r"请稍等|please wait", content.casefold()):
                raise GuardError(
                    "TOOL_REQUIRED_FOR_PROGRESS",
                    "Use a native tool or ask a concrete question.",
                )
            return
        super().check_text(content)

    def observe(self, call: ToolCall, response: ToolMessage) -> None:
        """仅根据成功创建的客户记录建立会话。"""
        if call.name == "register_account" and not response.error:
            self.profile = json.loads(response.content)
            self.user_id = self.profile["user_id"]
        else:
            super().observe(call, response)
