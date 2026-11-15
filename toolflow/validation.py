"""基于证据的 Retail 守卫，独立于参考任务与答案。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from pydantic import ValidationError
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType


class GuardError(ValueError):
    """可恢复的提案拒绝，绝不是官方工具响应。"""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def requests_write_confirmation(content: str) -> bool:
    """识别请求批准执行的文字，供核对单入口检查使用。

    Parameters
    ----------
    content : str
        模型生成的用户可见回复。

    Returns
    -------
    bool
        是否明确要求批准操作或回复确认词。
    """
    text = re.sub(r"[*_`]", "", content)
    pattern = (
        r"(?:请(?:您)?|是否|能否)\s*确认(?:购买|下单|支付|付款|注册|换货|修改|提交|执行)"
        r"|(?:请(?:您)?|等待(?:您)?|需要(?:您)?)\s*确认\s*[。！？?!]?\s*$"
        r"|(?:回复|输入|点击)\s*[“\"「]?(?:确认|同意)"
        r"|\b(?:please\s+confirm|confirm\s+(?:(?:this|the)\s+)?"
        r"(?:purchase|order|payment|registration))\b"
    )
    return re.search(pattern, text, re.IGNORECASE) is not None


def fingerprint(call: ToolCall) -> str:
    """将确认绑定到完整的工具名称和精确的 JSON 参数。"""
    payload = json.dumps([call.name, call.arguments], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def normalized(text: str) -> str:
    """规范化拼写，但不从助手消息中推导事实。"""
    return re.sub(r"[\s_\-]+", "", text.casefold())


# 针对已观察到的选项值的保守双语别名，绝不用于商品 ID。
OPTION_ALIASES = {
    "clicky": ("青轴", "有声段落"),
    "linear": ("线性", "红轴"),
    "none": ("无背光", "不要背光", "不带背光", "没有背光"),
    "fullsize": ("全尺寸", "全大小"),
    "googleassistant": ("谷歌助手", "google assistant", "google home"),
    "black": ("黑色",),
    "white": ("白色",),
}

PRODUCT_ALIASES = {
    "mechanicalkeyboard": ("键盘", "keyboard"),
    "smartthermostat": ("恒温器", "温控器", "thermostat"),
}


@dataclass
class Proposal:
    """面向用户展示的、针对精确拟议写入及其写入前状态的核对信息。"""

    fingerprint: str
    name: str
    arguments: dict[str, Any]
    review: str
    db_hash: str


class RetailGuard:
    """根据用户输入和成功观测到的工具结果校验提案。

    Parameters
    ----------
    environment : Environment
        每次运行的官方环境。数据库仅用于可信的归属/时效性检查，
        绝不作为模型上下文转储暴露。
    """

    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.definitions = {tool.name: tool for tool in environment.get_tools()}
        self.user_texts: list[str] = []
        self.user_id: str | None = None
        self.profile: dict[str, Any] | None = None
        self.orders: dict[str, dict[str, Any]] = {}
        self.products: dict[str, dict[str, Any]] = {}
        self.catalog_ids: set[str] = set()
        self.pending: Proposal | None = None
        self.confirmed: str | None = None
        self.attempted_writes: set[str] = set()
        self.enhanced_validation = True
        self.failed_identity_inputs: dict[str, int] = {}
        self.identity_needs_input = False

    def receive(self, text: str) -> bool:
        """仅对紧随其前的运行时核对接受“是”。"""
        self.identity_needs_input = False
        answer = text.strip().casefold().rstrip("。.!！")
        pending = self.pending
        accepted = pending is not None and (
            answer in {"yes", "确认", "是，确认", "是,确认"}
            or answer == f"/confirm {pending.fingerprint[:12]}"
        )
        self.confirmed = pending.fingerprint if accepted else None
        if not accepted:
            self.pending = None
            self.user_texts.append(text)
        return accepted

    def mentioned(self, value: str) -> bool:
        """要求真实由用户提供的值，而非模式示例。"""
        return bool(value.strip()) and any(
            normalized(value) in normalized(text) for text in self.user_texts
        )

    def option_requested(self, value: str) -> bool:
        """保守地识别肯定且明确的选项提及。"""
        values = (value,) + OPTION_ALIASES.get(normalized(value), ())
        for text in reversed(self.user_texts):
            compact = normalized(text)
            for alias in values:
                term = normalized(alias)
                for match in re.finditer(re.escape(term), compact):
                    prefix = compact[max(0, match.start() - 8):match.start()]
                    # '不要背光' 是选项 'none' 的肯定别名。
                    if not re.search(r"(?:不要|不选|不是|not|without|no)$", prefix):
                        return True
        return False

    def is_write(self, call: ToolCall) -> bool:
        """使用官方工具元数据，而非模型提供的标签。"""
        return self.environment.tools.tool_type(call.name) == ToolType.WRITE

    @staticmethod
    def _without_emails(text: str) -> str:
        """邮箱不能作为姓名或邮编的字段来源。"""
        return re.sub(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", " ", text)

    def identity_source_turn(self, call: ToolCall) -> int | None:
        """检查身份字段的完整来源，不使用任意子串命中。

        Parameters
        ----------
        call : ToolCall
            拟议的身份查询。

        Returns
        -------
        int | None
            最后提供这些字段的用户轮次；无法对应真实字段时为 None。
        """
        args = call.arguments
        if call.name == "find_user_id_by_email":
            matches = [index for index, text in enumerate(self.user_texts, 1)
                       if args["email"].casefold() in {
                           email.casefold() for email in re.findall(
                               r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)}]
            return max(matches) if matches else None
        if call.name != "find_user_id_by_name_zip":
            return None
        first, last, zip_code = args["first_name"], args["last_name"], args["zip"]
        if not all(re.fullmatch(r"[^\W\d_][\w'’-]*", value) for value in (first, last)):
            return None
        if not re.fullmatch(r"\d{5}", zip_code):
            return None
        name_pattern = r"(?<![A-Za-z])" + re.escape(first) + r"\s+" + re.escape(last) + r"(?![A-Za-z])"
        names, codes = [], []
        for index, text in enumerate(self.user_texts, 1):
            text = self._without_emails(text)
            if re.search(name_pattern, text, re.IGNORECASE):
                names.append(index)
            if re.search(r"(?<!\d)" + re.escape(zip_code) + r"(?!\d)", text):
                codes.append(index)
        return max(max(names), max(codes)) if names and codes else None

    @staticmethod
    def _identity_key(call: ToolCall) -> str:
        """同一身份仅改变大小写时仍视为同一次输入。"""
        canonical = call.model_copy(update={"arguments": {
            key: value.strip().casefold() for key, value in call.arguments.items()
        }})
        return fingerprint(canonical)

    def identity_input(self) -> dict[str, Any] | None:
        """只推荐用户提供且尚未查询失败的身份字段，可跨轮补充邮编。"""
        name = None
        zip_code = None
        for text in reversed(self.user_texts):
            clean = self._without_emails(text)
            if zip_code is None:
                found = re.search(r"(?<!\d)\d{5}(?!\d)", clean)
                zip_code = found.group() if found else None
            if name is None:
                name = re.search(
                    r"(?:我叫|我是|姓名[是为：:]?|名字[是为：:]?|my name is|I am|this is)\s*"
                    r"([A-Za-z][A-Za-z'’-]*)\s+([A-Za-z][A-Za-z'’-]*)\b", clean, re.IGNORECASE,
                )
                if name is None:
                    name = re.search(
                        r"(?<![A-Za-z])([A-Za-z][A-Za-z'’-]*)\s+([A-Za-z][A-Za-z'’-]*)"
                        r"(?=\s*[,，。]?\s*(?:in\s+)?(?:zip(?:\s+code)?|邮编)?\s*[:：]?\s*\d{5}(?!\d))",
                        clean, re.IGNORECASE,
                    )
                if name is None:
                    name = re.fullmatch(r"\s*([A-Za-z][A-Za-z'’-]*)\s+([A-Za-z][A-Za-z'’-]*)[。.!]?\s*", clean)
            candidates = [{"name": "find_user_id_by_email", "arguments": {"email": value}}
                          for value in re.findall(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)]
            if name and zip_code:
                candidates.append({"name": "find_user_id_by_name_zip", "arguments": {
                    "first_name": name[1], "last_name": name[2], "zip": zip_code}})
            for candidate in candidates:
                call = ToolCall(id="identity-candidate", **candidate)
                source = self.identity_source_turn(call)
                if source is not None and source > self.failed_identity_inputs.get(self._identity_key(call), 0):
                    return candidate
        return None

    def needs_identity_input(self) -> bool:
        """只有查询失败且没有其他已提供候选时，才必须向用户补问。"""
        return not self.user_id and self.identity_needs_input and self.identity_input() is None

    def identity_summary(self) -> dict[str, Any]:
        """把成功、可查、待补充三个状态明确区分，供模型和日志共用。"""
        candidate = self.identity_input() if not self.user_id else None
        status = "authenticated" if self.user_id else "needs_user_input" if self.needs_identity_input() else "ready" if candidate else "missing"
        return {"status": status, "candidate": candidate, "failed_inputs": len(self.failed_identity_inputs)}

    def check_text(self, content: str) -> None:
        """捕获已观察到的认证遗漏和仅口头承诺的停滞。

        这些保守的语言检查是对工具守卫的补充；它们并非
        对所有可能的客服回复进行通用语义判断。
        """
        if self.needs_identity_input():
            return
        if not self.enhanced_validation:
            return
        request = self.user_texts[-1].casefold() if self.user_texts else ""
        business = re.search(r"换货|换个|退货|订单|修改.{0,4}地址|exchange|cancel.{0,8}order", request)
        identity_question = re.search(r"邮箱|email|e-mail|姓名|first name", content.casefold())
        refusal = re.search(r"无法|不能|不支持|cannot|can't|unsupported", content.casefold())
        if not self.user_id and business and not identity_question and not refusal:
            raise GuardError("AUTH_QUESTION_REQUIRED", "For this retail request ask for email or first name + last name + ZIP first. Do not ask the customer to provide catalog IDs or model numbers.")
        if re.search(r"请稍等|我将.{0,12}(?:核实|查询|检查)|please wait|let me check", content.casefold()):
            raise GuardError("TOOL_REQUIRED_FOR_PROGRESS", "Do not end the turn by promising future tool work. Make the required native tool call now, or ask a concrete missing-information question.")

    def check(self, call: ToolCall) -> Proposal | None:
        """校验候选调用；若需要新的确认则返回核对信息。"""
        self.validate_arguments(call)
        return self.check_business(call)

    def validate_arguments(self, call: ToolCall) -> None:
        """在领域特定校验之前应用官方严格模式。"""
        tool = self.definitions.get(call.name)
        if tool is None:
            raise GuardError("UNKNOWN_TOOL", "Use a provided native tool only.")
        extra = set(call.arguments) - set(tool.params.model_fields)
        try:
            tool.params.model_validate(call.arguments, strict=True)
        except ValidationError as error:
            raise GuardError("ARGUMENT_SCHEMA", str(error)) from error
        if extra:
            raise GuardError("ARGUMENT_SCHEMA", f"Unknown arguments: {sorted(extra)}")

    def check_business(self, call: ToolCall) -> Proposal | None:
        """利用观测到的证据校验官方 Retail 前置条件。"""
        args = call.arguments
        if call.name.startswith("find_user_id_by_"):
            if self.user_id:
                raise GuardError("IDENTITY_LOCKED", "Already authenticated; use /new for another customer.")
            source = self.identity_source_turn(call)
            if source is None:
                raise GuardError("UNGROUNDED_IDENTITY", "Use the user's complete email or real first/last name and five-digit ZIP. Never split an email into name/ZIP fields.")
            if source <= self.failed_identity_inputs.get(self._identity_key(call), 0):
                raise GuardError("IDENTITY_INPUT_REJECTED", "This exact identity input already returned no match. Ask for corrected information or use a different user-provided identity.")
            return None
        if call.name == "transfer_to_human_agents":
            return None
        if not self.user_id:
            raise GuardError("AUTH_REQUIRED", "Authenticate from user-provided email or name + ZIP before any business query.")
        if "user_id" in args and args["user_id"] != self.user_id:
            raise GuardError("WRONG_OWNER", "Only the authenticated customer's profile is accessible.")
        if "order_id" in args:
            order_id = args["order_id"]
            known = set(self.orders) | set((self.profile or {}).get("orders", []))
            if order_id not in known and not self.mentioned(order_id):
                raise GuardError("UNKNOWN_ORDER", "Order ID was not supplied or observed. Use get_user_details for the authenticated user to find their orders, or ask which order.")
            # 即使是首次查询，也绝不泄露其他客户的数据。
            order = self.environment.tools.db.orders.get(order_id)
            if order is not None and order.user_id != self.user_id:
                raise GuardError("WRONG_OWNER", "This order cannot be accessed by this customer.")
        related_products = {
            item["product_id"] for order in self.orders.values()
            for item in order["items"]
        } | self.catalog_ids
        if "product_id" in args and args["product_id"] not in related_products:
            raise GuardError("UNKNOWN_PRODUCT", "Use a product_id from an observed order or list_all_product_types.")
        if call.name == "get_item_details":
            raise GuardError("UPSTREAM_ITEM_LOOKUP", "The pinned upstream item lookup is defective. Use get_product_details with an observed product_id; it includes every variant.")
        if not self.is_write(call):
            return None
        if "order_id" in args and args["order_id"] not in self.orders:
            raise GuardError("READ_BEFORE_WRITE", "Read get_order_details before proposing changes.")
        if "user_id" in args and not self.profile:
            raise GuardError("READ_BEFORE_WRITE", "Read get_user_details before proposing profile changes.")
        payment = args.get("payment_method_id")
        if payment and payment not in (self.profile or {}).get("payment_methods", {}):
            raise GuardError("PAYMENT_NOT_OBSERVED", "Read get_user_details to retrieve the customer's payment methods; never invent one.")
        details = []
        if call.name == "exchange_delivered_order_items":
            details = self.exchange_details(args)
        return self.review(call, details)

    def review(self, call: ToolCall, details: list[str]) -> Proposal | None:
        """将有依据的核对绑定到精确参数和数据库版本。"""
        key = fingerprint(call)
        if key in self.attempted_writes:
            raise GuardError("WRITE_ALREADY_ATTEMPTED", "This exact write was already attempted. Inspect the recorded outcome and current state; do not repeat it.")
        if self.pending and self.pending.fingerprint == key and self.confirmed == key:
            if self.pending.db_hash != self.environment.get_db_hash():
                self.pending = None
                self.confirmed = None
                raise GuardError("STALE_CONFIRMATION", "State changed after confirmation. Read current details and request confirmation again.")
            return None
        review = "请核对以下模拟数据库操作（尚未执行）：\n" + call.name + "\n"
        review += json.dumps(call.arguments, ensure_ascii=False, indent=2)
        if details:
            review += "\n" + "\n".join(details)
        review += (
            "\n请确认已包含本次全部变更。回复“确认”或 "
            f"/confirm {key[:12]} 才会执行；修改方案将重新确认。"
        )
        self.pending = Proposal(key, call.name, dict(call.arguments), review,
                                self.environment.get_db_hash())
        self.confirmed = None
        return self.pending

    def exchange_details(self, args: dict[str, Any]) -> list[str]:
        """校验数量、商品/选项、库存以及带符号的差价。"""
        order = self.orders[args["order_id"]]
        old_ids, new_ids = args["item_ids"], args["new_item_ids"]
        counts = Counter(item["item_id"] for item in order["items"])
        if not old_ids or len(old_ids) != len(new_ids) or Counter(old_ids) - counts:
            raise GuardError("EXCHANGE_ITEMS", "Provide a nonempty, equal-length exchange list within the original quantities.")
        if order["status"] != "delivered":
            raise GuardError("ORDER_STATE", "Exchange requires a delivered order; inspect its current state.")
        required = self.requested_items(order)
        missing = required - set(old_ids)
        if missing and self.enhanced_validation:
            records = [{"item_id": item["item_id"], "product_id": item["product_id"],
                        "name": item["name"]} for item in order["items"]
                       if item["item_id"] in missing]
            raise GuardError("INCOMPLETE_EXCHANGE", "The user requested additional items: "
                             f"{records}. Read their product catalogs and include "
                             "ALL requested items in the single exchange proposal.")
        details: list[str] = []
        difference = 0.0
        for old_id, new_id in zip(old_ids, new_ids):
            old = next(item for item in order["items"] if item["item_id"] == old_id)
            product = self.products.get(old["product_id"])
            if not product:
                raise GuardError("CATALOG_REQUIRED", "Read get_product_details for each exchanged product before choosing variants.")
            new = product["variants"].get(new_id)
            if not new or not new["available"] or new_id == old_id:
                raise GuardError("VARIANT_UNAVAILABLE", "Choose an available, different variant of the same observed product.")
            for option, value in new["options"].items():
                if (self.enhanced_validation and value != old["options"].get(option)
                        and not self.option_requested(value)):
                    raise GuardError("UNREQUESTED_OPTION", f"Changing {option} to {value!r} is not grounded in user input. Preserve unchanged options or ask the user to specify this option explicitly.")
            difference += new["price"] - old["price"]
            details.append(
                f"{product['name']}: {old_id} {old['options']} "
                f"→ {new_id} {new['options']}；有库存；"
                f"${old['price']:.2f} → ${new['price']:.2f}"
            )
        difference = round(difference, 2)
        payment = self.profile["payment_methods"][args["payment_method_id"]]
        if payment["source"] == "gift_card" and payment["balance"] < difference:
            raise GuardError("INSUFFICIENT_BALANCE", "The selected gift card cannot cover the price difference.")
        direction = "退回" if difference < 0 else "补付" if difference > 0 else "无需补退"
        details.append(f"差价（新价−原订单价）：${difference:.2f}；{direction} ${abs(difference):.2f}。")
        details.append(f"支付/退款方式：{args['payment_method_id']} ({payment['source']})。仅提交模拟换货申请。")
        return details

    def requested_items(self, order: dict[str, Any]) -> set[str]:
        """从已观察到的多商品失败案例中捕获被遗漏的指名商品。

        支持英文目录名称和小型已记录的双语别名表。
        含糊的泛指仍需用户澄清。
        """
        required: set[str] = set()
        for text in self.user_texts:
            compact = normalized(text)
            if re.search(r"只换|只更换|仅换|only", compact):
                required.clear()
            for item in order["items"]:
                name = item.get("name") or self.products.get(item["product_id"], {}).get("name", "")
                aliases = (name,) + PRODUCT_ALIASES.get(normalized(name), ())
                for alias in aliases:
                    term = normalized(alias)
                    if not term or term not in compact:
                        continue
                    excluded = re.search(
                        r"(?:不换|不改|保留原|keeporiginal)" + re.escape(term)
                        + "|" + re.escape(term) + r"(?:不换|不动|不改|暂时不换)", compact,
                    )
                    if excluded:
                        required.discard(item["item_id"])
                    else:
                        required.add(item["item_id"])
        return required

    def observe(self, call: ToolCall, response: ToolMessage) -> None:
        """仅缓存成功的官方结果；失败不作为证据。"""
        if response.error:
            if call.name.startswith("find_user_id_by_") and response.content.strip() == "Error: User not found":
                self.failed_identity_inputs[self._identity_key(call)] = len(self.user_texts)
                self.identity_needs_input = self.identity_input() is None
            return
        if call.name.startswith("find_user_id_by_"):
            self.user_id = response.content
            self.identity_needs_input = False
        elif call.name == "get_user_details":
            self.profile = json.loads(response.content)
        elif call.name == "get_order_details":
            self.orders[call.arguments["order_id"]] = json.loads(response.content)
        elif call.name == "get_product_details":
            self.products[call.arguments["product_id"]] = json.loads(response.content)
        elif call.name == "list_all_product_types":
            self.catalog_ids.update(re.findall(r"\d{10}", response.content))

    def consume_confirmation(self, call: ToolCall) -> None:
        """在写入前使批准失效，包括可能失败的写入。"""
        self.attempted_writes.add(fingerprint(call))
        self.pending = None
        self.confirmed = None
