"""针对两个真实适配器实现的离线 HTTP 契约检查。"""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import run_p1

run_p1.bootstrap()

from tau2.data_model.message import UserMessage
from tau2.domains.retail.environment import get_environment
from toolflow.model import (
    CloudChatModel, LocalChatModel, ModelConfig, NoRedirect, create_model,
)
from toolflow.runtime import RetailRuntime, SessionState
from toolflow.validation import RetailGuard


KEY_NAME = "TOOLFLOW_UNIT_TEST_KEY"
TEST_KEY = "unit-test-credential-not-a-real-key"


def cloud_config() -> ModelConfig:
    """创建一个不可路由的契约测试端点，不使用真实凭据。"""
    return ModelConfig(
        backend="cloud", base_url="https://model.example.invalid/v1",
        model="contract-model", api_key_env=KEY_NAME,
    )


def http_fixture() -> dict:
    """描述一个 HTTP 解析器 fixture，而不是替代 Agent 或模型。"""
    return {
        "id": "contract-response", "model": "contract-model",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{
                "id": "contract-call", "type": "function", "function": {
                    "name": "find_user_id_by_name_zip",
                    "arguments": '{"first_name":"Yusuf","last_name":"Rossi","zip":"19122"}',
                },
            }],
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10,
                  "total_tokens": 30},
    }


class BackendTests(unittest.TestCase):
    """在无外部连接的情况下检验请求/认证边界。"""

    def test_local_request_matches_accepted_p1_record(self) -> None:
        """使用脱敏协议样例校验请求格式，不依赖作者的私有历史记录。"""
        saved = Path(__file__).parent / "fixtures/local_request.json"
        request = json.loads(saved.read_text(encoding="utf-8"))
        model = create_model(ModelConfig(model=request["model"], max_tokens=request["max_tokens"]))
        self.assertEqual(
            model.build_request(request["messages"], request["tools"]), request,
        )

    def test_local_defaults_preserved_without_authorization(self) -> None:
        """本地采样设置保持不变，且绝不读取云端密钥。"""
        config = ModelConfig()
        with patch.dict(os.environ, {config.api_key_env: TEST_KEY}):
            model = create_model(config)
        self.assertIsInstance(model, LocalChatModel)
        body = model.build_request([], [])
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["reasoning_effort"], "none")
        stream = Mock()
        stream.__enter__ = Mock(return_value=BytesIO(b"{}"))
        stream.__exit__ = Mock(return_value=False)
        model.opener.open = Mock(return_value=stream)
        model.complete(body)
        request = model.opener.open.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))

    def test_cloud_defaults_do_not_forward_local_options(self) -> None:
        """通用云端请求默认省略依赖具体服务商的调优选项。"""
        with patch.dict(os.environ, {KEY_NAME: TEST_KEY}):
            model = create_model(cloud_config())
        self.assertIsInstance(model, CloudChatModel)
        body = model.build_request([], [])
        for name in ("seed", "temperature", "reasoning_effort"):
            self.assertNotIn(name, body)
        self.assertNotIn(TEST_KEY, json.dumps(body))
        self.assertNotIn(TEST_KEY, json.dumps(asdict(model.config)))

    def test_cloud_key_required_and_invalid_urls_rejected(self) -> None:
        """当认证或端点无效时，在使用网络之前就失败。"""
        with patch.dict(os.environ, {KEY_NAME: ""}):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                create_model(cloud_config())
        for url in (
            "http://model.example.invalid/v1",
            "https://secret@model.example.invalid/v1",
            "https://model.example.invalid/v1?api_key=secret",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ModelConfig(backend="cloud", base_url=url, model="model")

    def test_cloud_header_only_and_raw_response_preserved(self) -> None:
        """使用真实的 Runtime 和 Retail 执行器，配合离线 HTTP 回复。"""
        with patch.dict(os.environ, {KEY_NAME: TEST_KEY}):
            model = create_model(cloud_config())
        raw = http_fixture()
        stream = Mock()
        stream.__enter__ = Mock(return_value=BytesIO(json.dumps(raw).encode()))
        stream.__exit__ = Mock(return_value=False)
        model.opener.open = Mock(return_value=stream)
        with tempfile.TemporaryDirectory() as folder:
            runtime = RetailRuntime.__new__(RetailRuntime)
            runtime.output = Path(folder)
            (runtime.output / "provider").mkdir()
            runtime.config = model.config
            runtime.model = model
            runtime.environment = get_environment()
            runtime.guard = RetailGuard(runtime.environment)
            runtime.guard.user_texts = ["Yusuf Rossi 19122"]
            runtime.provider_history = [{"role": "user", "content": "Yusuf Rossi 19122"}]
            runtime.tools = runtime.environment.get_tools()
            runtime.tool_names = {tool.name for tool in runtime.tools}
            runtime.seen_call_ids = set()
            runtime.max_model_calls = 2
            runtime.max_total_tokens = 10_000
            runtime.system_prompt = "HTTP contract fixture; no real inference."
            runtime.state = SessionState(
                "contract", status="running",
                messages=[UserMessage(role="user", content="Yusuf Rossi 19122")],
            )
            message = runtime.decide()
            response = runtime.execute(message)
            self.assertEqual(response.id, "contract-call")
            request = model.opener.open.call_args.args[0]
            self.assertEqual(request.full_url,
                             "https://model.example.invalid/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"),
                             f"Bearer {TEST_KEY}")
            saved = json.loads((runtime.output / "provider/001.response.json").read_text())
            self.assertEqual(saved, raw)
            summary = json.loads((runtime.output / "summary.json").read_text())
            self.assertIsNone(summary["api_fee"])
            self.assertEqual(summary["model_backend"], "cloud")
            self.assertIsNone(message.cost)
            for path in runtime.output.rglob("*.json*"):
                self.assertNotIn(TEST_KEY, path.read_text(encoding="utf-8"))
        model.opener.open.assert_called_once()

    def test_http_errors_redact_credentials_without_retry(self) -> None:
        """不在运行时错误消息中持久化回显的 bearer 密钥。"""
        with patch.dict(os.environ, {KEY_NAME: TEST_KEY}):
            model = create_model(cloud_config())
        error = HTTPError(
            "https://model.example.invalid/v1/chat/completions", 401,
            "unauthorized", {}, BytesIO(f"invalid credential: {TEST_KEY}".encode()),
        )
        model.opener.open = Mock(side_effect=error)
        with self.assertRaises(RuntimeError) as captured:
            model.complete(model.build_request([], []))
        self.assertNotIn(TEST_KEY, str(captured.exception))
        self.assertIn("[REDACTED]", str(captured.exception))
        model.opener.open.assert_called_once()
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, ""))

    def test_cloud_cli_does_not_start_local_server_without_key(self) -> None:
        """缺失云端密钥不能触发 GPU 加载或本地回退。"""
        arguments = [
            "run_p1.py", "--backend", "cloud", "--base-url",
            "https://model.example.invalid/v1", "--model", "contract-model",
            "--api-key-env", KEY_NAME,
        ]
        with patch.dict(os.environ, {KEY_NAME: ""}), patch("sys.argv", arguments):
            with patch("sys.stderr", new=StringIO()):
                with patch("toolflow.local_server.LocalServer.ensure_ready") as start:
                    self.assertEqual(run_p1.main(), 1)
                    start.assert_not_called()

    def test_check_config_never_prints_key_or_connects(self) -> None:
        """配置预览在提供云端密钥之前即可工作。"""
        arguments = [
            "run_p1.py", "--backend", "cloud", "--base-url",
            "https://model.example.invalid/v1", "--model", "contract-model",
            "--api-key-env", KEY_NAME, "--check-config",
        ]
        output = StringIO()
        with patch.dict(os.environ, {KEY_NAME: TEST_KEY}), patch("sys.argv", arguments):
            with patch("sys.stdout", new=output):
                self.assertEqual(run_p1.main(), 0)
        self.assertNotIn(TEST_KEY, output.getvalue())
        preview = json.loads(output.getvalue())
        self.assertEqual(preview["backend"], "cloud")
        self.assertEqual(preview["network_requests"], 0)


if __name__ == "__main__":
    unittest.main()
