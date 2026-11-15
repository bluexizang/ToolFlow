"""最终版离线验收：使用测试传输记录，不请求真实模型服务。"""

from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import sys
import unittest
import uuid
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from run_p1 import ROOT, bootstrap

bootstrap()

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import AgentInfo, Info, Results, UserInfo
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.base import BaseUser, UserState
from toolflow.benchmark import BenchmarkAgentAdapter, BudgetedOfficialAgent, ModelAudit
from toolflow.metrics import comparison_report
from toolflow.model import ModelConfig
from toolflow.runtime import RetailRuntime
from toolflow.web import DemoServer
from toolflow.local_server import LocalServer


def raw_call(name: str, arguments: dict, call_id: str) -> dict:
    """构造传输边界测试记录，不作为生产模型实现。"""
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}],
    }}], "usage": {"total_tokens": 11, "prompt_tokens": 6, "completion_tokens": 5}}


def raw_text(content: str) -> dict:
    """构造只含文本的测试响应。"""
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content,
    }}], "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}}


class AdapterTests(unittest.TestCase):
    """在真实数据库和官方消息类型上验证适配，不用假环境。"""

    def setUp(self) -> None:
        """将测试运行日志写到临时目录，保留项目历史记录。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        original = RetailRuntime

        def runtime_factory(*args, **kwargs):
            kwargs["output_group"] = str(self.folder / "runtime")
            return original(*args, **kwargs)

        self.factory = patch("toolflow.benchmark.RetailRuntime", side_effect=runtime_factory)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.environment = get_environment()
        self.agent = BenchmarkAgentAdapter(ROOT, ModelConfig(), self.environment)

    def test_external_tool_executes_only_in_orchestrator(self) -> None:
        """提出查询时零执行，交出执行权后只接收相同 ID 的结果。"""
        state = self.agent.get_init_state()
        fixture = raw_call("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}, "identity")
        with patch.object(state.model, "complete", return_value=(fixture, 0.01)), patch.object(self.environment, "get_response", wraps=self.environment.get_response) as execute:
            message, state = self.agent.generate_next_message(UserMessage(role="user", content="我叫 Yusuf Rossi，邮编 19122，我要换货"), state)
            execute.assert_not_called()
            response = self.environment.get_response(message.tool_calls[0])
            with patch.object(state.model, "complete", return_value=(raw_text("请告诉我订单号。"), 0.01)):
                self.agent.generate_next_message(response, state)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(state.guard.user_id, "yusuf_rossi_9620")
            with self.assertRaisesRegex(ValueError, "correlation"):
                self.agent.generate_next_message(response, state)

    def test_history_reconstruction_and_no_answer_leak(self) -> None:
        """只从观测恢复身份，不允许新会话继承旧确认或看到评测答案。"""
        call = ToolCall(id="h1", name="find_user_id_by_name_zip", arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"})
        history = [UserMessage(role="user", content="Yusuf Rossi 19122"),
                   AssistantMessage(role="assistant", tool_calls=[call]), self.environment.get_response(call)]
        state = self.agent.get_init_state(history)
        self.assertEqual(state.guard.user_id, "yusuf_rossi_9620")
        self.assertIsNone(state.guard.confirmed)
        self.assertIsNone(state.guard.pending)
        with patch.object(state.model, "complete", return_value=(raw_text("请告诉我订单号。"), .01)) as send:
            self.agent.generate_next_message(UserMessage(role="user", content="继续"), state)
        body = json.dumps(send.call_args.args[0])
        self.assertNotIn("evaluation_criteria", body)
        self.assertNotIn("resolution_steps", body)
        self.assertNotIn("7706410293", body)

    def test_wrong_tool_id_is_rejected(self) -> None:
        """错误关联 ID 不能污染观测缓存。"""
        state = self.agent.get_init_state()
        with self.assertRaisesRegex(ValueError, "correlation"):
            self.agent.generate_next_message(ToolMessage(role="tool", id="unknown", requestor="assistant", content="{}"), state)
        self.assertIsNone(state.guard.user_id)

    def test_benchmark_identity_failure_uses_shared_clarification_route(self) -> None:
        """官方编排的错误观测也走同一分流，不另外累计身份失败次数。"""
        state = self.agent.get_init_state()
        raw = raw_call("find_user_id_by_email", {"email": "missing@example.invalid"}, "no-match")
        with patch.object(state.model, "complete", return_value=(raw, .01)):
            proposal, state = self.agent.generate_next_message(UserMessage(role="user", content="missing@example.invalid"), state)
        result = self.environment.get_response(proposal.tool_calls[0])
        self.assertTrue(result.error)
        with patch.object(state.model, "complete", return_value=(raw_text("请核对邮箱，或提供您的姓名和邮编。"), .01)):
            reply, state = self.agent.generate_next_message(result, state)
        self.assertIn("邮编", reply.content)
        self.assertEqual(state.state.status, "waiting_user")
        self.assertEqual(state.state.turn_repairs, 0)
        self.assertEqual(state.state.tool_calls, 1)
        self.assertEqual(state.state.recovery_action, "ask_user")

    def test_unexecuted_write_in_history_cannot_resume(self) -> None:
        """历史消息中的写入提案不代表本次仍然获得授权。"""
        call = ToolCall(id="old-write", name="cancel_pending_order", arguments={"order_id": "#W2378156", "reason": "ordered by mistake"})
        before = self.environment.get_db_hash()
        with self.assertRaisesRegex(ValueError, "fresh confirmation"):
            self.agent.get_init_state([AssistantMessage(role="assistant", tool_calls=[call])])
        self.assertEqual(before, self.environment.get_db_hash())

    def test_ablation_changes_only_declared_rule_and_recovery_flags(self) -> None:
        """三种自研方法使用同一提示词和工具，仅切换声明的策略。"""
        states = []
        for enhanced, recovery in ((False, False), (True, False), (True, True)):
            agent = BenchmarkAgentAdapter(ROOT, ModelConfig(), get_environment(), enhanced_validation=enhanced, recovery=recovery)
            state = agent.get_init_state()
            states.append(state)
            self.assertEqual(state.guard.enhanced_validation, enhanced)
            self.assertEqual(state.repair_limit, 10 if recovery else 1)
        self.assertEqual(len({s.system_prompt for s in states}), 1)
        self.assertEqual(states[0].tool_names, states[2].tool_names)

    def test_google_home_alias_matches_observed_catalog_option(self) -> None:
        """用户使用 Google Home 时能与目录中的 Google Assistant 对齐。"""
        state = self.agent.get_init_state()
        state.guard.user_texts = ["Please exchange for one compatible with Google Home"]
        self.assertTrue(state.guard.option_requested("Google Assistant"))

    def test_official_orchestrator_full_exchange_and_evaluator(self) -> None:
        """仅在离线测试使用参考动作，证明确认、单次写入和官方评测链路。"""
        task = next(task for task in get_tasks() if task.id == "0")
        actions = task.evaluation_criteria.actions
        response_fixtures = [raw_call(actions[0].name, actions[0].arguments, "a0"),
                             raw_call("get_user_details", {"user_id": "yusuf_rossi_9620"}, "profile")]
        response_fixtures.extend(raw_call(action.name, action.arguments, f"a{i}") for i, action in enumerate(actions[1:4], 1))
        response_fixtures += [raw_call("prepare_exchange", actions[4].arguments, "preview"),
                              raw_call(actions[4].name, actions[4].arguments, "commit"),
                              raw_text("The exchange request has been submitted and verified.")]
        initial = self.environment.get_db_hash()

        class TestUser(BaseUser):
            """测试专用用户输入回放，不会安装到生产入口。"""

            def get_init_state(self, message_history=None):
                return UserState(system_messages=[], messages=[])

            def generate_next_message(self, message, state):
                index = len(state.messages)
                texts = ["Yusuf Rossi 19122。订单 #W2378156：键盘换 clicky 青轴，全尺寸不变；RGB 没货可以无背光；黑色恒温器换 Google Assistant，黑色不变。用原信用卡退差价。", "yes", "###STOP###"]
                result = UserMessage(role="user", content=texts[min(index, 2)], cost=0.0)
                state.messages.append(result)
                return result, state

            def set_seed(self, seed):
                pass

            @classmethod
            def is_stop(cls, message):
                return "###STOP###" in (message.content or "")

        with patch("toolflow.model.ChatCompletionsModel.complete", side_effect=[(raw, .01) for raw in response_fixtures]), patch.object(self.environment, "get_response", wraps=self.environment.get_response) as execute:
            simulation = Orchestrator("retail", self.agent, TestUser(), self.environment, task,
                                      max_steps=60, seed=42, validate_communication=True).run()
        writes = [call for call in execute.call_args_list if call.args[0].name == "exchange_delivered_order_items"]
        self.assertEqual(len(writes), 1)
        self.assertNotEqual(initial, self.environment.get_db_hash())
        self.assertEqual(simulation.termination_reason.value, "user_stop")
        reward = evaluate_simulation(simulation, task, EvaluationType.ALL, False, "retail")
        self.assertEqual(reward.reward, 1.0)
        self.assertTrue(any(m.role == "assistant" and m.tool_calls and m.tool_calls[0].id.startswith("tf_readback_") for m in simulation.messages))
        self.assertFalse(any(m.role == "assistant" and m.tool_calls and m.tool_calls[0].name == "prepare_exchange" for m in simulation.messages))
        simulation.reward_info = reward
        results = Results(info=Info(git_commit="test", num_trials=1, max_steps=60, max_errors=3,
                                   agent_info=AgentInfo(implementation="toolflow"), user_info=UserInfo(implementation="test-only"),
                                   environment_info=self.environment.get_info()), tasks=[task], simulations=[simulation])
        target = self.folder / "results.json"
        target.write_text(results.model_dump_json(), encoding="utf-8")
        self.assertEqual(Results.load(target).simulations[0].reward_info.reward, 1.0)

    def test_official_baseline_calls_original_generate(self) -> None:
        """Baseline 保留官方提示词和工具全集，不偷加 ToolFlow 守卫。"""
        from litellm import ModelResponse
        baseline = BudgetedOfficialAgent(self.environment.get_tools(), self.environment.get_policy(), ModelConfig(), self.folder / "baseline", 2, 10000)
        state = baseline.get_init_state()
        response = ModelResponse(model="qwen3:8b", choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "What is your email?"}}], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        with patch("tau2.utils.llm_utils.completion", return_value=response) as completion:
            message, _ = baseline.generate_next_message(UserMessage(role="user", content="hello"), state)
        self.assertEqual(message.content, "What is your email?")
        self.assertEqual(len(completion.call_args.kwargs["tools"]), len(self.environment.get_tools()))
        self.assertNotIn("prepare_exchange", json.dumps(completion.call_args.kwargs["tools"]))

    def test_budget_blocks_before_second_request(self) -> None:
        """达到预算之后不再发送模型请求。"""
        audit = ModelAudit(self.folder / "budget", 1, 5000, 1024)
        audit.before()
        with self.assertRaisesRegex(RuntimeError, "budget"):
            audit.before()

    def test_benchmark_cli_writes_replayable_results_without_inference(self) -> None:
        """通过真实 CLI 和官方用户类落盘，再离线回放；传输仅在测试中替换。"""
        import sys
        import run_benchmark
        from litellm import ModelResponse
        response = ModelResponse(model="gpt-4o-mini", choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "###STOP###"}}], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        target = self.folder / "cli"
        args = ["run_benchmark.py", "--run", "--variants", "toolflow", "--tasks", "0", "--output-root", str(target)]
        with patch.object(sys, "argv", args), patch("tau2.utils.llm_utils.completion", return_value=response) as completion:
            self.assertEqual(run_benchmark.main(), 0)
            self.assertEqual(completion.call_count, 1)
        result_file = next(target.glob("*/toolflow.results.json"))
        results = Results.load(result_file)
        self.assertEqual(results.simulations[0].reward_info.reward, 0)
        self.assertEqual(results.simulations[0].termination_reason.value, "user_stop")
        self.assertTrue(next(target.glob("*/comparison.md")).exists())
        with patch.object(sys, "argv", ["run_benchmark.py", "--replay", str(result_file)]), patch("tau2.utils.llm_utils.completion") as completion:
            self.assertEqual(run_benchmark.main(), 0)
            completion.assert_not_called()
        self.assertTrue(result_file.with_name("toolflow.results.replayed.json").exists())


class WebTests(unittest.TestCase):
    """只访问回环地址的 HTTP 验收；外部提供方无请求。"""

    def setUp(self) -> None:
        """启动临时本机服务，运行数据写到临时目录。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        original = RetailRuntime

        def factory(*args, **kwargs):
            kwargs["output_group"] = str(Path(self.temp.name) / "web")
            return original(*args, **kwargs)

        self.factory = patch("toolflow.web.RetailRuntime", side_effect=factory)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.server = DemoServer(ROOT, {"local": ModelConfig()}, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, path, body=None, sid="", token=None, extra=None):
        """返回 JSON，令牌只在请求头中传递。"""
        if path == "/api/message" and isinstance(body, dict):
            body = {"request_id": uuid.uuid4().hex, **body}
        headers = {"Content-Type": "application/json", "X-ToolFlow-Token": token if token is not None else self.server.token, "X-Session-ID": sid}
        headers.update(extra or {})
        request = Request(self.url + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_auth_origin_and_no_arbitrary_file_access(self) -> None:
        """无令牌、跨源请求与任意文件路径均被拒绝。"""
        for path, body, token, extra, status in [
            ("/api/session", {}, "wrong", {}, 403),
            ("/api/session", {}, None, {"Origin": "https://evil.example"}, 403),
            ("/.env", None, None, {}, 404),
            ("/api/bootstrap", None, None, {"Host": "evil.example"}, 403),
        ]:
            with self.assertRaises(HTTPError) as error:
                self.request(path, body, token=token, extra=extra)
            self.assertEqual(error.exception.code, status)

    def test_web_message_and_session_isolation(self) -> None:
        """网页消息进入同一个 Runtime，第二个会话看不到第一个的历史。"""
        first = self.request("/api/session", {"backend": "local", "retail_mode": "official"})
        second = self.request("/api/session", {"backend": "local", "retail_mode": "official"})
        self.assertNotEqual(first["run_id"], second["run_id"])
        with patch("toolflow.model.ChatCompletionsModel.complete", return_value=(raw_text("您好，请问需要什么帮助？"), .01)):
            self.request("/api/message", {"content": "你好"}, first["session_id"])
            for _ in range(100):
                state = self.request("/api/session", sid=first["session_id"])
                if not state["busy"]:
                    break
                time.sleep(.02)
        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual(self.request("/api/session", sid=second["session_id"])["messages"], [])
        self.request("/api/close", {}, first["session_id"])
        with self.assertRaises(HTTPError) as error:
            self.request("/api/session", sid=first["session_id"])
        self.assertEqual(error.exception.code, 404)

    def test_concurrent_submit_cannot_double_execute(self) -> None:
        """上一轮未完成时，第二次提交被拒绝。"""
        session = self.request("/api/session", {"backend": "local", "retail_mode": "official"})
        started = threading.Event()
        release = threading.Event()

        def slow_response(body):
            started.set()
            release.wait(timeout=3)
            return raw_text("您好"), .01

        with patch("toolflow.model.ChatCompletionsModel.complete", side_effect=slow_response) as complete:
            self.request("/api/message", {"content": "你好"}, session["session_id"])
            self.assertTrue(started.wait(2))
            try:
                with self.assertRaises(HTTPError) as error:
                    self.request("/api/message", {"content": "你好"}, session["session_id"])
                self.assertIn(error.exception.code, {400, 409})
            finally:
                release.set()
            for _ in range(100):
                if not self.request("/api/session", sid=session["session_id"])["busy"]:
                    break
                time.sleep(.02)
            self.assertEqual(complete.call_count, 1)

    def test_request_id_retry_returns_same_receipt(self) -> None:
        """响应丢失后的重发不启动第二轮推理，改变正文也不能复用 ID。"""
        session = self.request("/api/session", {"backend": "local", "retail_mode": "official"})
        payload = {"content": "你好", "request_id": "stable-request-id-12345"}
        with patch("toolflow.model.ChatCompletionsModel.complete", return_value=(raw_text("您好"), .01)) as complete:
            first = self.request("/api/message", payload, session["session_id"])
            for _ in range(100):
                if not self.request("/api/session", sid=session["session_id"])["busy"]:
                    break
                time.sleep(.02)
            retried = self.request("/api/message", payload, session["session_id"])
            self.assertEqual(first, retried)
            self.assertEqual(complete.call_count, 1)
            with self.assertRaises(HTTPError) as error:
                self.request("/api/message", payload | {"content": "不同消息"}, session["session_id"])
            self.assertEqual(error.exception.code, 409)

    def test_page_contains_no_external_assets(self) -> None:
        """展示页不依赖远程资源，也不会把用户消息拼入 HTML。"""
        with urlopen(self.url, timeout=5) as response:
            body = response.read().decode("utf-8")
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertNotIn("innerHTML", body)
        self.assertNotIn('<script src="http', body)


class MetricsTests(unittest.TestCase):
    """配对比较不混用不同任务和种子。"""

    def test_only_matching_runs_are_paired(self) -> None:
        """缺少对应组的试验不参与配对提升估计。"""
        common = {"trial": 0, "normal_stop": True, "agent_model_calls": 3, "agent_tokens": 100, "tool_errors": 0}
        rows = [common | {"variant": "baseline", "task_id": "0", "seed": 42, "reward": 0},
                common | {"variant": "toolflow", "task_id": "0", "seed": 42, "reward": 1},
                common | {"variant": "toolflow", "task_id": "1", "seed": 42, "reward": 1}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.md"
            comparison_report(rows, path)
            report = path.read_text(encoding="utf-8")
            self.assertIn("完整配对数量：1", report)
            self.assertIn("1.000", report)


class LocalServerTests(unittest.TestCase):
    """本地启动器遵循实际选择的模型，不下载或误关已有服务。"""

    def test_existing_custom_model_does_not_require_qwen(self) -> None:
        """已启动服务中可用的其他本地模型不再被旧 Qwen 检查误拒。"""
        server = LocalServer(ROOT)
        name = "installed-local:test"
        with patch.object(server, "request", side_effect=[{"version": "test"}, {"models": [{"name": name}]}]), patch("toolflow.local_server.subprocess.Popen") as launch:
            server.ensure_ready(model=name)
            launch.assert_not_called()
        self.assertEqual(server.model, name)
        self.assertIsNone(server.process)
        server.close()

    def test_stopped_service_without_start_flag_is_not_started(self) -> None:
        """只检查时不擅自启动服务。"""
        server = LocalServer(ROOT)
        with patch.object(server, "request", side_effect=OSError("offline")), patch("toolflow.local_server.subprocess.Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                server.ensure_ready()
            launch.assert_not_called()

    def test_model_name_cannot_be_a_filesystem_path(self) -> None:
        """模型名不能变成任意 manifest 文件路径。"""
        with self.assertRaises(ValueError):
            LocalServer(ROOT).ensure_ready(start=True, model="../../private")


class WebConfigurationTests(unittest.TestCase):
    """已有云端配置自动显示，不要求用户记住额外启动参数。"""

    def config_check(self, flags: list[str], values: dict[str, str]) -> tuple[dict, str]:
        """只执行配置检查，既不监听端口也不请求模型。"""
        import run_web
        output, errors = StringIO(), StringIO()
        with patch.object(sys, "argv", ["run_web.py", "--check-config", *flags]), patch.dict(os.environ, values), redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(run_web.main(), 0)
        self.assertNotIn("fixture-cloud-secret", output.getvalue())
        return json.loads(output.getvalue()), errors.getvalue()

    def test_existing_cloud_settings_are_enabled_automatically(self) -> None:
        """配置完整时，默认同时保留本地和云端接口。"""
        result, errors = self.config_check([], {
            "TOOLFLOW_CLOUD_BASE_URL": "https://provider.example.invalid/v1",
            "TOOLFLOW_CLOUD_MODEL": "fixture-model", "TOOLFLOW_API_KEY": "fixture-cloud-secret",
        })
        self.assertEqual(set(result["models"]), {"local", "cloud"})
        self.assertEqual(result["network_requests"], 0)
        self.assertEqual(errors, "")

    def test_disable_cloud_hides_configured_backend(self) -> None:
        """显式仅本地模式不会因已有密钥而开放云端。"""
        result, _ = self.config_check(["--disable-cloud"], {
            "TOOLFLOW_CLOUD_BASE_URL": "https://provider.example.invalid/v1",
            "TOOLFLOW_CLOUD_MODEL": "fixture-model", "TOOLFLOW_API_KEY": "fixture-cloud-secret",
        })
        self.assertEqual(set(result["models"]), {"local"})

    def test_partial_cloud_settings_keep_local_with_reason(self) -> None:
        """配置不完整时，本地仍可用，终端明确给出隐藏原因。"""
        result, errors = self.config_check([], {
            "TOOLFLOW_CLOUD_BASE_URL": "", "TOOLFLOW_CLOUD_MODEL": "fixture-model", "TOOLFLOW_API_KEY": "",
        })
        self.assertEqual(set(result["models"]), {"local"})
        self.assertIn("云端选项未启用", errors)


if __name__ == "__main__":
    unittest.main()
