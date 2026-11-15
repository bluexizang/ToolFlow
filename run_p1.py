"""使用已安装的本地模型或云端 Chat Completions 服务运行 ToolFlow。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from toolflow.budgets import (
    DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_REPAIRS,
)


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("TOOLFLOW_PYTHON") or sys.executable)


def bootstrap() -> None:
    """检查可选的解释器约束，优先加载隔离的项目依赖。"""
    if sys.version_info < (3, 10):
        raise RuntimeError("ToolFlow requires Python 3.10 or newer")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use the required interpreter: {PYTHON}")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / ".p0_deps"), str(ROOT / "upstream/tau2-bench-verified/src")]
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ["TAU2_DATA_DIR"] = str(ROOT / "upstream/tau2-bench-verified/data")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def main() -> int:
    """在显式循环中驱动决策、官方工具和真实用户回复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "cloud"), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="TOOLFLOW_API_KEY")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS)
    parser.add_argument("--max-total-tokens", type=int, default=DEFAULT_MAX_TOTAL_TOKENS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS, help="单步骤连续修正次数上限")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--single-line", action="store_true", help="Enter 提交单行；默认 /send 提交整段输入")
    parser.add_argument("--retail-mode", choices=("extended", "official"), default="extended")
    parser.add_argument("--store", type=Path, help="独立扩展数据库路径，仅用于 extended 模式")
    args = parser.parse_args()
    bootstrap()
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    from toolflow.model import ModelConfig
    from toolflow.local_config import DEFAULT_LOCAL_MODEL
    from toolflow.runtime import RetailRuntime
    from toolflow.console import read_user_input

    server = None
    runtime = None
    try:
        if args.backend == "cloud" and args.start_server:
            parser.error("--start-server only applies to the local backend")
        if args.retail_mode == "official" and args.store is not None:
            parser.error("--store only applies to --retail-mode extended")
        base_url = args.base_url
        model_name = args.model
        if args.backend == "cloud":
            base_url = base_url or os.environ.get("TOOLFLOW_CLOUD_BASE_URL")
            model_name = model_name or os.environ.get("TOOLFLOW_CLOUD_MODEL")
            if not base_url or not model_name:
                parser.error("Cloud mode needs --base-url/--model or TOOLFLOW_CLOUD_* settings")
        config = ModelConfig(
            backend=args.backend,
            base_url=base_url or "http://127.0.0.1:11434/v1",
            model=model_name or DEFAULT_LOCAL_MODEL, api_key_env=args.api_key_env,
            seed=args.seed, temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_output_tokens,
            timeout_seconds=args.request_timeout,
        )
        if args.check_config:
            from dataclasses import asdict
            import json
            summary = asdict(config)
            summary["api_key_configured"] = (
                bool(os.environ.get(config.api_key_env, "").strip())
                if config.backend == "cloud" else False
            )
            summary["network_requests"] = 0
            summary["max_model_calls"] = args.max_model_calls
            summary["max_total_tokens"] = args.max_total_tokens
            summary["max_repairs"] = args.max_repairs
            summary["retail_mode"] = args.retail_mode
            summary["store_path"] = str((args.store or ROOT / "state/retail_extended.sqlite3").resolve()) if args.retail_mode == "extended" else None
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if config.backend == "local":
            from toolflow.local_server import LocalServer
            server = LocalServer(ROOT)
            server.ensure_ready(start=args.start_server, model=config.model)
        runtime = RetailRuntime(
            ROOT, config, args.max_model_calls,
            args.max_total_tokens, retail_mode=args.retail_mode, max_repairs=args.max_repairs,
            store_path=(args.store or ROOT / "state/retail_extended.sqlite3") if args.retail_mode == "extended" else None,
        )
        print(f"Run directory: {runtime.output}", flush=True)
        print(f"Model: {config.backend} / {config.model}", flush=True)
        print(f"Retail mode: {args.retail_mode}", flush=True)
        print("/finish 结束，/status 查看预算，/new 新会话。", flush=True)
        print("默认可粘贴多行；单独输入 /send 提交整段消息。确认单可直接用 /confirm 编号。", flush=True)
        print("扩展模式会持久保存模拟账号和订单；官方模式使用每次运行的独立数据。", flush=True)
        while True:
            try:
                content = read_user_input(single_line=args.single_line)
            except ValueError as error:
                print(f"[input] {error}（未调用模型）", flush=True)
                continue
            if content == "/finish":
                runtime.close("user_stop")
                break
            if content == "/status":
                print({
                    "status": runtime.state.status,
                    "model_calls": runtime.state.model_calls,
                    "tool_calls": runtime.state.tool_calls,
                    "total_tokens": runtime.state.total_tokens,
                }, flush=True)
                continue
            if content == "/new":
                runtime.close("user_reset")
                runtime = RetailRuntime(
                    ROOT, config, args.max_model_calls,
                    args.max_total_tokens, retail_mode=args.retail_mode, max_repairs=args.max_repairs,
                    store_path=(args.store or ROOT / "state/retail_extended.sqlite3") if args.retail_mode == "extended" else None,
                )
                print(f"New isolated run: {runtime.output}", flush=True)
                continue
            if not content:
                continue
            runtime.receive(content)
            while runtime.state.status == "running":
                print("[model] 等待下一步决策…", flush=True)
                message = runtime.decide()
                if message.tool_calls:
                    call = message.tool_calls[0]
                    print(f"[tool] {call.name} {call.arguments} [{call.id}]", flush=True)
                    result = runtime.execute(message)
                    text = result.content or ""
                    preview = text if len(text) <= 800 else text[:800] + " … [完整内容已保存]"
                    print(f"[result error={result.error}] {preview}", flush=True)
                    if runtime.state.status == "waiting_user":
                        print(f"\nAgent> {runtime.state.messages[-1].content}", flush=True)
                else:
                    print(f"\nAgent> {message.content}", flush=True)
    except (EOFError, KeyboardInterrupt):
        if runtime:
            runtime.close("operator_interrupt")
    except Exception as error:
        if runtime:
            runtime.event("runtime_error", {"type": type(error).__name__, "message": str(error)})
            runtime.close("runtime_error")
        print(f"P1 stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
