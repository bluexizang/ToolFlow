"""启动本机 ToolFlow Web Demo，不在启动时调用模型。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys

from toolflow.budgets import (
    DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_REPAIRS,
)

from run_p1 import ROOT, bootstrap
from toolflow.local_config import DEFAULT_LOCAL_MODEL


def main() -> int:
    """载入现有环境配置，按需开放本地和云端后端。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start-server", action="store_true", help="启动已有本地 Ollama 服务")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    cloud = parser.add_mutually_exclusive_group()
    cloud.add_argument("--enable-cloud", dest="enable_cloud", action="store_true",
                       help="显式要求启用云端；默认自动读取已有配置")
    cloud.add_argument("--disable-cloud", dest="enable_cloud", action="store_false",
                       help="仅显示本地模型，不加载云端配置")
    parser.set_defaults(enable_cloud=None)
    parser.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS)
    parser.add_argument("--max-total-tokens", type=int, default=DEFAULT_MAX_TOTAL_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS, help="单步骤连续修正次数上限")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.max_model_calls < 1 or args.max_repairs < 1 or args.max_total_tokens < args.max_output_tokens:
        parser.error("invalid run budget")
    bootstrap()
    from loguru import logger
    logger.remove()
    from toolflow.settings import model_config
    from toolflow.web import DemoServer
    from toolflow.recovery import RUNTIME_REVISION
    from toolflow.local_server import LocalServer
    configs = {"local": model_config("local", args.local_model, max_tokens=args.max_output_tokens, timeout=args.request_timeout)}
    cloud_present = any(os.environ.get(name, "").strip() for name in (
        "TOOLFLOW_CLOUD_BASE_URL", "TOOLFLOW_CLOUD_MODEL", "TOOLFLOW_API_KEY",
    ))
    if args.enable_cloud is not False and (args.enable_cloud or cloud_present):
        try:
            config = model_config("cloud", max_tokens=args.max_output_tokens, timeout=args.request_timeout)
            if not os.environ.get(config.api_key_env, "").strip():
                raise ValueError("云端密钥未配置")
            configs["cloud"] = config
        except ValueError as error:
            if args.enable_cloud:
                raise
            # 自动检测失败不阻断本地使用，但要说明为什么没有云端选项。
            print(f"云端选项未启用：{error}", file=sys.stderr)
    if args.check_config:
        print(json.dumps({"models": {k: asdict(v) for k, v in configs.items()}, "network_requests": 0, "max_model_calls": args.max_model_calls, "max_total_tokens": args.max_total_tokens, "max_repairs": args.max_repairs, "runtime_revision": RUNTIME_REVISION}, ensure_ascii=False, indent=2))
        return 0
    server = None
    ollama = None
    try:
        if args.start_server:
            ollama = LocalServer(ROOT)
            ollama.ensure_ready(start=True, model=args.local_model)
        server = DemoServer(ROOT, configs, args.port, args.max_model_calls, args.max_total_tokens, args.max_repairs)
        print(f"ToolFlow Web: http://127.0.0.1:{args.port} （仅本机；按 Ctrl+C 结束）", flush=True)
        print(f"Runtime: {RUNTIME_REVISION}; calls={args.max_model_calls}; tokens={args.max_total_tokens}; repairs={args.max_repairs}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            for session in list(server.sessions.values()):
                session.stopping.set()
                if not session.busy:
                    session.runtime.close("web_server_shutdown")
            server.server_close()
        if ollama is not None:
            ollama.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Web stopped: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
