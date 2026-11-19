"""ToolFlow 官方 Retail 评测入口；默认只检查，--run 才会请求模型。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

from toolflow.budgets import (
    DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_REPAIRS,
)
import subprocess
import sys
import time
import uuid

from run_cli import ROOT, bootstrap

PINNED_SHA = "864350a8971a8f8ee9e7b8472e2edc380a806b0c"


def verify_upstream() -> str:
    """检查固定版本及跟踪文件，防止不一致的评测环境。"""
    upstream = ROOT / "upstream/tau2-bench-verified"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if commit != PINNED_SHA or changed:
        raise ValueError("upstream must match the clean audited commit")
    return commit


def main() -> int:
    """组织独立试验并保存官方 Results；运行和回放明确分开。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="实际运行，云端配置可能产生费用")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--replay", type=Path, help="离线重新计算官方评分，不调用模型")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/benchmark")
    parser.add_argument("--tasks", nargs="+", default=["0"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "runtime", "guard", "toolflow"), default=["baseline", "toolflow"],
                        help="baseline=官方原版；runtime=基础安全；guard=增强规则；toolflow=增强规则+恢复")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS)
    parser.add_argument("--max-total-tokens", type=int, default=DEFAULT_MAX_TOTAL_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS, help="单步骤连续修正次数上限")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    for role in ("agent", "user"):
        parser.add_argument(f"--{role}-backend", choices=("local", "cloud"), default="local")
        parser.add_argument(f"--{role}-model")
        parser.add_argument(f"--{role}-base-url")
        parser.add_argument(f"--{role}-api-key-env", default="TOOLFLOW_API_KEY")
    args = parser.parse_args()
    bootstrap()
    from loguru import logger
    logger.remove()
    from tau2.data_model.simulation import AgentInfo, Info, Results, SimulationRun, UserInfo
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.utils.utils import get_now
    from toolflow.benchmark import BenchmarkAgentAdapter, BudgetedOfficialAgent, BudgetedOfficialUser
    from toolflow.runtime import save_json
    from toolflow.settings import model_config
    from toolflow.metrics import comparison_report

    commit = verify_upstream()
    if args.replay:
        results = Results.load(args.replay)
        if results.info.git_commit != commit:
            raise ValueError("replay commit mismatch")
        tasks = {task.id: task for task in results.tasks}
        rewards = []
        for simulation in results.simulations:
            reward = evaluate_simulation(simulation, tasks[simulation.task_id], EvaluationType.ALL, False, "retail")
            rewards.append({"simulation_id": simulation.id, "reward": reward.model_dump(mode="json")})
        destination = args.replay.with_name(args.replay.stem + ".replayed.json")
        save_json(destination, rewards)
        print(f"离线评测完成：{destination}")
        return 0
    if min(args.trials, args.max_steps, args.max_model_calls, args.max_output_tokens, args.max_repairs) < 1:
        parser.error("trials/steps/calls/output-tokens must be positive")
    if args.max_total_tokens < args.max_output_tokens:
        parser.error("total token budget must cover output token limit")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.variants)) != len(args.variants):
        parser.error("duplicate task/variant would invalidate paired comparison")
    tasks_by_id = {task.id: task for task in get_tasks(args.split)}
    if any(task_id not in tasks_by_id for task_id in args.tasks):
        parser.error("task not found in selected split")
    configs = {}
    for role in ("agent", "user"):
        configs[role] = model_config(
            getattr(args, f"{role}_backend"), getattr(args, f"{role}_model"),
            getattr(args, f"{role}_base_url"), getattr(args, f"{role}_api_key_env"),
            args.max_output_tokens, args.request_timeout,
        )
    plan = {
        "commit": commit, "domain": "retail", "split": args.split,
        "tasks": args.tasks, "variants": args.variants, "trials": args.trials,
        "models": {role: asdict(config) for role, config in configs.items()},
        "per_role_max_calls": args.max_model_calls, "per_role_max_tokens": args.max_total_tokens,
        "max_steps": args.max_steps, "seed": args.seed, "max_repairs": args.max_repairs,
        "network_requests": 0, "reference_answers_visible_to_agent": False,
        "local_persistent_store_used": False,
        "data_sha256": {
            name: hashlib.sha256((ROOT / "upstream/tau2-bench-verified/data/tau2/domains/retail" / name).read_bytes()).hexdigest()
            for name in ("db.json", "tasks.json", "split_tasks.json", "policy.md")
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sorted((ROOT / "toolflow").glob("*.py"))]
        },
        "dependencies": {name: version(name) for name in ("litellm", "pydantic", "pandas", "loguru")},
    }
    if not args.run or args.check_config:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    for config in configs.values():
        if config.backend == "cloud" and not os.environ.get(config.api_key_env, "").strip():
            raise ValueError(f"missing key in {config.api_key_env}")
    output = args.output_root.resolve() / uuid.uuid4().hex
    output.mkdir(parents=True)
    save_json(output / "manifest.json", plan | {"network_requests": "see role audit logs"})
    metrics = []
    failures = 0
    for variant in args.variants:
        simulations = []
        selected = [tasks_by_id[task_id] for task_id in args.tasks]
        for task in selected:
            for trial in range(args.trials):
                seed = args.seed + trial
                folder = output / variant / f"{task.id}_{trial}"
                folder.mkdir(parents=True)
                environment = get_environment()
                initial_hash = environment.get_db_hash()
                agent = (
                    BenchmarkAgentAdapter(ROOT, configs["agent"], environment, args.max_model_calls, args.max_total_tokens,
                                          enhanced_validation=variant != "runtime", recovery=variant == "toolflow", max_repairs=args.max_repairs)
                    if variant != "baseline" else BudgetedOfficialAgent(
                        environment.get_tools(), environment.get_policy(), configs["agent"],
                        folder / "agent", args.max_model_calls, args.max_total_tokens,
                    )
                )
                user = BudgetedOfficialUser(str(task.user_scenario), configs["user"], folder / "user",
                                            args.max_model_calls, args.max_total_tokens)
                orchestrator = Orchestrator("retail", agent, user, environment, task,
                                            args.max_steps, 3, seed, validate_communication=True)
                started = time.perf_counter()
                error = None
                try:
                    simulation = orchestrator.run()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    for config in configs.values():
                        key = os.environ.get(config.api_key_env, "")
                        if key:
                            error = error.replace(key, "[REDACTED]")
                    failures += 1
                    simulation = SimulationRun(
                        id=uuid.uuid4().hex, task_id=task.id, start_time=get_now(), end_time=get_now(),
                        duration=time.perf_counter() - started,
                        termination_reason="user_error" if getattr(orchestrator.to_role, "value", None) == "user" else "agent_error",
                        messages=orchestrator.get_trajectory(), seed=seed,
                    )
                    if variant != "baseline" and agent.runtime:
                        agent.runtime.close("benchmark_error")
                simulation.trial = trial
                # 上游对未知模型价格可能回退为零；这里统一标为未知，避免误报免费。
                simulation.agent_cost = 0.0 if configs["agent"].backend == "local" else None
                simulation.user_cost = 0.0 if configs["user"].backend == "local" else None
                # 评分器异常也必须留下真实轨迹和最终数据库，不能丢失失败证据。
                save_json(folder / "simulation.json", simulation.model_dump(mode="json"))
                save_json(folder / "final_db.json", environment.tools.db.model_dump(mode="json"))
                simulation.reward_info = evaluate_simulation(simulation, task, EvaluationType.ALL, False, "retail")
                simulations.append(simulation)
                save_json(folder / "simulation.json", simulation.model_dump(mode="json"))
                save_json(folder / "final_db.json", environment.tools.db.model_dump(mode="json"))
                runtime = agent.runtime if variant != "baseline" else None
                calls = runtime.state.model_calls if runtime else agent.audit.calls
                tokens = runtime.state.total_tokens if runtime else agent.audit.tokens
                metrics.append({
                    "variant": variant, "task_id": task.id, "trial": trial, "seed": seed,
                    "reward": simulation.reward_info.reward,
                    "normal_stop": simulation.termination_reason.value in {"user_stop", "agent_stop"},
                    "termination_reason": simulation.termination_reason.value,
                    "business_success": simulation.reward_info.reward == 1.0,
                    "agent_model_calls": calls, "agent_tokens": tokens,
                    "user_model_calls": user.audit.calls, "user_tokens": user.audit.tokens,
                    "blocked_proposals": runtime.state.blocked_proposals if runtime else 0,
                    "tool_errors": sum(1 for m in simulation.messages if m.role == "tool" and m.error),
                    "duration_seconds": simulation.duration, "error": error,
                    "initial_db_hash": initial_hash, "final_db_hash": environment.get_db_hash(),
                    "runtime_output": str(runtime.output) if runtime else None,
                    "api_fee": 0.0 if all(c.backend == "local" for c in configs.values()) else None,
                })
                print(f"{variant} task={task.id} trial={trial} reward={simulation.reward_info.reward} stop={simulation.termination_reason.value}", flush=True)
        result = Results(info=Info(
            git_commit=commit, num_trials=args.trials, max_steps=args.max_steps, max_errors=3,
            agent_info=AgentInfo(implementation=variant, llm=configs["agent"].model,
                                 llm_args=asdict(configs["agent"])),
            user_info=UserInfo(implementation="official_user_simulator", llm=configs["user"].model,
                               llm_args=asdict(configs["user"])),
            environment_info=get_environment().get_info(), seed=args.seed,
        ), tasks=selected, simulations=simulations)
        save_json(output / f"{variant}.results.json", result.model_dump(mode="json"))
    save_json(output / "metrics.json", metrics)
    comparison_report(metrics, output / "comparison.md")
    print(f"运行记录（含离线参考答案，禁止公开上传）：{output}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Benchmark stopped: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
