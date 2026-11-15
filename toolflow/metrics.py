"""依据真实运行记录生成配对比较，缺失试验不补成成功。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def comparison_report(rows: list[dict[str, Any]], path: Path) -> None:
    """输出任务、种子一致的对照报告。

    Parameters
    ----------
    rows : list[dict[str, Any]]
        已保存的试验指标，不包含模型参考答案。
    path : Path
        Markdown 报告目标路径。
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        groups[row["variant"]].append(row)
        pairs[(row["task_id"], row["trial"], row["seed"])][row["variant"]] = row["reward"]
    lines = ["# ToolFlow 对照报告", "", "这是实际选定任务上的描述性统计，不能代表全量 Benchmark 排名。", "",
             "|方法|试验数|平均官方奖励|正常结束比例|平均 Agent 调用|平均 Agent tokens|工具错误|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, items in groups.items():
        n = len(items)
        def mean(key: str) -> float:
            return sum(item[key] for item in items) / n
        lines.append(f"|{variant}|{n}|{mean('reward'):.3f}|{mean('normal_stop'):.3f}|{mean('agent_model_calls'):.1f}|{mean('agent_tokens'):.0f}|{sum(i['tool_errors'] for i in items)}|")
    matched = [pair["toolflow"] - pair["baseline"] for pair in pairs.values() if {"baseline", "toolflow"} <= set(pair)]
    lines += ["", f"完整配对数量：{len(matched)}。"]
    if matched:
        lines += [f"配对平均奖励差（ToolFlow − Baseline）：{sum(matched) / len(matched):.3f}。"]
    for before, after, explanation in (("runtime", "guard", "增强规则"), ("guard", "toolflow", "有限恢复")):
        changes = [pair[after] - pair[before] for pair in pairs.values() if {before, after} <= set(pair)]
        if changes:
            lines += [f"{explanation}：{len(changes)} 个配对，平均奖励差 {sum(changes) / len(changes):.3f}。"]
    from tau2.metrics.agent_metrics import is_successful, pass_hat_k
    lines += ["", "## 官方 pass^k", "", "pass^k 表示抽取 k 次全部成功的稳定性，不是至少成功一次的 pass@k。", ""]
    for variant, items in groups.items():
        task_groups: dict[str, list[float]] = defaultdict(list)
        for item in items:
            task_groups[item["task_id"]].append(item["reward"])
        for k in (1, 2, 4):
            if all(len(rewards) >= k for rewards in task_groups.values()):
                values = [pass_hat_k(len(rewards), sum(is_successful(r) for r in rewards), k) for rewards in task_groups.values()]
                lines.append(f"- {variant} pass^{k} = {sum(values) / len(values):.3f}（{len(values)} 个任务）。")
    lines += ["", "正常停止、业务评分、规则拦截、模型 token 和费用是不同指标。云端费用未估算时保持空值。",
              "相同 seed 不能保证远端服务逐字复现；两组用户模拟器会因对话分支不同而产生不同后续消息。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
