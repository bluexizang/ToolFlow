"""运行离线验收，阻断外部连接，保存检查报告与受保护数据哈希。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_cli import bootstrap


def file_hash(path: Path) -> str | None:
    """返回已有文件哈希，不创建或修改数据库。"""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> int:
    """运行全部离线回归，网络只允许回环测试服务。

    Returns
    -------
    int
        0 表示回归通过且受保护文件未变，其余情况返回 1。
    """
    bootstrap()
    from loguru import logger
    logger.remove()
    output = ROOT / "runs/checks" / uuid.uuid4().hex
    output.mkdir(parents=True)
    targets = {
        "official_db": ROOT / "upstream/tau2-bench-verified/data/tau2/domains/retail/db.json",
        "persistent_store": ROOT / "state/retail_extended.sqlite3",
    }
    before = {name: file_hash(path) for name, path in targets.items()}
    original_connect = socket.socket.connect
    blocked: list[str] = []

    def local_connect(connection: socket.socket, address: tuple) -> None:
        """在传输边界阻断外部连接，测试不能误请求收费服务。"""
        if address[0] not in {"127.0.0.1", "::1", "localhost"}:
            blocked.append(str(address[0]))
            raise OSError("offline acceptance blocks external connections")
        original_connect(connection, address)

    socket.socket.connect = local_connect
    try:
        with (output / "tests.txt").open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                unittest.defaultTestLoader.discover(str(ROOT / "tests"))
            )
    finally:
        socket.socket.connect = original_connect
    after = {name: file_hash(path) for name, path in targets.items()}
    passed = result.wasSuccessful() and before == after
    report = {
        "passed": passed, "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "before": before, "after": after,
        "protected_files_unchanged": before == after,
        "blocked_external_hosts": sorted(set(blocked)),
        "model_inference": False, "paid_api_requests": 0,
        "interpretation": "offline_contract_tests_not_model_success_rate",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"验收报告：{output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
