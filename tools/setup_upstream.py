"""获取固定版本的官方基准，不修改已有的不同版本工作区。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def git(*arguments: str) -> str:
    """运行非交互 Git 命令，认证信息不写入命令行。

    Parameters
    ----------
    arguments : str
        Git 子命令及参数。

    Returns
    -------
    str
        命令标准输出。
    """
    command = ["git", "-c", "http.version=HTTP/1.1", "-c", "core.longpaths=true"]
    if sys.platform == "win32":
        command.extend(["-c", "http.sslBackend=openssl"])
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(command + list(arguments), text=True, env=environment).strip()


def main() -> int:
    """克隆并验证锁定的上游；已有工作区只检查，不覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    lock = json.loads((root / "upstream.lock.json").read_text(encoding="utf-8"))
    target = root / lock["directory"]
    if target.exists():
        if not (target / ".git").exists():
            raise RuntimeError(f"Existing path is not a Git checkout: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", lock["repository"], str(target))
        git("-C", str(target), "config", "core.longpaths", "true")
        git("-C", str(target), "checkout", "--detach", lock["commit"])
    actual = git("-C", str(target), "rev-parse", "HEAD")
    changes = git("-C", str(target), "status", "--porcelain", "--untracked-files=no")
    if actual != lock["commit"] or changes:
        raise RuntimeError("Upstream differs from the clean pinned version; preserve your checkout and use a separate project directory.")
    print(f"Verified upstream: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
