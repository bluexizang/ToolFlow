"""新入口共享的配置解析；只读取指定的环境变量。"""

from __future__ import annotations

import os

from .model import ModelConfig
from .local_config import DEFAULT_LOCAL_MODEL
from .budgets import DEFAULT_MAX_OUTPUT_TOKENS


def model_config(backend: str, model: str | None = None,
                 base_url: str | None = None, api_key_env: str = "TOOLFLOW_API_KEY",
                 max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, timeout: float = 180.0) -> ModelConfig:
    """解析本地或云端配置，不访问网络。

    Parameters
    ----------
    backend : str
        local 或 cloud。
    model : str | None
        显式模型名，优先于环境变量。
    base_url : str | None
        显式服务地址，优先于环境变量。
    api_key_env : str
        密钥所在环境变量的名字。
    max_tokens : int
        单次输出上限。
    timeout : float
        请求超时秒数。

    Returns
    -------
    ModelConfig
        可保存的配置，不包含密钥值。
    """
    if backend == "cloud":
        model = model or os.environ.get("TOOLFLOW_CLOUD_MODEL")
        base_url = base_url or os.environ.get("TOOLFLOW_CLOUD_BASE_URL")
        if not model or not base_url:
            raise ValueError("请设置 TOOLFLOW_CLOUD_MODEL 和 TOOLFLOW_CLOUD_BASE_URL")
    return ModelConfig(
        backend=backend, model=model or DEFAULT_LOCAL_MODEL,
        base_url=base_url or "http://127.0.0.1:11434/v1",
        api_key_env=api_key_env, max_tokens=max_tokens, timeout_seconds=timeout,
    )
