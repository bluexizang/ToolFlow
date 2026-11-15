"""在消耗一次模型请求之前，先收集完整的控制台消息。"""

from __future__ import annotations

from collections.abc import Callable
import unicodedata


def clean_input(content: str) -> str:
    """在每个入口拒绝控制字符与超长消息。

    Parameters
    ----------
    content : str
        完整的用户消息，包含有意的换行。

    Returns
    -------
    str
        去除首尾空白、适合进入对话历史的 Unicode 文本。
    """
    if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t"
           for char in content):
        raise ValueError("输入含控制字符，请重新粘贴；Ctrl+V 本身不是用户消息。")
    content = content.strip()
    if not content or len(content) > 16_000:
        raise ValueError("请输入 1–16000 个字符；多行内容用 /send 一次提交。")
    return content


def read_user_input(
    *, single_line: bool = False,
    read_line: Callable[[str], str] = input,
) -> str:
    """读取一条消息；显式提交可防止粘贴行被逐行误发。

    Parameters
    ----------
    single_line : bool
        选择启用传统的回车即提交行为。
    read_line : Callable[[str], str]
        控制台行读取器，可注入以便离线进行输入回归测试。
    """
    if single_line:
        return clean_input(read_line("\nUser> "))
    lines: list[str] = []
    commands = {"/finish", "/status", "/new"}
    while True:
        line = read_line("\nUser> " if not lines else "... ")
        if not lines and (line.strip() in commands
                          or line.strip().startswith("/confirm ")):
            return clean_input(line)
        if line.strip() == "/send":
            return clean_input("\n".join(lines))
        # 立即拒绝无效粘贴，避免继续提示输入或发起 API 调用。
        if line.strip():
            clean_input(line)
        lines.append(line)
