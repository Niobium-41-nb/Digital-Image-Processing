"""
============================================================
进度条工具模块 —— 为长时间处理任务提供可视化进度显示
============================================================

提供基于 \r 的终端进度条，支持：
  - 百分比 + 进度条 + 当前/总数
  - 可选的描述文字前缀
  - 自动清理行尾

用法:
    from src.progress_bar import update_progress

    total = 450
    for i in range(total):
        # ... 处理 ...
        update_progress(i + 1, total, prefix="读取视频")
    print()  # 完成后换行
============================================================
"""

import sys


def update_progress(current: int, total: int, prefix: str = "",
                    bar_length: int = 40) -> None:
    """
    在终端同一行更新进度条显示。

    参数:
        current: 当前进度值（从 1 开始）
        total: 总进度值
        prefix: 进度条前的描述文字
        bar_length: 进度条字符长度（默认 40）
    """
    if total <= 0:
        return

    fraction = current / total
    percentage = fraction * 100

    filled_length = int(bar_length * fraction)
    bar = '█' * filled_length + '─' * (bar_length - filled_length)

    # 构建进度条字符串
    # 格式: 前缀 |█████────| 45.3% (204/450)
    progress_str = f"\r{prefix} |{bar}| {percentage:5.1f}% ({current}/{total})"

    # 写入终端并刷新
    sys.stdout.write(progress_str)
    sys.stdout.flush()


def finish_progress(prefix: str = "", bar_length: int = 40) -> None:
    """
    完成进度显示，输出 100% 状态并换行。

    参数:
        prefix: 进度条前的描述文字
        bar_length: 进度条字符长度
    """
    bar = '█' * bar_length
    sys.stdout.write(f"\r{prefix} |{bar}| 100.0% (完成)  \n")
    sys.stdout.flush()
