"""
============================================================
配置加载模块 —— 从 .env 文件加载运行参数
============================================================

提供从 .env 配置文件加载视频选择、卷积核大小、图像放大
倍数等参数的功能。

依赖：
  - pathlib
============================================================
"""

from pathlib import Path


def load_env_config(env_file=".env"):
    """从 .env 文件加载配置参数"""
    config = {}
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    return config


def get_config():
    """加载并返回所有配置参数"""
    env_config = load_env_config()

    return {
        'VIDEO_CHOICE': int(env_config.get('VIDEO_CHOICE', 2)),
        'CONVOLUTION_SIZE': int(env_config.get('CONVOLUTION_SIZE', 3)),
        'SCALE_FACTOR': int(env_config.get('SCALE_FACTOR', 4)),
        'POOL_SIZE': int(env_config.get('POOL_SIZE', int(env_config.get('SCALE_FACTOR', 4)))),
    }
