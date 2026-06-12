"""
数据加载/保存工具模块

提供统一的数据读写功能。
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Any, Optional


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    加载 CSV 文件

    Args:
        file_path: 文件路径
        **kwargs: pandas.read_csv 的其他参数

    Returns:
        DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, **kwargs)


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> str:
    """
    保存 DataFrame 到 CSV 文件

    Args:
        df: 要保存的 DataFrame
        file_path: 保存路径
        **kwargs: pandas.to_csv 的其他参数

    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False, **kwargs)
    return file_path


def save_model(model: Any, file_path: str) -> str:
    """
    使用 joblib 保存模型

    Args:
        model: 模型对象
        file_path: 保存路径

    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    return file_path


def load_model(file_path: str) -> Any:
    """
    使用 joblib 加载模型

    Args:
        file_path: 模型文件路径

    Returns:
        加载的模型对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return joblib.load(file_path)


def save_metrics(metrics: dict, file_path: str) -> str:
    """
    保存评估指标到 JSON 文件

    Args:
        metrics: 指标字典
        file_path: 保存路径

    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 将 numpy 类型转换为 Python 原生类型
    converted = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            converted[k] = int(v)
        elif isinstance(v, (np.floating,)):
            converted[k] = float(v)
        elif isinstance(v, np.ndarray):
            converted[k] = v.tolist()
        else:
            converted[k] = v

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    return file_path


def load_metrics(file_path: str) -> dict:
    """
    从 JSON 文件加载评估指标

    Args:
        file_path: JSON 文件路径

    Returns:
        指标字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
