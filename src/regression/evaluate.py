"""
回归任务评估工具模块

提供回归模型的评估指标计算功能。
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算回归任务的各项评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        Dict: 包含各项指标的字典
            - mae: 平均绝对误差
            - mse: 均方误差
            - rmse: 均方根误差
            - r2: R²决定系数
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred))
    }
