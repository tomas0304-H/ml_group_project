"""
分类任务评估工具模块

提供分类模型的评估指标计算功能。
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算分类任务的各项评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选，用于计算 AUC）

    Returns:
        Dict: 包含各项指标的字典
            - accuracy: 准确率
            - precision: 精确率（宏平均）
            - recall: 召回率（宏平均）
            - f1: F1分数（宏平均）
            - auc: AUC值（如果提供概率）
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "auc": None
    }

    # 如果提供了预测概率，计算 AUC
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                # 多分类情况
                metrics["auc"] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
            elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # 二分类情况，取正类概率
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc"] = None

    return metrics
