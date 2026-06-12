"""
聚类任务评估工具模块

提供聚类模型的评估指标计算功能。
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    计算聚类任务的各项评估指标

    Args:
        X: 输入特征数据
        labels: 聚类标签

    Returns:
        Dict: 包含各项指标的字典
            - silhouette: 轮廓系数（越接近1越好）
            - calinski_harabasz: CH指数（越大越好）
            - davies_bouldin: DB指数（越小越好）
    """
    # 处理 pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels))
    }
