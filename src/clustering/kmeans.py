"""
K-Means 聚类算法实现

示例文件，展示如何继承 BaseClusterer 并实现接口。
组员可以根据需要修改或替换此文件。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.cluster import KMeans
from .base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """K-Means 聚类器"""

    def __init__(self, n_clusters: int = 3, random_state: int = 42, max_iter: int = 300):
        """
        初始化 K-Means 聚类器

        Args:
            n_clusters: 聚类数量
            random_state: 随机种子
            max_iter: 最大迭代次数
        """
        super().__init__(model_name="KMeans")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter
        )

    def train(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        训练 K-Means 模型

        Args:
            X: 输入特征（无标签）

        Returns:
            Dict: 训练结果信息
        """
        import time
        start_time = time.time()

        # 处理 pandas DataFrame
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        self.model.fit(X_values)
        self.labels_ = self.model.labels_
        self.is_trained = True

        train_time = time.time() - start_time
        return {
            "model_name": self.model_name,
            "train_time": train_time,
            "n_clusters": self.n_clusters,
            "inertia": float(self.model.inertia_),
            "n_iterations": self.model.n_iter_
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用 K-Means 模型预测聚类标签

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 聚类标签
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        # 处理 pandas DataFrame
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_values)

    def get_cluster_centers(self) -> np.ndarray:
        """
        获取聚类中心

        Returns:
            np.ndarray: 聚类中心坐标
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.cluster_centers_
