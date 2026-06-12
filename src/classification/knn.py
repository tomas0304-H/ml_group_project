"""
KNN 分类算法实现

示例文件，展示如何继承 BaseClassifier 并实现接口。
组员可以根据需要修改或替换此文件。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseClassifier


class KNNClassifier(BaseClassifier):
    """KNN 分类器"""

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        初始化 KNN 分类器

        Args:
            n_neighbors: 邻居数量
            weights: 权重类型，可选 'uniform', 'distance'
        """
        super().__init__(model_name="KNN")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        训练 KNN 模型

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            Dict: 训练结果信息
        """
        import time
        start_time = time.time()

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_time = time.time() - start_time
        return {
            "model_name": self.model_name,
            "train_time": train_time,
            "train_samples": len(X_train)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用 KNN 模型进行预测

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        使用 KNN 模型进行概率预测

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 预测概率
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)
