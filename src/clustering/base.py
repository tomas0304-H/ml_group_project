"""
聚类算法基类

所有聚类算法必须继承此基类，并实现 train()、predict() 方法。
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import os


class BaseClusterer(ABC):
    """聚类算法基类，所有聚类算法必须继承此类"""

    def __init__(self, model_name: str):
        """
        初始化聚类器

        Args:
            model_name: 模型名称，用于标识和保存
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.labels_ = None

    @abstractmethod
    def train(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        训练聚类模型

        Args:
            X: 输入特征（无标签）
            **kwargs: 其他参数

        Returns:
            Dict: 训练结果信息
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测聚类标签

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 聚类标签
        """
        pass

    def evaluate(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        评估聚类性能

        Args:
            X: 输入特征

        Returns:
            Dict: 包含各项指标的字典
                - silhouette: 轮廓系数
                - calinski_harabasz: CH指数
                - davies_bouldin: DB指数
        """
        from .evaluate import clustering_metrics
        labels = self.predict(X) if self.labels_ is None else self.labels_
        return clustering_metrics(X, labels)

    def save_model(self, path: str) -> str:
        """
        保存模型到指定路径

        Args:
            path: 保存路径

        Returns:
            保存的文件路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load_model(self, path: str) -> 'BaseClusterer':
        """
        从指定路径加载模型

        Args:
            path: 模型文件路径

        Returns:
            self
        """
        self.model = joblib.load(path)
        self.is_trained = True
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
