"""
分类算法基类

所有分类算法必须继承此基类，并实现 train()、predict()、predict_proba() 方法。
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import os


class BaseClassifier(ABC):
    """分类算法基类，所有分类算法必须继承此类"""

    def __init__(self, model_name: str):
        """
        初始化分类器

        Args:
            model_name: 模型名称，用于标识和保存
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            **kwargs: 其他参数

        Returns:
            Dict: 训练结果信息（如训练时间等）
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        模型预测

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 预测结果
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        预测概率（如果模型支持）

        Args:
            X: 输入特征

        Returns:
            Optional[np.ndarray]: 预测概率，不支持则返回 None
        """
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            Dict: 包含各项指标的字典
                - accuracy: 准确率
                - precision: 精确率（宏平均）
                - recall: 召回率（宏平均）
                - f1: F1分数（宏平均）
                - auc: AUC值（如果支持概率预测）
        """
        from .evaluate import classification_metrics
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        return classification_metrics(y_test, y_pred, y_proba)

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

    def load_model(self, path: str) -> 'BaseClassifier':
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
