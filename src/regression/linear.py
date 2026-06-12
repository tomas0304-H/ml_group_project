"""
线性回归算法实现

示例文件，展示如何继承 BaseRegressor 并实现接口。
组员可以根据需要修改或替换此文件。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge
from .base import BaseRegressor


class LinearRegressor(BaseRegressor):
    """线性回归器"""

    def __init__(self, use_ridge: bool = False, alpha: float = 1.0):
        """
        初始化线性回归器

        Args:
            use_ridge: 是否使用 Ridge 回归（带 L2 正则化）
            alpha: 正则化强度（仅在 use_ridge=True 时生效）
        """
        super().__init__(model_name="LinearRegression" if not use_ridge else "RidgeRegression")
        self.use_ridge = use_ridge
        self.alpha = alpha

        if use_ridge:
            self.model = Ridge(alpha=alpha)
        else:
            self.model = LinearRegression()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        训练线性回归模型

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
        result = {
            "model_name": self.model_name,
            "train_time": train_time,
            "train_samples": len(X_train)
        }

        # 如果是线性回归，返回系数
        if hasattr(self.model, 'coef_'):
            result["coefficients"] = self.model.coef_.tolist()
            result["intercept"] = float(self.model.intercept_)

        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用线性回归模型进行预测

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)
