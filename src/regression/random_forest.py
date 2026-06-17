"""
随机森林回归算法实现

继承 BaseRegressor 基类，封装 sklearn.ensemble.RandomForestRegressor。
支持特征重要性提取与超参数配置。

提供两种使用方式：
1. 类接口：RandomForestRegressor 类（用于 main_regression.py）
2. 函数接口：load_data, preprocess, train_model 等（用于 Streamlit 页面）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.model_selection import train_test_split
from .base import BaseRegressor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 默认配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "regression", "SeoulBikeData.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "regression")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "regression")

DEFAULT_TARGET_COL = "Rented Bike Count"
DEFAULT_DROP_COLS = ["Date"]
DEFAULT_CATEGORICAL_COLS = ["Seasons", "Holiday", "Functioning Day"]
DEFAULT_FILTER_FUNCTIONING = True


# ========== Streamlit 友好的函数接口 ==========

def load_data(csv_path=None, encoding='ISO-8859-1'):
    """
    加载回归数据集。

    Args:
        csv_path: CSV文件路径，默认使用 SeoulBikeData
        encoding: 文件编码

    Returns:
        pd.DataFrame: 原始数据
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件：{csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)

    # 清理列名中的乱码字符（°C 编码问题）
    col_rename = {}
    for col in df.columns:
        cleaned = col.encode('ISO-8859-1').decode('utf-8', errors='ignore')
        if cleaned != col:
            col_rename[col] = cleaned
    if col_rename:
        df = df.rename(columns=col_rename)

    return df


def preprocess(df, target_col=None, drop_cols=None, categorical_cols=None,
               filter_functioning=True):
    """
    数据预处理：删除无用列、过滤非运营日、One-Hot编码。

    Args:
        df: 原始DataFrame
        target_col: 目标列名
        drop_cols: 需要删除的列
        categorical_cols: 分类特征列
        filter_functioning: 是否过滤非运营日

    Returns:
        dict: {
            'X': 特征矩阵,
            'y': 目标变量,
            'feature_names': 特征列名列表,
            'df_processed': 处理后的DataFrame
        }
    """
    if target_col is None:
        target_col = DEFAULT_TARGET_COL
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS.copy()
    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL_COLS.copy()

    df_processed = df.copy()

    # 删除无用列
    cols_to_drop = [c for c in drop_cols if c in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)

    # 过滤非运营日
    if filter_functioning and "Functioning Day" in df_processed.columns:
        df_processed = df_processed[df_processed["Functioning Day"] == "Yes"]
        df_processed = df_processed.drop(columns=["Functioning Day"])
        if "Functioning Day" in categorical_cols:
            categorical_cols.remove("Functioning Day")

    # One-Hot 编码
    cols_to_encode = [c for c in categorical_cols if c in df_processed.columns]
    if cols_to_encode:
        df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True)

    # 分离特征和标签
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    return {
        'X': X,
        'y': y,
        'feature_names': list(X.columns),
        'df_processed': df_processed
    }


def train_model(X, y, test_size=0.2, random_state=42,
                n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2):
    """
    训练随机森林回归模型。

    Args:
        X: 特征矩阵
        y: 目标变量
        test_size: 测试集比例
        random_state: 随机种子
        n_estimators: 树的数量
        max_depth: 最大深度
        min_samples_split: 最小分裂样本数
        min_samples_leaf: 最小叶节点样本数

    Returns:
        dict: {
            'model': RandomForestRegressor 实例,
            'X_train': 训练特征,
            'X_test': 测试特征,
            'y_train': 训练标签,
            'y_test': 测试标签,
            'train_info': 训练信息,
            'metrics': 评估指标,
            'feature_names': 特征名列表
        }
    """
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 创建并训练模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    train_info = model.train(X_train, y_train)

    # 评估
    metrics = model.evaluate(X_test, y_test)

    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_info': train_info,
        'metrics': metrics,
        'feature_names': list(X.columns)
    }


def plot_feature_importance(importances, top_n=15):
    """
    绘制特征重要性柱状图。

    Args:
        importances: 特征重要性字典 {特征名: 重要性分数}
        top_n: 显示前N个特征

    Returns:
        matplotlib.Figure
    """
    # 取前N个特征
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n])

    features = list(sorted_imp.keys())
    scores = list(sorted_imp.values())

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color='steelblue', edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('重要性分数', fontsize=12)
    ax.set_title(f'随机森林特征重要性 (Top {len(features)})', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_prediction_vs_actual(y_true, y_pred):
    """
    绘制预测值 vs 真实值散点图。

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        matplotlib.Figure
    """
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    ax.set_xlabel('真实值', fontsize=12)
    ax.set_ylabel('预测值', fontsize=12)
    ax.set_title(f'Prediction vs Actual (R2={r2:.4f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    """
    绘制残差分布图。

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        matplotlib.Figure
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('残差 (真实值 - 预测值)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('残差分布', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


class RandomForestRegressor(BaseRegressor):
    """随机森林回归器"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        初始化随机森林回归器

        Args:
            n_estimators: 树的数量，默认 100
            max_depth: 树的最大深度，None 表示不限制
            min_samples_split: 内部节点再划分所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            max_features: 每棵树考虑的最大特征数，'sqrt'/'log2'/float/int
            random_state: 随机种子
            n_jobs: 并行线程数，-1 表示使用全部 CPU
            **kwargs: 传递给 sklearn RandomForestRegressor 的其他参数
        """
        super().__init__(model_name="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.model = SKRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        self.feature_names: Optional[List[str]] = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        训练随机森林模型

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            Dict: 训练结果信息
        """
        import time
        start_time = time.time()

        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_time = time.time() - start_time
        result = {
            "model_name": self.model_name,
            "train_time": train_time,
            "train_samples": len(X_train),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "feature_importances": self.get_feature_importances(),
        }
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用随机森林模型进行预测

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        获取特征重要性

        Returns:
            Dict: {特征名: 重要性分数}，按重要性降序排列
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        importances = self.model.feature_importances_
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, importances))
        else:
            importance_dict = {f"feature_{i}": float(v) for i, v in enumerate(importances)}

        # 按重要性降序排列
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
