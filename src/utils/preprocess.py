"""
数据预处理工具模块

提供常用的数据预处理功能，供各算法模块使用。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    划分训练集和测试集

    Args:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例，默认 0.2
        random_state: 随机种子，默认 42

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def standard_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    标准化数据（均值为0，方差为1）

    Args:
        X_train: 训练特征
        X_test: 测试特征

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def minmax_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    归一化数据到指定范围

    Args:
        X_train: 训练特征
        X_test: 测试特征
        feature_range: 目标范围，默认 (0, 1)

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    将分类标签编码为数值

    Args:
        y: 原始标签

    Returns:
        编码后的标签, 编码器
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean'
) -> pd.DataFrame:
    """
    处理缺失值

    Args:
        df: 输入数据
        strategy: 填充策略，可选 'mean', 'median', 'mode', 'drop'

    Returns:
        处理后的数据
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
