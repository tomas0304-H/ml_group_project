"""
回归任务模块

包含各种回归算法的实现。
所有回归算法必须继承 BaseRegressor 基类。
"""
from .base import BaseRegressor
from .linear import LinearRegressor
from .random_forest import (
    RandomForestRegressor,
    load_data,
    preprocess,
    train_model,
    plot_feature_importance,
    plot_prediction_vs_actual,
    plot_residuals,
    DEFAULT_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_TARGET_COL,
    DEFAULT_DROP_COLS,
    DEFAULT_CATEGORICAL_COLS
)
