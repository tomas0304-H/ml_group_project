"""
分类任务模块

包含各种分类算法的实现。
所有分类算法必须继承 BaseClassifier 基类。
"""
from .base import BaseClassifier
from .svm import SVMClassifier
from .knn import KNNClassifier
from .model_run_result import (
    # 核心函数
    smiles_to_features,
    load_and_process_data,
    train_and_evaluate_models,
    plot_all_results,
    predict_test_set,
    run_admet_pipeline,

    # 模型保存/加载
    save_models,
    load_models,

    # 配置常量
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    SMILES_COL,
    LABEL_COLS,
    LABEL_DESC,
    get_feature_names,
)
