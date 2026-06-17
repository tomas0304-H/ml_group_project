"""
可视化工具模块

提供统一的可视化功能，供展示系统和各算法模块使用。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict
from sklearn.metrics import confusion_matrix, roc_curve, auc


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "混淆矩阵"
) -> go.Figure:
    """
    绘制混淆矩阵（Plotly 交互式）

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签名称列表
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    fig = px.imshow(
        cm,
        labels=dict(x="预测标签", y="真实标签", color="数量"),
        x=labels,
        y=labels,
        title=title,
        color_continuous_scale="Blues"
    )
    fig.update_layout(width=500, height=400)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC 曲线"
) -> go.Figure:
    """
    绘制 ROC 曲线（Plotly 交互式）

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.4f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='随机分类器'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='假正率 (FPR)',
        yaxis_title='真正率 (TPR)',
        width=600, height=500
    )
    return fig


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "真实值 vs 预测值"
) -> go.Figure:
    """
    绘制真实值与预测值对比图（Plotly 交互式）

    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        name='预测点',
        marker=dict(color='blue', opacity=0.6)
    ))
    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='理想预测'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='真实值',
        yaxis_title='预测值',
        width=600, height=500
    )
    return fig


def plot_scatter_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "聚类结果散点图",
    feature_names: Optional[List[str]] = None
) -> go.Figure:
    """
    绘制二维聚类散点图（Plotly 交互式）

    Args:
        X: 二维特征数据
        labels: 聚类标签
        title: 图表标题
        feature_names: 特征名称列表

    Returns:
        Plotly Figure 对象
    """
    if feature_names is None:
        feature_names = ["特征1", "特征2"]

    df = pd.DataFrame({
        feature_names[0]: X[:, 0],
        feature_names[1]: X[:, 1],
        "聚类标签": labels.astype(str)
    })

    fig = px.scatter(
        df,
        x=feature_names[0],
        y=feature_names[1],
        color="聚类标签",
        title=title,
        width=600, height=500
    )
    return fig


def plot_feature_importance(
    importances: Dict[str, float],
    top_n: int = 15,
    title: str = "特征重要性"
) -> go.Figure:
    """
    绘制特征重要性柱状图（Plotly 交互式）

    Args:
        importances: {特征名: 重要性分数}，已按重要性降序排列
        top_n: 显示前 N 个特征，默认 15
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    items = list(importances.items())[:top_n]
    features = [k for k, _ in items]
    scores = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=features[::-1],
        orientation='h',
        marker_color='steelblue'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='重要性分数',
        yaxis_title='特征',
        width=700,
        height=max(400, top_n * 30)
    )
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "残差分布"
) -> go.Figure:
    """
    绘制残差直方图（Plotly 交互式）

    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    residuals = y_true - y_pred
    fig = go.Figure(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.8
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(
        title=title,
        xaxis_title='残差 (真实值 - 预测值)',
        yaxis_title='频数',
        width=700,
        height=450
    )
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "模型指标对比"
) -> go.Figure:
    """
    绘制多模型指标对比柱状图

    Args:
        metrics_dict: {模型名: {指标名: 值}}
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    models = list(metrics_dict.keys())
    metric_names = list(metrics_dict[models[0]].keys())

    fig = go.Figure()
    for metric in metric_names:
        values = [metrics_dict[model][metric] for model in models]
        fig.add_trace(go.Bar(name=metric, x=models, y=values))

    fig.update_layout(
        title=title,
        barmode='group',
        width=800, height=500
    )
    return fig
