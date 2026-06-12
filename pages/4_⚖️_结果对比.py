"""
结果对比页面

对比各任务中不同算法的结果。
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import plot_metrics_comparison

st.set_page_config(page_title="结果对比", page_icon="⚖️", layout="wide")

st.title("⚖️ 结果对比")
st.markdown("---")

st.markdown("""
本页面用于对比各任务中不同算法的评估结果。
请先在各任务页面完成模型训练，然后返回此页面查看对比结果。
""")

# 分类任务对比
st.header("📊 分类任务对比")
st.markdown("对比 SVM 和 KNN 在同一数据集上的分类效果。")

classification_metrics = {}
if 'classification_metrics' in st.session_state:
    # 这里简化处理，实际应该保存多个模型的结果
    st.info("💡 提示：请分别使用 SVM 和 KNN 训练模型，然后在此页面查看对比结果。")

    metrics = st.session_state['classification_metrics']
    st.subheader("当前模型指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1", f"{metrics['f1']:.4f}")
    with col5:
        auc_val = metrics.get('auc')
        st.metric("AUC", f"{auc_val:.4f}" if auc_val else "N/A")
else:
    st.warning("⚠️ 尚未完成分类模型训练。")

st.markdown("---")

# 回归任务结果
st.header("📈 回归任务结果")

if 'regression_metrics' in st.session_state:
    metrics = st.session_state['regression_metrics']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"{metrics['mae']:.4f}")
    with col2:
        st.metric("MSE", f"{metrics['mse']:.4f}")
    with col3:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    with col4:
        st.metric("R²", f"{metrics['r2']:.4f}")
else:
    st.warning("⚠️ 尚未完成回归模型训练。")

st.markdown("---")

# 聚类任务结果
st.header("🔍 聚类任务结果")

if 'clustering_metrics' in st.session_state:
    metrics = st.session_state['clustering_metrics']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("轮廓系数", f"{metrics['silhouette']:.4f}")
    with col2:
        st.metric("CH指数", f"{metrics['calinski_harabasz']:.2f}")
    with col3:
        st.metric("DB指数", f"{metrics['davies_bouldin']:.4f}")
else:
    st.warning("⚠️ 尚未完成聚类模型训练。")

st.markdown("---")

# 总结
st.header("📝 总结分析")
st.markdown("""
在此处添加对实验结果的总结分析，包括：

1. **分类任务**：SVM 和 KNN 哪个效果更好？为什么？
2. **回归任务**：线性回归的预测误差如何？
3. **聚类任务**：K-Means 的聚类效果如何？
4. **整体结论**：对本次实验的总结和收获。
""")
