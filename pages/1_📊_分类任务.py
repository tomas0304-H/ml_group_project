"""
分类任务页面

展示 SVM 和 KNN 分类算法的结果。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification import SVMClassifier, KNNClassifier
from src.utils.preprocess import split_data, standard_scale
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_metrics_comparison

st.set_page_config(page_title="分类任务", page_icon="📊", layout="wide")

st.title("📊 分类任务")
st.markdown("---")

# 侧边栏 - 算法选择
with st.sidebar:
    st.header("⚙️ 设置")
    algorithm = st.selectbox(
        "选择算法",
        ["SVM", "KNN"],
        index=0
    )

    st.markdown("---")
    st.subheader("算法参数")

    if algorithm == "SVM":
        kernel = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"], index=0)
        C = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1)
    else:  # KNN
        n_neighbors = st.slider("邻居数量 K", 1, 20, 5, 1)
        weights = st.selectbox("权重", ["uniform", "distance"], index=0)

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据上传", "🚀 模型训练", "📈 结果展示"])

with tab1:
    st.subheader("上传数据集")
    st.info("请上传 CSV 格式的数据文件，最后一列应为分类标签。")

    uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['classification_data'] = df
            st.success(f"✅ 数据加载成功！共 {len(df)} 条记录，{len(df.columns)} 个特征。")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"❌ 数据加载失败：{str(e)}")

with tab2:
    st.subheader("模型训练")

    if 'classification_data' not in st.session_state:
        st.warning("⚠️ 请先在「数据上传」标签页上传数据。")
    else:
        df = st.session_state['classification_data']

        # 选择特征和标签列
        columns = df.columns.tolist()
        feature_cols = st.multiselect("选择特征列", columns, default=columns[:-1])
        label_col = st.selectbox("选择标签列", columns, index=len(columns)-1)

        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)

        if st.button("开始训练", type="primary"):
            with st.spinner("正在训练模型..."):
                try:
                    # 准备数据
                    X = df[feature_cols]
                    y = df[label_col]

                    # 划分数据集
                    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

                    # 标准化
                    X_train_scaled, X_test_scaled, scaler = standard_scale(X_train, X_test)

                    # 创建并训练模型
                    if algorithm == "SVM":
                        model = SVMClassifier(kernel=kernel, C=C)
                    else:
                        model = KNNClassifier(n_neighbors=n_neighbors, weights=weights)

                    train_result = model.train(X_train_scaled, y_train)

                    # 评估模型
                    metrics = model.evaluate(X_test_scaled, y_test)

                    # 保存到 session_state
                    st.session_state['classification_model'] = model
                    st.session_state['classification_metrics'] = metrics
                    st.session_state['classification_test_data'] = (X_test_scaled, y_test)
                    st.session_state['classification_train_result'] = train_result

                    st.success("✅ 模型训练完成！")
                    st.json(train_result)

                except Exception as e:
                    st.error(f"❌ 训练失败：{str(e)}")

with tab3:
    st.subheader("结果展示")

    if 'classification_metrics' not in st.session_state:
        st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
    else:
        metrics = st.session_state['classification_metrics']
        model = st.session_state['classification_model']
        X_test, y_test = st.session_state['classification_test_data']

        # 显示指标
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

        st.markdown("---")

        # 可视化
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("混淆矩阵")
            y_pred = model.predict(X_test)
            fig_cm = plot_confusion_matrix(y_test.values, y_pred)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.subheader("ROC 曲线")
            y_proba = model.predict_proba(X_test)
            if y_proba is not None:
                if y_proba.ndim == 2:
                    y_proba_plot = y_proba[:, 1]
                else:
                    y_proba_plot = y_proba
                fig_roc = plot_roc_curve(y_test.values, y_proba_plot)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("该模型不支持概率预测，无法绘制 ROC 曲线。")
