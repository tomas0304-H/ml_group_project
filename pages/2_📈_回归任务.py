"""
回归任务页面

展示线性回归算法的结果。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression import LinearRegressor
from src.utils.preprocess import split_data, standard_scale
from src.utils.visualization import plot_prediction_vs_actual

st.set_page_config(page_title="回归任务", page_icon="📈", layout="wide")

st.title("📈 回归任务")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    use_ridge = st.checkbox("使用 Ridge 回归", value=False)
    if use_ridge:
        alpha = st.slider("正则化参数 alpha", 0.01, 10.0, 1.0, 0.01)
    else:
        alpha = 1.0

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据上传", "🚀 模型训练", "📈 结果展示"])

with tab1:
    st.subheader("上传数据集")
    st.info("请上传 CSV 格式的数据文件，最后一列应为目标值（连续变量）。")

    uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['regression_data'] = df
            st.success(f"✅ 数据加载成功！共 {len(df)} 条记录，{len(df.columns)} 个特征。")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"❌ 数据加载失败：{str(e)}")

with tab2:
    st.subheader("模型训练")

    if 'regression_data' not in st.session_state:
        st.warning("⚠️ 请先在「数据上传」标签页上传数据。")
    else:
        df = st.session_state['regression_data']
        columns = df.columns.tolist()

        feature_cols = st.multiselect("选择特征列", columns, default=columns[:-1])
        label_col = st.selectbox("选择目标列", columns, index=len(columns)-1)
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)

        if st.button("开始训练", type="primary"):
            with st.spinner("正在训练模型..."):
                try:
                    X = df[feature_cols]
                    y = df[label_col]

                    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
                    X_train_scaled, X_test_scaled, scaler = standard_scale(X_train, X_test)

                    model = LinearRegressor(use_ridge=use_ridge, alpha=alpha)
                    train_result = model.train(X_train_scaled, y_train)

                    metrics = model.evaluate(X_test_scaled, y_test)

                    st.session_state['regression_model'] = model
                    st.session_state['regression_metrics'] = metrics
                    st.session_state['regression_test_data'] = (X_test_scaled, y_test)
                    st.session_state['regression_train_result'] = train_result

                    st.success("✅ 模型训练完成！")
                    st.json(train_result)

                except Exception as e:
                    st.error(f"❌ 训练失败：{str(e)}")

with tab3:
    st.subheader("结果展示")

    if 'regression_metrics' not in st.session_state:
        st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
    else:
        metrics = st.session_state['regression_metrics']
        model = st.session_state['regression_model']
        X_test, y_test = st.session_state['regression_test_data']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col2:
            st.metric("MSE", f"{metrics['mse']:.4f}")
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col4:
            st.metric("R²", f"{metrics['r2']:.4f}")

        st.markdown("---")

        st.subheader("真实值 vs 预测值")
        y_pred = model.predict(X_test)
        fig = plot_prediction_vs_actual(y_test.values, y_pred)
        st.plotly_chart(fig, use_container_width=True)
