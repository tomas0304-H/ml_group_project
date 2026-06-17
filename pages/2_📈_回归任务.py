"""
回归任务页面

展示线性回归和随机森林回归算法的结果。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression import LinearRegressor
from src.regression.random_forest import (
    load_data as rf_load_data,
    preprocess as rf_preprocess,
    train_model as rf_train_model,
    plot_feature_importance,
    plot_prediction_vs_actual as rf_plot_prediction,
    plot_residuals,
    DEFAULT_CSV,
)
from src.utils.preprocess import split_data, standard_scale
from src.utils.visualization import plot_prediction_vs_actual

st.set_page_config(page_title="回归任务", page_icon="📈", layout="wide")

st.title("📈 回归任务")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

    st.subheader("选择模型")
    model_type = st.radio(
        "回归算法",
        ["线性回归", "随机森林回归"],
        label_visibility="collapsed"
    )

    if model_type == "线性回归":
        use_ridge = st.checkbox("使用 Ridge 回归", value=False)
        if use_ridge:
            alpha = st.slider("正则化参数 alpha", 0.01, 10.0, 1.0, 0.01)
        else:
            alpha = 1.0
    else:
        st.subheader("随机森林参数")
        n_estimators = st.slider("树的数量", 50, 500, 200, 50)
        max_depth = st.slider("最大深度", 5, 30, 15, 1)
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据上传", "🚀 模型训练", "📈 结果展示"])

with tab1:
    st.subheader("上传数据集")

    if model_type == "线性回归":
        st.info("请上传 CSV 格式的数据文件，最后一列应为目标值（连续变量）。")
        uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"], key="linear_upload")
    else:
        st.info("请上传 SeoulBikeData.csv，或使用内置数据集。")
        use_builtin = st.checkbox("使用内置数据集", value=True)

        if use_builtin:
            uploaded_file = None
            if os.path.exists(DEFAULT_CSV):
                st.success(f"✅ 找到内置数据集: {DEFAULT_CSV}")
            else:
                st.error(f"❌ 内置数据集不存在: {DEFAULT_CSV}")
        else:
            uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"], key="rf_upload")

    # 加载数据
    if model_type == "线性回归" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['regression_data'] = df
            st.session_state['regression_model_type'] = 'linear'
            st.success(f"✅ 数据加载成功！共 {len(df)} 条记录，{len(df.columns)} 个特征。")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"❌ 数据加载失败：{str(e)}")

    elif model_type == "随机森林回归":
        if use_builtin and os.path.exists(DEFAULT_CSV):
            try:
                df = rf_load_data()
                st.session_state['regression_data'] = df
                st.session_state['regression_model_type'] = 'random_forest'
                st.success(f"✅ 内置数据加载成功！共 {len(df)} 条记录，{len(df.columns)} 个特征。")
                st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"❌ 数据加载失败：{str(e)}")
        elif uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['regression_data'] = df
                st.session_state['regression_model_type'] = 'random_forest'
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
        model_type = st.session_state.get('regression_model_type', 'linear')

        if model_type == 'linear':
            # 线性回归训练流程
            columns = df.columns.tolist()
            feature_cols = st.multiselect("选择特征列", columns, default=columns[:-1])
            label_col = st.selectbox("选择目标列", columns, index=len(columns)-1)

            if st.button("开始训练", type="primary"):
                with st.spinner("正在训练模型..."):
                    try:
                        X = df[feature_cols]
                        y = df[label_col]

                        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
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

        else:
            # 随机森林训练流程
            st.info("随机森林将使用 SeoulBikeData 数据集，自动进行预处理。")

            if st.button("开始训练", type="primary"):
                with st.spinner("正在预处理数据并训练模型..."):
                    try:
                        # 预处理
                        preprocessed = rf_preprocess(df)
                        X = preprocessed['X']
                        y = preprocessed['y']

                        # 训练
                        result = rf_train_model(
                            X, y,
                            test_size=test_size,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                        )

                        st.session_state['rf_result'] = result
                        st.session_state['regression_metrics'] = result['metrics']
                        st.session_state['regression_model_type'] = 'random_forest'

                        st.success("✅ 随机森林训练完成！")

                        # 显示训练信息
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("训练样本", result['train_info']['train_samples'])
                        with col2:
                            st.metric("树的数量", result['train_info']['n_estimators'])
                        with col3:
                            st.metric("训练耗时", f"{result['train_info']['train_time']:.2f}s")

                        # 显示特征重要性 Top 5
                        st.subheader("特征重要性 Top 5")
                        importances = result['train_info']['feature_importances']
                        for i, (feat, score) in enumerate(list(importances.items())[:5]):
                            st.write(f"{i+1}. **{feat}**: {score:.4f}")

                    except Exception as e:
                        st.error(f"❌ 训练失败：{str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

with tab3:
    st.subheader("结果展示")

    if 'regression_metrics' not in st.session_state:
        st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
    else:
        metrics = st.session_state['regression_metrics']
        model_type = st.session_state.get('regression_model_type', 'linear')

        # 指标展示
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

        if model_type == 'linear':
            # 线性回归结果展示
            model = st.session_state['regression_model']
            X_test, y_test = st.session_state['regression_test_data']

            st.subheader("真实值 vs 预测值")
            y_pred = model.predict(X_test)
            fig = plot_prediction_vs_actual(y_test.values, y_pred)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # 随机森林结果展示
            result = st.session_state['rf_result']
            model = result['model']
            y_test = result['y_test']
            y_pred = model.predict(result['X_test'])

            # 创建标签页展示不同图表
            fig_tab1, fig_tab2, fig_tab3 = st.tabs(["📊 特征重要性", "📈 预测散点图", "📉 残差分布"])

            with fig_tab1:
                importances = result['train_info']['feature_importances']
                fig_imp = plot_feature_importance(importances, top_n=15)
                st.pyplot(fig_imp)

            with fig_tab2:
                fig_pred = rf_plot_prediction(y_test.values, y_pred)
                st.pyplot(fig_pred)

            with fig_tab3:
                fig_res = plot_residuals(y_test.values, y_pred)
                st.pyplot(fig_res)

            # 特征重要性完整列表
            with st.expander("查看完整特征重要性"):
                importances = result['train_info']['feature_importances']
                imp_df = pd.DataFrame([
                    {"特征": k, "重要性": v}
                    for k, v in importances.items()
                ])
                st.dataframe(imp_df, use_container_width=True)
