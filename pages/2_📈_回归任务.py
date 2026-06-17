"""
回归任务页面

展示线性回归和随机森林回归算法的结果。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression import (
    LinearRegressor,
    load_data, preprocess, train_model,
    plot_feature_importance, plot_prediction_vs_actual, plot_residuals,
    DEFAULT_CSV, DEFAULT_TARGET_COL, DEFAULT_DROP_COLS, DEFAULT_CATEGORICAL_COLS
)

# 页面配置
st.set_page_config(page_title="回归任务", page_icon="📈", layout="wide")

st.title("📈 回归任务")
st.markdown("---")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")

    st.subheader("📊 数据来源")
    data_source = st.radio("选择数据来源", ["内置数据集", "上传CSV"], key="data_source")

    if data_source == "内置数据集":
        st.info("SeoulBikeData 首尔自行车租赁数据集")
    else:
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"], key="csv_uploader")

    st.markdown("---")

    model_type = st.radio("选择模型", ["线性回归", "随机森林"], key="model_type")

    if model_type == "随机森林":
        st.subheader("🌲 随机森林参数")
        n_estimators = st.slider("树的数量", 50, 500, 200, 50)
        max_depth = st.slider("最大深度", 3, 30, 15, 1)
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据预览", "🚀 模型训练", "📈 结果展示"])

# ============== 数据预览 ==============
with tab1:
    st.subheader("数据预览")

    if data_source == "内置数据集":
        # 内置 SeoulBikeData 数据集
        if model_type == "线性回归":
            st.warning("⚠️ 线性回归请选择「上传CSV」并上传您自己的数据集。")
        else:
            st.info(f"当前选择：**SeoulBikeData** 首尔自行车租赁数据集")
            if st.button("加载数据", type="primary"):
                with st.spinner("正在加载数据..."):
                    try:
                        df = load_data(DEFAULT_CSV)
                        st.session_state['rf_df'] = df

                        # 自动预处理
                        preprocess_result = preprocess(df)
                        st.session_state['rf_preprocess_result'] = preprocess_result
                        st.session_state['uploaded_df'] = df  # 标记已有数据

                        st.success(f"✅ 数据加载并预处理完成！共 {len(df)} 条记录")
                        st.dataframe(df.head(10))

                        # 显示数据基本信息
                        with st.expander("📊 数据信息"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("样本数", df.shape[0])
                            with col2:
                                st.metric("特征数", preprocess_result['X'].shape[1])
                            with col3:
                                st.metric("目标列", DEFAULT_TARGET_COL)

                    except Exception as e:
                        st.error(f"❌ 数据加载失败：{str(e)}")
    else:
        # 上传 CSV 文件
        if 'uploaded_file' not in st.session_state:
            st.session_state['uploaded_file'] = None

        if uploaded_file is not None:
            try:
                # 尝试多种编码读取
                df = None
                for encoding in ['utf-8', 'ISO-8859-1', 'gbk', 'gb2312']:
                    try:
                        uploaded_file.seek(0)  # 重置文件指针
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue

                if df is None:
                    st.error("❌ 无法识别文件编码，请检查文件格式。")
                else:
                    st.session_state['uploaded_df'] = df
                    st.success(f"✅ 文件加载成功！共 {len(df)} 条记录，{len(df.columns)} 列")
                    st.dataframe(df.head(10))

                    # 显示列信息
                    with st.expander("📊 列信息"):
                        st.write("**数值列：**", df.select_dtypes(include=[np.number]).columns.tolist())
                        st.write("**分类列：**", df.select_dtypes(include=['object']).columns.tolist())

            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")

        if 'uploaded_df' in st.session_state:
            df = st.session_state['uploaded_df']

            st.markdown("---")
            st.subheader("配置预处理参数")

            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("选择目标列（预测目标）",
                                          df.columns.tolist(),
                                          index=df.columns.tolist().index(DEFAULT_TARGET_COL)
                                          if DEFAULT_TARGET_COL in df.columns.tolist() else 0)
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                drop_cols = st.multiselect("选择要删除的列",
                                           df.columns.tolist(),
                                           default=[c for c in DEFAULT_DROP_COLS if c in df.columns.tolist()])

            # 分类特征选择
            categorical_cols = st.multiselect("选择分类特征列（将进行One-Hot编码）",
                                              df.select_dtypes(include=['object']).columns.tolist(),
                                              default=[c for c in DEFAULT_CATEGORICAL_COLS
                                                       if c in df.select_dtypes(include=['object']).columns.tolist()])

            if st.button("预处理数据", type="primary"):
                with st.spinner("正在预处理数据..."):
                    try:
                        result = preprocess(df, target_col=target_col, drop_cols=drop_cols,
                                            categorical_cols=categorical_cols,
                                            filter_functioning=False)

                        st.session_state['rf_df'] = df
                        st.session_state['rf_preprocess_result'] = result
                        st.session_state['uploaded_config'] = {
                            'target_col': target_col,
                            'drop_cols': drop_cols,
                            'categorical_cols': categorical_cols
                        }

                        st.success(f"✅ 预处理完成！特征数：{result['X'].shape[1]}，样本数：{result['X'].shape[0]}")

                        # 显示预处理后的数据
                        with st.expander("📊 预处理后数据"):
                            st.dataframe(result['df_processed'].head(10))

                    except Exception as e:
                        st.error(f"❌ 预处理失败：{str(e)}")
        else:
            st.info("请在左侧边栏上传 CSV 文件。")

# ============== 模型训练 ==============
with tab2:
    st.subheader("模型训练")

    # 检查是否有数据
    if 'uploaded_df' not in st.session_state:
        st.warning("⚠️ 请先在「数据预览」标签页加载或上传数据。")
    else:
        df = st.session_state['uploaded_df']
        preprocess_result = st.session_state.get('rf_preprocess_result', None)
        config = st.session_state.get('uploaded_config', {})

        if model_type == "线性回归":
            # 线性回归训练逻辑
            st.info("📊 使用线性回归模型")

            if data_source == "内置数据集":
                st.warning("⚠️ 线性回归仅支持上传CSV数据。请在侧边栏切换数据来源。")
            elif preprocess_result is None:
                st.warning("⚠️ 请先在「数据预览」标签页点击「预处理数据」按钮。")
            else:
                X = preprocess_result['X']
                y = preprocess_result['y']
                feature_names = preprocess_result['feature_names']

                if st.button("开始训练", type="primary", key="train_linear"):
                    with st.spinner("正在训练线性回归模型..."):
                        try:
                            model = LinearRegressor()
                            model.fit(X.values, y.values)

                            y_pred = model.predict(X.values)

                            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

                            r2 = r2_score(y, y_pred)
                            mae = mean_absolute_error(y, y_pred)
                            mse = mean_squared_error(y, y_pred)
                            rmse = np.sqrt(mse)

                            st.session_state['linear_model'] = model
                            st.session_state['linear_metrics'] = {
                                'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse
                            }
                            st.session_state['linear_predictions'] = {
                                'y_true': y.values,
                                'y_pred': y_pred
                            }
                            st.session_state['linear_feature_names'] = feature_names

                            st.success("✅ 模型训练完成！")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² 分数", f"{r2:.4f}")
                            with col2:
                                st.metric("MAE", f"{mae:.2f}")
                            with col3:
                                st.metric("RMSE", f"{rmse:.2f}")

                        except Exception as e:
                            st.error(f"❌ 训练失败：{str(e)}")
        else:
            # 随机森林训练逻辑
            st.info("📊 使用随机森林回归模型")

            if preprocess_result is None:
                st.warning("⚠️ 请先在「数据预览」标签页点击「预处理数据」按钮。")
            else:
                X = preprocess_result['X']
                y = preprocess_result['y']
                feature_names = preprocess_result['feature_names']

                st.markdown(f"**参数配置：** 树数量={n_estimators}, 最大深度={max_depth}, 测试集比例={test_size}")

                if st.button("开始训练", type="primary", key="train_rf"):
                    with st.spinner("正在训练随机森林模型..."):
                        try:
                            result = train_model(
                                X, y,
                                test_size=test_size,
                                n_estimators=n_estimators,
                                max_depth=max_depth
                            )

                            # 生成预测
                            model = result['model']
                            y_pred = model.predict(result['X_test'])
                            y_test = result['y_test']

                            st.session_state['rf_model'] = model
                            st.session_state['rf_metrics'] = result['metrics']
                            st.session_state['rf_predictions'] = {
                                'y_true': y_test.values,
                                'y_pred': y_pred
                            }
                            st.session_state['rf_feature_names'] = feature_names

                            st.success("✅ 模型训练完成！")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² 分数", f"{result['metrics']['r2']:.4f}")
                            with col2:
                                st.metric("MAE", f"{result['metrics']['mae']:.2f}")
                            with col3:
                                st.metric("RMSE", f"{result['metrics']['rmse']:.2f}")

                        except Exception as e:
                            st.error(f"❌ 训练失败：{str(e)}")

# ============== 结果展示 ==============
with tab3:
    st.subheader("结果展示")

    if model_type == "线性回归":
        if 'linear_metrics' not in st.session_state:
            st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
        else:
            metrics = st.session_state['linear_metrics']
            predictions = st.session_state['linear_predictions']

            st.markdown("### 📊 线性回归评估指标")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² 分数", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.2f}")
            with col3:
                st.metric("MSE", f"{metrics['mse']:.2f}")
            with col4:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")

            st.markdown("---")

            # 预测 vs 真实值散点图
            st.subheader("预测值 vs 真实值")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(predictions['y_true'], predictions['y_pred'], alpha=0.5, s=10)
            ax.plot([predictions['y_true'].min(), predictions['y_true'].max()],
                    [predictions['y_true'].min(), predictions['y_true'].max()],
                    'r--', lw=2, label='理想预测线')
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title('预测值 vs 真实值')
            ax.legend()
            st.pyplot(fig)

            # 残差图
            st.subheader("残差分布")
            residuals = predictions['y_true'] - predictions['y_pred']
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(predictions['y_pred'], residuals, alpha=0.5, s=10)
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('预测值')
            ax2.set_ylabel('残差')
            ax2.set_title('残差图')
            st.pyplot(fig2)

    else:  # 随机森林
        required_keys = ['rf_metrics', 'rf_predictions', 'rf_model', 'rf_feature_names']
        if not all(key in st.session_state for key in required_keys):
            st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
        else:
            metrics = st.session_state['rf_metrics']
            predictions = st.session_state['rf_predictions']
            model = st.session_state['rf_model']
            feature_names = st.session_state['rf_feature_names']

            st.markdown("### 📊 随机森林评估指标")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² 分数", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.2f}")
            with col3:
                st.metric("MSE", f"{metrics['mse']:.2f}")
            with col4:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")

            st.markdown("---")

            # 创建子标签页
            result_tab1, result_tab2, result_tab3 = st.tabs([
                "🎯 预测效果", "📊 特征重要性", "📉 残差分析"
            ])

            with result_tab1:
                st.subheader("预测值 vs 真实值")
                fig_pred = plot_prediction_vs_actual(
                    predictions['y_true'],
                    predictions['y_pred']
                )
                st.pyplot(fig_pred)

            with result_tab2:
                st.subheader("特征重要性排序")
                importances = model.get_feature_importances()
                fig_importance = plot_feature_importance(importances)
                st.pyplot(fig_importance)

            with result_tab3:
                st.subheader("残差分析")
                fig_residual = plot_residuals(
                    predictions['y_true'],
                    predictions['y_pred']
                )
                st.pyplot(fig_residual)
