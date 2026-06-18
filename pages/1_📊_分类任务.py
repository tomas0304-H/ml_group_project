"""
分类任务页面

展示 SVM、KNN 分类算法的结果。
支持内置 ADMET 数据集和自定义数据上传（CSV/XLSX）。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification import (
    SVMClassifier, KNNClassifier,
    run_admet_pipeline, smiles_to_features,
    DEFAULT_DATA_PATH, LABEL_COLS, LABEL_DESC,
    get_feature_names
)
from src.utils.preprocess import split_data, standard_scale
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_metrics_comparison

# 页面配置
st.set_page_config(page_title="分类任务", page_icon="📊", layout="wide")

st.title("📊 分类任务")
st.markdown("---")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")

    st.subheader("📊 数据来源")
    data_source = st.radio("选择数据来源", ["内置数据集", "上传文件"], key="data_source")

    if data_source == "内置数据集":
        st.info("ADMET 药物属性数据集\n\n预测分子的5个ADMET属性")
        task_type = st.radio("分类任务类型", ["多标签分类", "单标签分类"], key="task_type")
        if task_type == "单标签分类":
            selected_label = st.selectbox("选择标签", LABEL_COLS, key="selected_label")
    else:
        uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx", "xls"], key="file_uploader")
        task_type = "单标签分类"  # 上传数据默认单标签

    st.markdown("---")

    # 算法选择
    st.subheader("🤖 算法选择")
    if data_source == "内置数据集" and task_type == "多标签分类":
        algorithm = st.selectbox("选择算法", ["BR+随机森林", "ML-kNN"], index=0)
    else:
        algorithm = st.selectbox("选择算法", ["SVM", "KNN"], index=0)

    st.markdown("---")

    # 算法参数
    st.subheader("⚙️ 算法参数")

    if algorithm == "SVM":
        kernel = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"], index=0)
        C = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1)
    elif algorithm == "KNN":
        n_neighbors = st.slider("邻居数量 K", 1, 20, 5, 1)
        weights = st.selectbox("权重", ["uniform", "distance"], index=0)
    elif algorithm == "BR+随机森林":
        rf_estimators = st.slider("树的数量", 50, 300, 100, 50)
    elif algorithm == "ML-kNN":
        mlknn_k = st.slider("近邻数 K", 3, 15, 5, 2)

    test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据预览", "🚀 模型训练", "📈 结果展示"])

# ============== 数据预览 ==============
with tab1:
    st.subheader("数据预览")

    if data_source == "内置数据集":
        # 内置 ADMET 数据集
        if task_type == "多标签分类":
            st.info("当前选择：**ADMET 药物属性数据集**（多标签分类）")

            if st.button("加载数据", type="primary", key="load_admet"):
                with st.spinner("正在加载 ADMET 数据集..."):
                    try:
                        df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name="training")
                        st.session_state['classification_data'] = df
                        st.session_state['classification_mode'] = 'multilabel'

                        st.success(f"✅ 数据加载成功！共 {len(df)} 条记录")

                        # 显示数据基本信息
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("样本数", df.shape[0])
                        with col2:
                            st.metric("特征列", "SMILES (分子结构)")
                        with col3:
                            st.metric("标签数", len(LABEL_COLS))

                        st.markdown("---")
                        st.subheader("数据预览")
                        st.dataframe(df.head(10))

                        # 显示标签分布
                        with st.expander("📊 标签分布"):
                            for col in LABEL_COLS:
                                counts = df[col].value_counts()
                                st.write(f"**{col}**: {LABEL_DESC.get(col, '')}")
                                st.write(f"  - 0: {counts.get(0, 0)} 条, 1: {counts.get(1, 0)} 条")

                    except Exception as e:
                        st.error(f"❌ 数据加载失败：{str(e)}")
        else:
            # 单标签分类模式
            st.info(f"当前选择：**ADMET 数据集**（单标签分类 - {selected_label}）")

            if st.button("加载数据", type="primary", key="load_admet_single"):
                with st.spinner("正在加载 ADMET 数据集..."):
                    try:
                        df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name="training")

                        # 提取分子特征
                        features_list = []
                        valid_indices = []
                        for idx, smi in enumerate(df['SMILES']):
                            feat = smiles_to_features(smi)
                            if feat is not None:
                                features_list.append(feat)
                                valid_indices.append(idx)

                        df_valid = df.iloc[valid_indices].reset_index(drop=True)
                        feature_names = get_feature_names()

                        # 构建特征 DataFrame
                        df_features = pd.DataFrame(features_list, columns=feature_names)
                        df_features[selected_label] = df_valid[selected_label].values

                        st.session_state['classification_data'] = df_features
                        st.session_state['classification_mode'] = 'singlelabel'
                        st.session_state['classification_target'] = selected_label
                        st.session_state['classification_feature_names'] = feature_names

                        st.success(f"✅ 数据加载成功！共 {len(df_features)} 条有效记录")

                        # 显示数据基本信息
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("样本数", len(df_features))
                        with col2:
                            st.metric("特征数", len(feature_names))
                        with col3:
                            st.metric("目标列", selected_label)

                        st.markdown("---")
                        st.subheader("数据预览")
                        st.dataframe(df_features.head(10))

                        # 显示标签分布
                        with st.expander("📊 标签分布"):
                            counts = df_features[selected_label].value_counts()
                            st.write(f"**{selected_label}**: {LABEL_DESC.get(selected_label, '')}")
                            st.write(f"  - 0: {counts.get(0, 0)} 条, 1: {counts.get(1, 0)} 条")

                    except Exception as e:
                        st.error(f"❌ 数据加载失败：{str(e)}")

    else:
        # 上传文件模式
        if uploaded_file is not None:
            try:
                # 根据文件类型读取
                file_ext = uploaded_file.name.split('.')[-1].lower()
                if file_ext == 'csv':
                    # 尝试多种编码
                    df = None
                    for encoding in ['utf-8', 'ISO-8859-1', 'gbk', 'gb2312']:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    if df is None:
                        st.error("❌ 无法识别文件编码，请检查文件格式。")
                        st.stop()
                elif file_ext in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("❌ 不支持的文件格式，请上传 CSV 或 XLSX 文件。")
                    st.stop()

                st.session_state['classification_data'] = df
                st.session_state['classification_mode'] = 'singlelabel'

                st.success(f"✅ 文件加载成功！共 {len(df)} 条记录，{len(df.columns)} 列")
                st.dataframe(df.head(10))

                # 显示列信息
                with st.expander("📊 列信息"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    st.write(f"**数值列 ({len(numeric_cols)})**: {numeric_cols}")
                    st.write(f"**分类列 ({len(categorical_cols)})**: {categorical_cols}")

            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")
        else:
            st.info("请在左侧边栏上传数据文件（支持 CSV、XLSX 格式）。")

    # 显示加载信息
    if 'classification_data' in st.session_state:
        st.markdown("---")
        st.subheader("📋 数据加载信息")
        df = st.session_state['classification_data']
        mode = st.session_state.get('classification_mode', 'singlelabel')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("样本数", df.shape[0])
        with col2:
            st.metric("列数", df.shape[1])
        with col3:
            st.metric("分类模式", "多标签" if mode == 'multilabel' else "单标签")

# ============== 模型训练 ==============
with tab2:
    st.subheader("模型训练")

    if 'classification_data' not in st.session_state:
        st.warning("⚠️ 请先在「数据预览」标签页加载数据。")
    else:
        df = st.session_state['classification_data']
        mode = st.session_state.get('classification_mode', 'singlelabel')

        if mode == 'multilabel':
            # 多标签分类（ADMET 专用）
            st.info("📊 多标签分类模式：使用分子特征预测5个ADMET属性")

            if algorithm == "BR+随机森林":
                st.markdown(f"**参数配置：** 树数量={rf_estimators}, 测试集比例={test_size}")
            else:
                st.markdown(f"**参数配置：** 近邻数={mlknn_k}, 测试集比例={test_size}")

            if st.button("开始训练", type="primary", key="train_multilabel"):
                with st.spinner("正在运行多标签分类流程..."):
                    try:
                        result = run_admet_pipeline(
                            data_path=DEFAULT_DATA_PATH,
                            test_size=test_size,
                            rf_estimators=rf_estimators if algorithm == "BR+随机森林" else 100,
                            mlknn_k=mlknn_k if algorithm == "ML-kNN" else 5,
                            save_model=True,
                            save_results=True
                        )

                        st.session_state['multilabel_result'] = result

                        st.success("✅ 多标签分类完成！")

                        # 显示训练信息
                        info = result['train_info']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("训练样本", info['train_samples'])
                        with col2:
                            st.metric("验证样本", info['val_samples'])
                        with col3:
                            st.metric("测试样本", info['test_samples'])

                        # 显示模型性能
                        st.markdown("---")
                        st.subheader("📊 模型性能")

                        for model_name, model_result in result['model_results'].items():
                            st.markdown(f"**{model_name}**")
                            metrics = model_result['overall_metrics']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{model_name} AUC-ROC", f"{metrics['平均AUC-ROC']:.4f}")
                            with col2:
                                st.metric(f"{model_name} 汉明损失", f"{metrics['汉明损失']:.4f}")
                            with col3:
                                st.metric(f"{model_name} 宏平均F1", f"{metrics['宏平均F1']:.4f}")

                    except Exception as e:
                        st.error(f"❌ 训练失败：{str(e)}")

        else:
            # 单标签分类
            st.info("📊 单标签分类模式")

            columns = df.columns.tolist()
            default_target = st.session_state.get('classification_target', columns[-1])

            col1, col2 = st.columns(2)
            with col1:
                feature_cols = st.multiselect("选择特征列", columns,
                                               default=[c for c in columns if c != default_target],
                                               key="feature_cols")
            with col2:
                label_col = st.selectbox("选择标签列", columns,
                                         index=columns.index(default_target) if default_target in columns else len(columns)-1,
                                         key="label_col")

            if st.button("开始训练", type="primary", key="train_singlelabel"):
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
                        st.session_state['classification_label_col'] = label_col

                        st.success("✅ 模型训练完成！")

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

                    except Exception as e:
                        st.error(f"❌ 训练失败：{str(e)}")

# ============== 结果展示 ==============
with tab3:
    st.subheader("结果展示")

    if 'multilabel_result' in st.session_state:
        # 多标签分类结果
        result = st.session_state['multilabel_result']

        st.markdown("### 📊 多标签分类结果")

        # 显示图表
        if 'figures' in result:
            st.subheader("单标签指标对比")
            st.pyplot(result['figures']['fig1'])

            st.subheader("整体指标对比")
            st.pyplot(result['figures']['fig2'])

        # 显示测试集预测结果
        if 'test_results' in result:
            st.subheader("测试集预测结果")
            st.dataframe(result['test_results'].head(20))

    elif 'classification_metrics' in st.session_state:
        # 单标签分类结果
        metrics = st.session_state['classification_metrics']
        model = st.session_state['classification_model']
        X_test, y_test = st.session_state['classification_test_data']
        label_col = st.session_state.get('classification_label_col', '标签')

        st.markdown(f"### 📊 单标签分类结果（{label_col}）")

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
    else:
        st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
