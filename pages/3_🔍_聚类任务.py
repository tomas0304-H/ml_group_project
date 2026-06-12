"""
聚类任务页面

展示 K-Means 聚类算法的结果。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import KMeansClusterer
from src.utils.visualization import plot_scatter_2d
from sklearn.decomposition import PCA

st.set_page_config(page_title="聚类任务", page_icon="🔍", layout="wide")

st.title("🔍 聚类任务")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    n_clusters = st.slider("聚类数量 K", 2, 10, 3, 1)
    max_iter = st.slider("最大迭代次数", 100, 1000, 300, 50)

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据上传", "🚀 模型训练", "📈 结果展示"])

with tab1:
    st.subheader("上传数据集")
    st.info("请上传 CSV 格式的数据文件，聚类任务不需要标签列。")

    uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # 只保留数值列
            numeric_df = df.select_dtypes(include=[np.number])
            st.session_state['clustering_data'] = numeric_df
            st.success(f"✅ 数据加载成功！共 {len(numeric_df)} 条记录，{len(numeric_df.columns)} 个数值特征。")
            st.dataframe(numeric_df.head(10))
        except Exception as e:
            st.error(f"❌ 数据加载失败：{str(e)}")

with tab2:
    st.subheader("模型训练")

    if 'clustering_data' not in st.session_state:
        st.warning("⚠️ 请先在「数据上传」标签页上传数据。")
    else:
        df = st.session_state['clustering_data']
        columns = df.columns.tolist()

        feature_cols = st.multiselect("选择特征列", columns, default=columns)

        if st.button("开始训练", type="primary"):
            with st.spinner("正在训练模型..."):
                try:
                    X = df[feature_cols]

                    model = KMeansClusterer(
                        n_clusters=n_clusters,
                        max_iter=max_iter
                    )
                    train_result = model.train(X)

                    metrics = model.evaluate(X)

                    st.session_state['clustering_model'] = model
                    st.session_state['clustering_metrics'] = metrics
                    st.session_state['clustering_data_selected'] = X
                    st.session_state['clustering_train_result'] = train_result

                    st.success("✅ 模型训练完成！")
                    st.json(train_result)

                except Exception as e:
                    st.error(f"❌ 训练失败：{str(e)}")

with tab3:
    st.subheader("结果展示")

    if 'clustering_metrics' not in st.session_state:
        st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
    else:
        metrics = st.session_state['clustering_metrics']
        model = st.session_state['clustering_model']
        X = st.session_state['clustering_data_selected']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("轮廓系数", f"{metrics['silhouette']:.4f}")
        with col2:
            st.metric("CH指数", f"{metrics['calinski_harabasz']:.2f}")
        with col3:
            st.metric("DB指数", f"{metrics['davies_bouldin']:.4f}")

        st.markdown("---")

        st.subheader("聚类结果可视化")

        # 使用 PCA 降维到 2D 进行可视化
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            feature_names = ["主成分1", "主成分2"]
        else:
            X_2d = X.values
            feature_names = X.columns.tolist()

        labels = model.predict(X)
        fig = plot_scatter_2d(X_2d, labels, feature_names=feature_names)
        st.plotly_chart(fig, use_container_width=True)

        # 显示聚类中心
        st.subheader("聚类中心")
        centers = model.get_cluster_centers()
        centers_df = pd.DataFrame(centers, columns=X.columns)
        centers_df.index.name = "聚类"
        st.dataframe(centers_df)
