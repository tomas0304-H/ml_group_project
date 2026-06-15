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

from src.clustering.k_means import (
    load_data, clean_data, select_k, train_model,
    plot_elbow, plot_clusters, get_cluster_summary,
    DEFAULT_FEATURES, DEFAULT_CSV
)

st.set_page_config(page_title="聚类任务", page_icon="🔍", layout="wide")

st.title("🔍 聚类任务")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

    st.subheader("数据来源")
    data_source = st.radio("选择数据来源", ["内置数据集", "上传CSV"], label_visibility="collapsed")

    if data_source == "内置数据集":
        # 获取可用城市列表
        if 'available_cities' not in st.session_state:
            try:
                result = load_data(csv_path=DEFAULT_CSV, target_city="北京", max_samples=100)
                st.session_state['available_cities'] = result['available_cities']
            except Exception as e:
                st.error(f"加载城市列表失败：{e}")
                st.session_state['available_cities'] = ["北京"]

        target_city = st.selectbox("选择城市", st.session_state['available_cities'],
                                   index=st.session_state['available_cities'].index("北京")
                                   if "北京" in st.session_state['available_cities'] else 0)
        st.session_state['target_city'] = target_city
    else:
        st.session_state['target_city'] = None

    st.subheader("聚类参数")
    auto_k = st.checkbox("自动选择最优K值", value=False)
    if auto_k:
        k_range = st.slider("K值搜索范围", 2, 10, (2, 7))
        st.session_state['k_range'] = range(k_range[0], k_range[1] + 1)
    else:
        n_clusters = st.slider("聚类数量 K", 2, 10, 4, 1)
        st.session_state['manual_k'] = n_clusters

    st.session_state['auto_k'] = auto_k

# 主界面
tab1, tab2, tab3 = st.tabs(["📁 数据预览", "🚀 模型训练", "📈 结果展示"])

with tab1:
    st.subheader("数据预览")

    if data_source == "内置数据集":
        st.info(f"当前选择：**{target_city}** 二手房数据集")
        if st.button("加载数据", type="primary"):
            with st.spinner("正在加载数据..."):
                try:
                    load_result = load_data(csv_path=DEFAULT_CSV, target_city=target_city)
                    df = load_result['df']
                    clean_result = clean_data(df, DEFAULT_FEATURES)
                    df_clean = clean_result['df']

                    st.session_state['clustering_data'] = df_clean
                    st.session_state['clustering_features'] = DEFAULT_FEATURES
                    st.session_state['load_info'] = {
                        'city': target_city,
                        'total': load_result['sample_count'],
                        'removed_missing': clean_result['removed_missing'],
                        'removed_outliers': clean_result['removed_outliers'],
                        'final': len(df_clean)
                    }

                    st.success(f"✅ 数据加载成功！共 {load_result['sample_count']} 条 → 清洗后 {len(df_clean)} 条")
                    st.dataframe(df_clean[DEFAULT_FEATURES].head(10))
                except Exception as e:
                    st.error(f"❌ 数据加载失败：{str(e)}")
    else:
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    st.error("❌ CSV 文件至少需要 2 个数值列用于聚类。")
                else:
                    st.session_state['uploaded_df'] = df
                    st.session_state['numeric_cols'] = numeric_cols
                    st.success(f"✅ 文件加载成功！检测到 {len(numeric_cols)} 个数值列。")
                    st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")

        if 'uploaded_df' in st.session_state:
            st.markdown("---")
            st.subheader("选择聚类特征")
            feature_cols = st.multiselect(
                "选择用于聚类的特征列（至少2个）",
                st.session_state['numeric_cols'],
                default=st.session_state['numeric_cols'][:2]
            )
            if len(feature_cols) >= 2:
                if st.button("清洗并加载数据", type="primary"):
                    df = st.session_state['uploaded_df']
                    clean_result = clean_data(df, feature_cols)
                    df_clean = clean_result['df']

                    st.session_state['clustering_data'] = df_clean
                    st.session_state['clustering_features'] = feature_cols
                    st.session_state['load_info'] = {
                        'city': '自定义',
                        'total': len(df),
                        'removed_missing': clean_result['removed_missing'],
                        'removed_outliers': clean_result['removed_outliers'],
                        'final': len(df_clean)
                    }

                    st.success(f"✅ 数据清洗完成！{len(df)} 条 → {len(df_clean)} 条")
                    st.dataframe(df_clean[feature_cols].head(10))
            else:
                st.warning("⚠️ 请至少选择 2 个特征列。")

    # 显示加载信息
    if 'load_info' in st.session_state:
        info = st.session_state['load_info']
        with st.expander("📊 数据清洗详情"):
            st.write(f"原始数据量：{info['total']} 条")
            st.write(f"缺失值移除：{info['removed_missing']} 条")
            for col, count in info['removed_outliers'].items():
                st.write(f"异常值移除（{col}）：{count} 条")
            st.write(f"最终数据量：{info['final']} 条")

with tab2:
    st.subheader("模型训练")

    if 'clustering_data' not in st.session_state:
        st.warning("⚠️ 请先在「数据预览」标签页加载数据。")
    else:
        df = st.session_state['clustering_data']
        features = st.session_state['clustering_features']
        X = df[features].values
        auto_k = st.session_state.get('auto_k', False)

        if st.button("开始训练", type="primary"):
            with st.spinner("正在训练模型..."):
                try:
                    # 自动选K
                    if auto_k:
                        st.info("🔍 正在搜索最优K值...")
                        k_results = select_k(X, st.session_state.get('k_range', range(2, 8)))
                        st.session_state['k_results'] = k_results

                        # 显示K值选择结果
                        k_df = pd.DataFrame({
                            'K值': k_results['k_values'],
                            '轮廓系数': [f"{v:.4f}" for v in k_results['silhouette']],
                            'CH指数': [f"{v:.0f}" for v in k_results['calinski_harabasz']],
                            'DB指数': [f"{v:.4f}" for v in k_results['davies_bouldin']]
                        })
                        st.dataframe(k_df, use_container_width=True)

                        best_k = k_results['best_k_sil']
                        st.success(f"✅ 自动选择 K={best_k}（基于轮廓系数）")

                        # 显示肘部图
                        fig_elbow = plot_elbow(k_results)
                        st.pyplot(fig_elbow)
                    else:
                        best_k = st.session_state.get('manual_k', 4)
                        k_results = None

                    # 训练模型
                    train_result = train_model(X, df, best_k, features)
                    metrics = train_result['metrics']

                    st.session_state['clustering_model'] = train_result['model']
                    st.session_state['clustering_scaler'] = train_result['scaler']
                    st.session_state['clustering_metrics'] = metrics
                    st.session_state['clustering_data_selected'] = df[features]
                    st.session_state['clustering_features'] = features
                    st.session_state['clustering_k'] = best_k
                    st.session_state['train_result'] = train_result
                    st.session_state['cluster_summaries'] = get_cluster_summary(
                        train_result['centers_real'], train_result['test_labels']
                    )

                    st.success("✅ 模型训练完成！")

                    # 显示训练信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("训练集大小", f"{train_result['train_size']} 条")
                    with col2:
                        st.metric("测试集大小", f"{train_result['test_size']} 条")
                    with col3:
                        st.metric("聚类数量", f"K={best_k}")

                except Exception as e:
                    st.error(f"❌ 训练失败：{str(e)}")

with tab3:
    st.subheader("结果展示")

    if 'clustering_metrics' not in st.session_state:
        st.warning("⚠️ 请先在「模型训练」标签页训练模型。")
    else:
        metrics = st.session_state['clustering_metrics']
        k = st.session_state['clustering_k']
        train_result = st.session_state['train_result']
        features = st.session_state['clustering_features']

        st.markdown(f"### 📊 K={k} 聚类评估指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("轮廓系数", f"{metrics['silhouette']:.4f}",
                       help="越接近1越好，衡量簇内紧密度和簇间分离度")
        with col2:
            st.metric("CH指数", f"{metrics['calinski_harabasz']:.2f}",
                       help="越大越好，簇间方差/簇内方差")
        with col3:
            st.metric("DB指数", f"{metrics['davies_bouldin']:.4f}",
                       help="越接近0越好，簇间距离与簇内距离的比值")

        st.markdown("---")

        # 聚类散点图
        st.subheader("聚类结果可视化")
        fig_cluster = plot_clusters(
            train_result['df_test'], train_result['test_labels'],
            train_result['centers_real'], features, k, metrics['silhouette']
        )
        st.pyplot(fig_cluster)

        # 聚类中心表
        st.subheader("聚类中心解读")
        summaries = st.session_state['cluster_summaries']
        summary_df = pd.DataFrame(summaries)
        summary_df.columns = ['类别', '原始标签', '平均总价(万)', '平均面积(㎡)', '平均单价(元/㎡)', '样本数']
        st.dataframe(summary_df, use_container_width=True)

        # K值选择结果（如果使用了自动选K）
        if 'k_results' in st.session_state:
            with st.expander("📈 K值选择详情"):
                k_results = st.session_state['k_results']
                fig_elbow = plot_elbow(k_results)
                st.pyplot(fig_elbow)

                k_df = pd.DataFrame({
                    'K值': k_results['k_values'],
                    'SSE': k_results['sse'],
                    '轮廓系数': k_results['silhouette'],
                    'CH指数': k_results['calinski_harabasz'],
                    'DB指数': k_results['davies_bouldin']
                })
                st.dataframe(k_df, use_container_width=True)
