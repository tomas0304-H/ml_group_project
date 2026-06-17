import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 默认配置 ==========
DEFAULT_COLUMN_MAPPING = {
    'total_price(w)': 'total_price',
    'area_sqm': 'square_meters'
}
DEFAULT_FEATURES = ['total_price', 'square_meters']
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "clustering", "SH-house-dataset.csv")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "clustering")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "clustering")


# ========== 核心函数 ==========

def load_data(csv_path=None, target_city="北京", max_samples=10000,
              column_mapping=None, features=None):
    """
    读取CSV，按城市过滤，重命名列，采样。

    Args:
        csv_path: CSV文件路径，默认使用内置数据集
        target_city: 目标城市名称
        max_samples: 最大采样数，None表示不采样
        column_mapping: 列名映射字典，默认使用 DEFAULT_COLUMN_MAPPING
        features: 聚类特征列名，默认使用 DEFAULT_FEATURES

    Returns:
        dict: {
            'df': 清洗前的DataFrame,
            'available_cities': 可用城市列表,
            'sample_count': 过滤后的样本数
        }
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if column_mapping is None:
        column_mapping = DEFAULT_COLUMN_MAPPING
    if features is None:
        features = DEFAULT_FEATURES

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件：{csv_path}")

    df_raw = pd.read_csv(csv_path, low_memory=False)
    available_cities = df_raw['city'].dropna().unique().tolist()

    df_filtered = df_raw[df_raw['city'] == target_city].copy()
    if df_filtered.empty:
        raise ValueError(f"未找到城市 '{target_city}' 的房源，可用城市：{available_cities}")

    if max_samples is not None and len(df_filtered) > max_samples:
        df_filtered = df_filtered.sample(n=max_samples, random_state=42)

    df = df_filtered.rename(columns=column_mapping)

    required_cols = features
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失必要列：{missing_cols}，请检查CSV文件或列名映射！")

    return {
        'df': df,
        'available_cities': available_cities,
        'sample_count': len(df)
    }


def clean_data(df, features):
    """
    去除缺失值和IQR异常值（3倍IQR）。

    Args:
        df: 输入DataFrame
        features: 需要清洗的特征列名

    Returns:
        dict: {
            'df': 清洗后的DataFrame,
            'removed_missing': 去除的缺失值行数,
            'removed_outliers': 各列去除的异常值行数
        }
    """
    df_clean = df.dropna(subset=features)
    removed_missing = len(df) - len(df_clean)

    removed_outliers = {}
    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        removed_outliers[col] = before - len(df_clean)

    return {
        'df': df_clean,
        'removed_missing': removed_missing,
        'removed_outliers': removed_outliers
    }


def select_k(X, k_range=range(2, 8)):
    """
    测试多个K值，计算评估指标。

    Args:
        X: 特征数据（已标准化）
        k_range: K值范围

    Returns:
        dict: {
            'k_values': K值列表,
            'sse': SSE列表,
            'silhouette': 轮廓系数列表,
            'calinski_harabasz': CH指数列表,
            'davies_bouldin': DB指数列表,
            'best_k_sil': 轮廓系数推荐的K,
            'best_k_ch': CH指数推荐的K,
            'best_k_db': DB指数推荐的K
        }
    """
    k_list = list(k_range)
    sse, sil_scores, ch_scores, db_scores = [], [], [], []

    for k in k_list:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans_temp.fit(X)
        labels = kmeans_temp.labels_

        sse.append(kmeans_temp.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    return {
        'k_values': k_list,
        'sse': sse,
        'silhouette': sil_scores,
        'calinski_harabasz': ch_scores,
        'davies_bouldin': db_scores,
        'best_k_sil': k_list[np.argmax(sil_scores)],
        'best_k_ch': k_list[np.argmax(ch_scores)],
        'best_k_db': k_list[np.argmin(db_scores)]
    }


def train_model(X, df_clean, k, features, test_size=0.2):
    """
    划分训练/测试集，标准化，训练KMeans，评估测试集。

    Args:
        X: 原始特征数据（未标准化）
        df_clean: 清洗后的完整DataFrame（用于元数据）
        k: 聚类数量
        features: 特征列名
        test_size: 测试集比例

    Returns:
        dict: {
            'model': KMeans模型,
            'scaler': StandardScaler,
            'X_train': 训练集（标准化后）,
            'X_test': 测试集（标准化后）,
            'df_test': 测试集元数据,
            'test_labels': 测试集预测标签,
            'metrics': {'silhouette', 'calinski_harabasz', 'davies_bouldin'},
            'centers_real': 聚类中心（原始尺度）,
            'train_size': 训练集大小,
            'test_size': 测试集大小
        }
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, _, df_test_meta = train_test_split(
        X_scaled, df_clean, test_size=test_size, random_state=42
    )

    model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    model.fit(X_train)

    test_labels = model.predict(X_test)
    sil = silhouette_score(X_test, test_labels)
    ch = calinski_harabasz_score(X_test, test_labels)
    db = davies_bouldin_score(X_test, test_labels)

    centers_real = scaler.inverse_transform(model.cluster_centers_)

    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'df_test': df_test_meta,
        'test_labels': test_labels,
        'metrics': {
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db
        },
        'centers_real': centers_real,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def predict(model, scaler, X):
    """
    用训练好的模型预测新数据。

    Args:
        model: KMeans模型
        scaler: StandardScaler
        X: 原始特征数据

    Returns:
        np.ndarray: 聚类标签
    """
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def plot_elbow(k_results):
    """
    绘制肘部法则图。

    Args:
        k_results: select_k() 的返回值

    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_results['k_values'], k_results['sse'], marker='o', linestyle='-')
    ax.set_title('K-Means 肘部法则（SSE vs K）')
    ax.set_xlabel('聚类数 K')
    ax.set_ylabel('SSE（簇内平方和）')
    ax.set_xticks(k_results['k_values'])
    ax.grid(alpha=0.5)
    plt.tight_layout()
    return fig


def plot_clusters(df_test, test_labels, centers_real, features, k, silhouette):
    """
    绘制聚类散点图。

    Args:
        df_test: 测试集DataFrame
        test_labels: 聚类标签
        centers_real: 聚类中心（原始尺度）
        features: 特征列名 [x_col, y_col]
        k: 聚类数量
        silhouette: 轮廓系数

    Returns:
        matplotlib.Figure
    """
    df_plot = df_test.copy()
    df_plot['Cluster_Label'] = test_labels

    # 按总价排序聚类中心
    sorted_indices = np.argsort(centers_real[:, 0])
    centers_sorted = centers_real[sorted_indices]

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        x=features[1], y=features[0], hue='Cluster_Label',
        palette='Set2', data=df_plot, s=50, alpha=0.7, edgecolor='none', ax=ax
    )
    ax.scatter(
        centers_sorted[:, 1], centers_sorted[:, 0],
        c='darkred', s=250, marker='X', label='聚类中心', edgecolor='black', zorder=5
    )
    for idx, center in enumerate(centers_sorted):
        ax.annotate(
            f"类别{idx}\n{center[0]:.0f}万\n{center[1]:.0f}㎡",
            xy=(center[1], center[0]), xytext=(10, 10),
            textcoords='offset points', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    ax.set_title(f'二手房总价-面积聚类图（K={k}）\n测试集轮廓系数：{silhouette:.2f}', fontsize=14)
    ax.set_xlabel('房屋面积（㎡）', fontsize=12)
    ax.set_ylabel('房屋总价（万元）', fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    return fig


def get_cluster_summary(centers_real, test_labels):
    """
    生成聚类中心的业务解读。

    Args:
        centers_real: 聚类中心（原始尺度）
        test_labels: 测试集标签

    Returns:
        list[dict]: 每个聚类的摘要信息
    """
    sorted_indices = np.argsort(centers_real[:, 0])
    summaries = []

    for idx, original_cluster in enumerate(sorted_indices):
        center = centers_real[original_cluster]
        count = int(np.sum(test_labels == original_cluster))
        avg_unit_price = center[0] / center[1] * 10000 if center[1] > 0 else 0
        summaries.append({
            'category': idx,
            'label': int(original_cluster),
            'avg_price': round(float(center[0]), 1),
            'avg_area': round(float(center[1]), 1),
            'avg_unit_price': round(avg_unit_price),
            'count': count
        })

    return summaries


# ========== 脚本入口（直接运行不受影响） ==========
if __name__ == "__main__":
    TARGET_CITY = "北京"
    USE_AUTO_K = False
    MANUAL_K = 4
    K_CANDIDATES = range(2, 8)

    # 1. 数据读取
    print("1. 正在读取二手房源数据集...")
    result = load_data(target_city=TARGET_CITY)
    df = result['df']
    print(f"可用城市：{result['available_cities']}")
    print(f"已过滤出 {TARGET_CITY} 房源共 {result['sample_count']} 条")

    # 2. 数据清洗
    print("\n2. 数据清洗与预处理...")
    clean_result = clean_data(df, DEFAULT_FEATURES)
    df_clean = clean_result['df']
    print(f"剔除缺失值：{clean_result['removed_missing']} 行")
    for col, count in clean_result['removed_outliers'].items():
        print(f"  {col}列：剔除{count}条极端异常值")
    print(f"最终清洗后数据维度: {df_clean.shape}")

    X = df_clean[DEFAULT_FEATURES].values

    # 3. K值选择
    print("\n3. 聚类数量选择...")
    k_results = select_k(X, K_CANDIDATES)
    for i, k in enumerate(k_results['k_values']):
        print(f"  K={k}: 轮廓系数={k_results['silhouette'][i]:.4f}, "
              f"CH指数={k_results['calinski_harabasz'][i]:.0f}, "
              f"DB指数={k_results['davies_bouldin'][i]:.4f}")

    print(f"\n各指标推荐的最优K值：")
    print(f"  轮廓系数（越大越好）：K={k_results['best_k_sil']}")
    print(f"  CH指数（越大越好）：K={k_results['best_k_ch']}")
    print(f"  DB指数（越小越好）：K={k_results['best_k_db']}")

    if USE_AUTO_K:
        best_k = k_results['best_k_sil']
        print(f"\n自动选择最优K值：{best_k}")
    else:
        best_k = MANUAL_K
        k_idx = k_results['k_values'].index(best_k)
        print(f"\n使用手动指定的K值：{best_k}")
        print(f"  对应指标：轮廓系数={k_results['silhouette'][k_idx]:.4f}, "
              f"CH指数={k_results['calinski_harabasz'][k_idx]:.0f}, "
              f"DB指数={k_results['davies_bouldin'][k_idx]:.4f}")

    # 肘部图
    fig_elbow = plot_elbow(k_results)
    fig_elbow.savefig("elbow_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig_elbow)

    # 4. 训练模型
    print(f"\n4. 划分训练集/测试集（8:2），训练K-Means（K={best_k}）...")
    train_result = train_model(X, df_clean, best_k, DEFAULT_FEATURES)
    print(f"训练集：{train_result['train_size']}条, 测试集：{train_result['test_size']}条")

    # 5. 评估
    print("\n5. 模型评估（测试集）...")
    metrics = train_result['metrics']
    print(f"-> 轮廓系数：{metrics['silhouette']:.4f}")
    print(f"-> CH分数：{metrics['calinski_harabasz']:.4f}")
    print(f"-> DB指数：{metrics['davies_bouldin']:.4f}")

    # 6. 保存模型
    print("\n6. 保存模型与标准化器...")
    model_dir = DEFAULT_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(train_result['model'], os.path.join(model_dir, f"kmeans_{TARGET_CITY}_k{best_k}.pkl"))
    joblib.dump(train_result['scaler'], os.path.join(model_dir, f"scaler_{TARGET_CITY}.pkl"))
    print(f"模型已保存至：{model_dir}/")

    # 7. 结果输出
    print("\n7. 生成结果报表与可视化...")
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    df_test_output = train_result['df_test'].copy()
    df_test_output['Cluster_Label'] = train_result['test_labels']
    csv_path = os.path.join(output_dir, f"{TARGET_CITY}_test_results_k{best_k}.csv")
    df_test_output.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"-> 带聚类标签的测试集已保存：{csv_path}")

    # 聚类中心解读
    summaries = get_cluster_summary(train_result['centers_real'], train_result['test_labels'])
    print(f"\n{TARGET_CITY} 二手房聚类中心（按总价从低到高）：")
    for s in summaries:
        print(f"  类别 {s['category']}（标签{s['label']}）:")
        print(f"    平均总价：{s['avg_price']}万")
        print(f"    平均面积：{s['avg_area']}㎡")
        print(f"    平均单价：{s['avg_unit_price']}元/㎡")
        print(f"    样本数量：{s['count']}套")
        print("-" * 30)

    # 聚类图
    fig_cluster = plot_clusters(
        train_result['df_test'], train_result['test_labels'],
        train_result['centers_real'], DEFAULT_FEATURES, best_k, metrics['silhouette']
    )
    img_path = os.path.join(output_dir, f"{TARGET_CITY}_clustering_k{best_k}.png")
    fig_cluster.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cluster)
    print(f"-> 聚类图已保存：{img_path}")
