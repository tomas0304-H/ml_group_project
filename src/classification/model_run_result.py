"""
ADMET 多标签分类模块

原作者：组员代码（机器学习大作业第三版）
修改：适配项目结构，支持 Streamlit 调用

功能：
- 分子特征提取（SMILES → RDKit 描述符）
- 多标签分类（BR+随机森林、ML-kNN）
- 可视化（ROC曲线、雷达图、指标表格）
- 模型保存/加载
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
from typing import Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')

# 机器学习核心工具
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    hamming_loss, roc_curve, auc
)

# 多标签算法
from skmultilearn.adapt import MLkNN

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 化学分子特征提取
from rdkit import Chem
from rdkit.Chem import Descriptors

# -------------------------- MLkNN 兼容性补丁 --------------------------
# scikit-multilearn 0.2.0 与 sklearn >= 1.3 不兼容，需要修补
from sklearn.neighbors import NearestNeighbors
_original_nn_init = NearestNeighbors.__init__

def _patched_nn_init(self, n_neighbors=5, **kwargs):
    _original_nn_init(self, n_neighbors=n_neighbors, **kwargs)

NearestNeighbors.__init__ = _patched_nn_init

# -------------------------- 项目路径配置 --------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "classification", "ADMET.xlsx")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "classification")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "classification")

# 列名配置
SMILES_COL = "SMILES"
LABEL_COLS = ["Caco-2", "CYP3A4", "hERG", "HOB", "MN"]  # 5个ADMET二分类标签
LABEL_DESC = {
    "Caco-2": "1=小肠上皮细胞渗透性好；0=渗透性差",
    "CYP3A4": "1=可被CYP3A4代谢；0=不能被代谢",
    "hERG": "1=具有心脏毒性；0=无心脏毒性",
    "HOB": "1=口服生物利用度好；0=生物利用度差",
    "MN": "1=具有遗传毒性；0=无遗传毒性"
}

# 实验超参数
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_RF_ESTIMATORS = 100
DEFAULT_MLKNN_K = 5

# 绘图风格配置
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# -------------------------- 1. 分子特征提取模块 --------------------------
def smiles_to_features(smiles: str):
    """将SMILES文本转换为RDKit数值描述符，用于建模"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 提取10个与ADMET高度相关的核心分子描述符
        features = [
            Descriptors.MolWt(mol),        # 分子量
            Descriptors.MolLogP(mol),      # 脂水分配系数
            Descriptors.NumHDonors(mol),   # 氢键供体数
            Descriptors.NumHAcceptors(mol),# 氢键受体数
            Descriptors.NumRotatableBonds(mol), # 可旋转键数
            Descriptors.TPSA(mol),         # 拓扑极性表面积
            Descriptors.NumAromaticRings(mol),  # 芳香环数
            Descriptors.FractionCSP3(mol), # sp3杂化碳占比
            Descriptors.HeavyAtomCount(mol), # 重原子数量
            Descriptors.NumHeteroatoms(mol) # 杂原子数量
        ]
        return features
    except:
        return None


def get_feature_names():
    """获取特征名称列表"""
    return [
        "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
        "NumRotatableBonds", "TPSA", "NumAromaticRings",
        "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms"
    ]


# -------------------------- 2. 数据加载模块 --------------------------
def load_and_process_data(data_path=None, test_size=None, random_state=None):
    """
    加载训练集、提取特征、划分训练/验证集；加载测试集。

    Args:
        data_path: Excel数据文件路径
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        dict: {
            'X_train', 'X_val', 'y_train', 'y_val',
            'X_test', 'test_smiles',
            'train_info': 训练信息字典
        }
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if test_size is None:
        test_size = DEFAULT_TEST_SIZE
    if random_state is None:
        random_state = DEFAULT_RANDOM_STATE

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    # 1. 加载训练集
    df_train = pd.read_excel(data_path, sheet_name="training")

    # 提取训练集特征
    feat_list = []
    valid_idx = []
    for idx, smi in enumerate(df_train[SMILES_COL]):
        feat = smiles_to_features(smi)
        if feat is not None:
            feat_list.append(feat)
            valid_idx.append(idx)

    df_train_valid = df_train.iloc[valid_idx].reset_index(drop=True)
    X = np.array(feat_list)
    y = df_train_valid[LABEL_COLS].values

    # 2. 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. 加载测试集
    df_test = pd.read_excel(data_path, sheet_name="test")

    # 提取测试集特征
    test_feat_list = []
    for smi in df_test[SMILES_COL]:
        feat = smiles_to_features(smi)
        test_feat_list.append(feat if feat is not None else [0] * 10)

    X_test = np.array(test_feat_list)
    test_smiles = df_test[SMILES_COL].values

    train_info = {
        "data_path": data_path,
        "train_total": len(df_train),
        "train_valid": len(df_train_valid),
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "feature_dim": X.shape[1],
        "label_cols": LABEL_COLS
    }

    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'X_test': X_test,
        'test_smiles': test_smiles,
        'train_info': train_info
    }


# -------------------------- 3. 模型训练与评估模块 --------------------------
def train_and_evaluate_models(X_train, X_val, y_train, y_val,
                               rf_estimators=None, mlknn_k=None):
    """
    训练两个模型：BR+随机森林、ML-kNN，返回所有评估指标。

    Args:
        X_train, X_val, y_train, y_val: 训练/验证数据
        rf_estimators: 随机森林树数量
        mlknn_k: ML-kNN近邻数

    Returns:
        dict: 模型结果字典
    """
    if rf_estimators is None:
        rf_estimators = DEFAULT_RF_ESTIMATORS
    if mlknn_k is None:
        mlknn_k = DEFAULT_MLKNN_K

    model_results = {}

    # --- 模型1：二元独立法(BR) + 随机森林 ---
    br_rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=rf_estimators, random_state=DEFAULT_RANDOM_STATE)
    )
    br_rf.fit(X_train, y_train)

    y_pred_br = br_rf.predict(X_val)
    y_prob_br = np.array([prob[:, 1] for prob in br_rf.predict_proba(X_val)]).T

    # 计算单标签指标
    br_single_metrics = []
    for i, label in enumerate(LABEL_COLS):
        prec = precision_score(y_val[:, i], y_pred_br[:, i], zero_division=0)
        rec = recall_score(y_val[:, i], y_pred_br[:, i], zero_division=0)
        f1 = f1_score(y_val[:, i], y_pred_br[:, i], zero_division=0)
        auc_score = roc_auc_score(y_val[:, i], y_prob_br[:, i])
        br_single_metrics.append({
            "标签": label,
            "精确率": round(prec, 4),
            "召回率": round(rec, 4),
            "F1-score": round(f1, 4),
            "AUC-ROC": round(auc_score, 4)
        })

    br_overall_metrics = {
        "平均AUC-ROC": round(np.mean([m["AUC-ROC"] for m in br_single_metrics]), 4),
        "汉明损失": round(hamming_loss(y_val, y_pred_br), 4),
        "宏平均F1": round(f1_score(y_val, y_pred_br, average="macro", zero_division=0), 4)
    }

    model_results["BR+随机森林"] = {
        "model": br_rf,
        "single_metrics": pd.DataFrame(br_single_metrics),
        "overall_metrics": br_overall_metrics,
        "y_pred": y_pred_br,
        "y_prob": y_prob_br
    }

    # --- 模型2：ML-kNN（多标签K近邻） ---
    mlknn = MLkNN(k=mlknn_k)
    mlknn.fit(X_train, y_train)

    y_pred_mlknn = mlknn.predict(X_val).toarray()
    y_prob_mlknn = mlknn.predict_proba(X_val).toarray()

    mlknn_single_metrics = []
    for i, label in enumerate(LABEL_COLS):
        prec = precision_score(y_val[:, i], y_pred_mlknn[:, i], zero_division=0)
        rec = recall_score(y_val[:, i], y_pred_mlknn[:, i], zero_division=0)
        f1 = f1_score(y_val[:, i], y_pred_mlknn[:, i], zero_division=0)
        auc_score = roc_auc_score(y_val[:, i], y_prob_mlknn[:, i])
        mlknn_single_metrics.append({
            "标签": label,
            "精确率": round(prec, 4),
            "召回率": round(rec, 4),
            "F1-score": round(f1, 4),
            "AUC-ROC": round(auc_score, 4)
        })

    mlknn_overall_metrics = {
        "平均AUC-ROC": round(np.mean([m["AUC-ROC"] for m in mlknn_single_metrics]), 4),
        "汉明损失": round(hamming_loss(y_val, y_pred_mlknn), 4),
        "宏平均F1": round(f1_score(y_val, y_pred_mlknn, average="macro", zero_division=0), 4)
    }

    model_results["ML-kNN"] = {
        "model": mlknn,
        "single_metrics": pd.DataFrame(mlknn_single_metrics),
        "overall_metrics": mlknn_overall_metrics,
        "y_pred": y_pred_mlknn,
        "y_prob": y_prob_mlknn
    }

    return model_results


# -------------------------- 4. 可视化模块 --------------------------
def plot_all_results(model_results, y_val):
    """
    生成所有可视化图表。

    Args:
        model_results: 模型结果字典
        y_val: 验证集标签

    Returns:
        dict: {'fig1': 单标签对比图, 'fig2': 整体指标图}
    """
    # ====================== 窗口1：单标签指标综合对比 ======================
    fig1 = plt.figure(figsize=(18, 12))
    gs = fig1.add_gridspec(3, 2, height_ratios=[1, 1.2, 0.9])

    # 子图1：AUC-ROC对比柱状图
    ax1 = fig1.add_subplot(gs[0, :])
    br_auc = model_results["BR+随机森林"]["single_metrics"].set_index("标签")["AUC-ROC"]
    mlknn_auc = model_results["ML-kNN"]["single_metrics"].set_index("标签")["AUC-ROC"]
    auc_compare_df = pd.DataFrame({
        "BR+随机森林": br_auc,
        "ML-kNN": mlknn_auc
    }).reset_index()
    auc_compare_melt = auc_compare_df.melt(id_vars="标签", var_name="模型", value_name="AUC-ROC")

    sns.barplot(data=auc_compare_melt, x="标签", y="AUC-ROC", hue="模型", ax=ax1, palette="Set2")
    ax1.set_title("各标签AUC-ROC指标对比", fontsize=14, fontweight="bold")
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(y=0.5, color="red", linestyle="--", label="随机基线")
    ax1.legend(loc="upper right", fontsize=10)

    # 子图2：BR+随机森林 ROC曲线
    ax2 = fig1.add_subplot(gs[1, 0])
    y_prob_br = model_results["BR+随机森林"]["y_prob"]
    for i, label in enumerate(LABEL_COLS):
        fpr, tpr, _ = roc_curve(y_val[:, i], y_prob_br[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.4f})')

    ax2.plot([0, 1], [0, 1], 'k--', label='随机基线')
    ax2.set_xlabel('假阳性率 (FPR)', fontsize=11)
    ax2.set_ylabel('真阳性率 (TPR)', fontsize=11)
    ax2.set_title('BR+随机森林 各标签ROC曲线', fontsize=12, fontweight="bold")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)

    # 子图3：ML-kNN ROC曲线
    ax3 = fig1.add_subplot(gs[1, 1])
    y_prob_mlknn = model_results["ML-kNN"]["y_prob"]
    for i, label in enumerate(LABEL_COLS):
        fpr, tpr, _ = roc_curve(y_val[:, i], y_prob_mlknn[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.4f})')

    ax3.plot([0, 1], [0, 1], 'k--', label='随机基线')
    ax3.set_xlabel('假阳性率 (FPR)', fontsize=11)
    ax3.set_ylabel('真阳性率 (TPR)', fontsize=11)
    ax3.set_title('ML-kNN 各标签ROC曲线', fontsize=12, fontweight="bold")
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(alpha=0.3)

    # 子图4：单标签详细指标表格
    ax_table1 = fig1.add_subplot(gs[2, :])
    ax_table1.set_title("单标签详细指标对比表", fontsize=12, fontweight="bold", pad=10)
    ax_table1.axis('off')

    br_single_df = model_results["BR+随机森林"]["single_metrics"]
    ml_single_df = model_results["ML-kNN"]["single_metrics"]

    col_labels = ["标签",
                  "BR-精确率", "BR-召回率", "BR-F1", "BR-AUC-ROC",
                  "ML-kNN-精确率", "ML-kNN-召回率", "ML-kNN-F1", "ML-kNN-AUC-ROC"]
    table_data = []
    for idx in range(len(LABEL_COLS)):
        br_row = br_single_df.iloc[idx]
        ml_row = ml_single_df.iloc[idx]
        row_data = [
            br_row["标签"],
            br_row["精确率"], br_row["召回率"], br_row["F1-score"], br_row["AUC-ROC"],
            ml_row["精确率"], ml_row["召回率"], ml_row["F1-score"], ml_row["AUC-ROC"]
        ]
        table_data.append(row_data)

    table1 = ax_table1.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#e6f2ff"] * 9,
        cellColours=[["#f8f9fa" for _ in range(9)] for _ in range(len(table_data))]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)

    plt.tight_layout()

    # ====================== 窗口2：多标签整体指标对比 ======================
    fig2 = plt.figure(figsize=(14, 7))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1])

    # 左侧：雷达图
    ax4 = fig2.add_subplot(gs2[0, 0], polar=True)
    radar_metrics = ["平均AUC-ROC", "宏平均F1", "1/汉明损失"]
    br_values = [
        model_results["BR+随机森林"]["overall_metrics"]["平均AUC-ROC"],
        model_results["BR+随机森林"]["overall_metrics"]["宏平均F1"],
        1 / model_results["BR+随机森林"]["overall_metrics"]["汉明损失"]
    ]
    mlknn_values = [
        model_results["ML-kNN"]["overall_metrics"]["平均AUC-ROC"],
        model_results["ML-kNN"]["overall_metrics"]["宏平均F1"],
        1 / model_results["ML-kNN"]["overall_metrics"]["汉明损失"]
    ]

    max_values = np.max([br_values, mlknn_values], axis=0)
    br_values_norm = br_values / max_values
    mlknn_values_norm = mlknn_values / max_values

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    br_values_norm = np.append(br_values_norm, br_values_norm[0])
    mlknn_values_norm = np.append(mlknn_values_norm, mlknn_values_norm[0])
    angles = np.append(angles, angles[0])

    ax4.plot(angles, br_values_norm, "o-", linewidth=2, label="BR+随机森林", color="#1f77b4")
    ax4.fill(angles, br_values_norm, alpha=0.25, color="#1f77b4")
    ax4.plot(angles, mlknn_values_norm, "o-", linewidth=2, label="ML-kNN", color="#ff7f0e")
    ax4.fill(angles, mlknn_values_norm, alpha=0.25, color="#ff7f0e")

    ax4.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=11)
    ax4.set_ylim(0, 1.1)
    ax4.set_title("两个模型整体核心指标对比", fontsize=14, fontweight="bold", pad=20)
    ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 右侧：整体指标表格
    ax_table2 = fig2.add_subplot(gs2[0, 1])
    ax_table2.set_title("多标签整体指标对比表", fontsize=12, fontweight="bold", pad=10)
    ax_table2.axis('off')

    overall_metric_names = ["平均AUC-ROC", "汉明损失", "宏平均F1"]
    br_overall = model_results["BR+随机森林"]["overall_metrics"]
    ml_overall = model_results["ML-kNN"]["overall_metrics"]

    overall_col_labels = ["指标", "BR+随机森林", "ML-kNN"]
    overall_table_data = [
        [metric_name, br_overall[metric_name], ml_overall[metric_name]]
        for metric_name in overall_metric_names
    ]

    table2 = ax_table2.table(
        cellText=overall_table_data,
        colLabels=overall_col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#e6f2ff"] * 3,
        cellColours=[["#f8f9fa" for _ in range(3)] for _ in range(len(overall_table_data))]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 1.8)

    plt.tight_layout()

    return {'fig1': fig1, 'fig2': fig2}


# -------------------------- 5. 测试集预测模块 --------------------------
def predict_test_set(model_results, X_test, test_smiles, output_dir=None):
    """
    对测试集进行预测，返回结果并保存为Excel。

    Args:
        model_results: 模型结果字典
        X_test: 测试集特征
        test_smiles: 测试集SMILES
        output_dir: 输出目录

    Returns:
        pd.DataFrame: 预测结果
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # 用两个模型分别预测
    br_model = model_results["BR+随机森林"]["model"]
    mlknn_model = model_results["ML-kNN"]["model"]

    br_pred = br_model.predict(X_test)
    br_prob = np.array([prob[:, 1] for prob in br_model.predict_proba(X_test)]).T

    mlknn_pred = mlknn_model.predict(X_test).toarray()
    mlknn_prob = mlknn_model.predict_proba(X_test).toarray()

    # 构建结果DataFrame
    result_list = []
    for i, smi in enumerate(test_smiles):
        row = {"SMILES": smi}
        for j, label in enumerate(LABEL_COLS):
            row[f"BR+随机森林_{label}_预测类别"] = int(br_pred[i, j])
            row[f"BR+随机森林_{label}_阳性概率"] = round(float(br_prob[i, j]), 4)
        for j, label in enumerate(LABEL_COLS):
            row[f"ML-kNN_{label}_预测类别"] = int(mlknn_pred[i, j])
            row[f"ML-kNN_{label}_阳性概率"] = round(float(mlknn_prob[i, j]), 4)
        result_list.append(row)

    test_result_df = pd.DataFrame(result_list)

    # 保存到Excel
    save_path = os.path.join(output_dir, "admet_test_predictions.xlsx")
    test_result_df.to_excel(save_path, index=False)

    return test_result_df


# -------------------------- 6. 模型保存/加载 --------------------------
def save_models(model_results, model_dir=None):
    """
    保存训练好的模型。

    Args:
        model_results: 模型结果字典
        model_dir: 模型保存目录

    Returns:
        dict: 保存路径信息
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    os.makedirs(model_dir, exist_ok=True)
    save_paths = {}

    for model_name, result in model_results.items():
        safe_name = model_name.replace("+", "_").replace("-", "_")
        model_path = os.path.join(model_dir, f"{safe_name}.pkl")
        joblib.dump(result['model'], model_path)
        save_paths[model_name] = model_path

    # 保存评估指标
    metrics_path = os.path.join(model_dir, "metrics.json")
    metrics_dict = {}
    for model_name, result in model_results.items():
        metrics_dict[model_name] = {
            "single_metrics": result['single_metrics'].to_dict(orient='records'),
            "overall_metrics": result['overall_metrics']
        }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    save_paths['metrics'] = metrics_path
    return save_paths


def load_models(model_dir=None):
    """
    加载已保存的模型。

    Args:
        model_dir: 模型目录

    Returns:
        dict: 模型字典
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    models = {}
    model_files = {
        "BR+随机森林": "BR_随机森林.pkl",
        "ML-kNN": "ML_kNN.pkl"
    }

    for model_name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)

    return models


# -------------------------- 7. Streamlit 包装函数 --------------------------
def run_admet_pipeline(data_path=None, test_size=None, random_state=None,
                       rf_estimators=None, mlknn_k=None,
                       save_model=True, save_results=True):
    """
    一键运行 ADMET 分类流程（供 Streamlit 调用）。

    Args:
        data_path: 数据文件路径
        test_size: 测试集比例
        random_state: 随机种子
        rf_estimators: 随机森林树数量
        mlknn_k: ML-kNN近邻数
        save_model: 是否保存模型
        save_results: 是否保存结果

    Returns:
        dict: {
            'model_results': 模型结果,
            'figures': {'fig1', 'fig2'},
            'test_results': 测试集预测结果,
            'train_info': 训练信息,
            'save_paths': 保存路径（如果保存了）
        }
    """
    # 1. 数据加载
    data = load_and_process_data(data_path, test_size, random_state)

    # 2. 模型训练
    model_results = train_and_evaluate_models(
        data['X_train'], data['X_val'],
        data['y_train'], data['y_val'],
        rf_estimators, mlknn_k
    )

    # 3. 可视化
    figures = plot_all_results(model_results, data['y_val'])

    # 4. 测试集预测
    test_results = predict_test_set(model_results, data['X_test'], data['test_smiles'])

    # 5. 保存
    save_paths = {}
    if save_model:
        save_paths.update(save_models(model_results))
    if save_results:
        results_path = os.path.join(DEFAULT_OUTPUT_DIR, "admet_test_predictions.xlsx")
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        test_results.to_excel(results_path, index=False)
        save_paths['test_results'] = results_path

    return {
        'model_results': model_results,
        'figures': figures,
        'test_results': test_results,
        'train_info': data['train_info'],
        'save_paths': save_paths
    }


# -------------------------- 8. 主函数（兼容原有调用方式） --------------------------
def main(data_path=None):
    """主函数，兼容命令行直接运行"""
    result = run_admet_pipeline(data_path=data_path)

    print("\n" + "=" * 70)
    print("ADMET 多标签分类完成！")
    print("=" * 70)
    print(f"训练样本：{result['train_info']['train_samples']}")
    print(f"验证样本：{result['train_info']['val_samples']}")
    print(f"测试样本：{result['train_info']['test_samples']}")
    print(f"特征维度：{result['train_info']['feature_dim']}")

    print("\n模型评估结果：")
    for model_name, model_result in result['model_results'].items():
        print(f"\n{model_name}:")
        for k, v in model_result['overall_metrics'].items():
            print(f"  {k}: {v}")

    if result['save_paths']:
        print("\n保存路径：")
        for k, v in result['save_paths'].items():
            print(f"  {k}: {v}")

    # 显示图表
    plt.show()

    return result


if __name__ == "__main__":
    main()
