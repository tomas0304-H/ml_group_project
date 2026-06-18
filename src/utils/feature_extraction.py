import pandas as pd
import numpy as np
import warnings
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

# -------------------------- 全局配置 --------------------------
# 数据集路径
DATA_PATH = r"C:\Users\Gensokyo\Desktop\机器学习大作业\Data\ADMET.xlsx"
# 备用测试路径
# DATA_PATH = "/mnt/ADMET.xlsx"

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
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_ESTIMATORS = 100
MLKNN_K = 5

# 绘图风格配置
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# -------------------------- 1. 分子特征提取模块 --------------------------
def smiles_to_features(smiles: str):
    """将SMILES文本转换为RDKit数值描述符"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumHeteroatoms(mol)
        ]
        return features
    except:
        return None


def load_and_process_data():
    """加载数据、提取特征、划分数据集；输出完整样本统计信息"""
    print("=" * 70)
    print("1. 数据加载与样本统计信息")
    print("=" * 70)
    
    # 加载训练集
    df_train = pd.read_excel(DATA_PATH, sheet_name="training")
    print(f"训练集原始样本数：{len(df_train)}")
    
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
    
    print(f"训练集有效样本数：{len(df_train_valid)}")
    print(f"特征维度：{X.shape[1]}，标签数量：{len(LABEL_COLS)}")
    
    # 新增：各标签正负样本统计
    print("\n训练集各标签正负样本分布：")
    for i, label in enumerate(LABEL_COLS):
        pos_count = int(np.sum(y[:, i]))
        neg_count = len(y) - pos_count
        pos_ratio = pos_count / len(y)
        print(f"  {label:<8} 正样本: {pos_count:4d} | 负样本: {neg_count:4d} | 正样本占比: {pos_ratio:.2%}")
    
    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n训练集：{X_train.shape[0]} 样本，验证集：{X_val.shape[0]} 样本")
    
    # 加载测试集
    df_test = pd.read_excel(DATA_PATH, sheet_name="test")
    print(f"测试集样本数：{len(df_test)}")
    
    # 提取测试集特征
    test_feat_list = []
    for smi in df_test[SMILES_COL]:
        feat = smiles_to_features(smi)
        test_feat_list.append(feat if feat is not None else [0]*10)
    
    X_test = np.array(test_feat_list)
    test_smiles = df_test[SMILES_COL].values
    
    return X_train, X_val, y_train, y_val, X_test, test_smiles


# -------------------------- 2. 模型训练与评估模块 --------------------------
def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    """训练两个模型，输出单标签指标、多标签整体指标与对比总结"""
    print("\n" + "=" * 70)
    print("2. 模型单标签核心指标与多标签整体指标")
    print("=" * 70)
    
    model_results = {}
    
    # -------------------------- 模型1：BR + 随机森林 --------------------------
    print("\n━━━━━ 模型1：二元独立法(BR) + 随机森林 ━━━━━")
    br_rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE)
    )
    br_rf.fit(X_train, y_train)
    
    y_pred_br = br_rf.predict(X_val)
    y_prob_br = np.array([prob[:, 1] for prob in br_rf.predict_proba(X_val)]).T
    
    # 单标签指标
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
    
    # 多标签整体指标
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
    
    # 打印单标签指标
    print("▶ 单标签核心指标（精确率/召回率/F1-score/AUC-ROC）：")
    print(pd.DataFrame(br_single_metrics).to_string(index=False))
    # 打印多标签整体指标
    print("\n▶ 多标签整体指标：")
    for k, v in br_overall_metrics.items():
        print(f"  {k:<12} {v}")
    
    # -------------------------- 模型2：ML-kNN --------------------------
    print("\n━━━━━ 模型2：ML-kNN（多标签K近邻） ━━━━━")
    mlknn = MLkNN(k=MLKNN_K)
    mlknn.fit(X_train, y_train)
    
    y_pred_mlknn = mlknn.predict(X_val).toarray()
    y_prob_mlknn = mlknn.predict_proba(X_val).toarray()
    
    # 单标签指标
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
    
    # 多标签整体指标
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
    
    # 打印单标签指标
    print("▶ 单标签核心指标（精确率/召回率/F1-score/AUC-ROC）：")
    print(pd.DataFrame(mlknn_single_metrics).to_string(index=False))
    # 打印多标签整体指标
    print("\n▶ 多标签整体指标：")
    for k, v in mlknn_overall_metrics.items():
        print(f"  {k:<12} {v}")
    
    # -------------------------- 模型对比总结 --------------------------
    print("\n" + "-" * 70)
    print("📊 两个模型整体指标对比总结")
    print("-" * 70)
    compare_df = pd.DataFrame({
        "指标": ["平均AUC-ROC", "汉明损失", "宏平均F1"],
        "BR+随机森林": [
            br_overall_metrics["平均AUC-ROC"],
            br_overall_metrics["汉明损失"],
            br_overall_metrics["宏平均F1"]
        ],
        "ML-kNN": [
            mlknn_overall_metrics["平均AUC-ROC"],
            mlknn_overall_metrics["汉明损失"],
            mlknn_overall_metrics["宏平均F1"]
        ]
    })
    print(compare_df.to_string(index=False))
    
    return model_results


# -------------------------- 3. 可视化模块（双窗口布局） --------------------------
def plot_all_results(model_results, y_val):
    """
    双窗口输出：
    窗口1：单标签指标（AUC对比柱状图 + 两个模型ROC曲线）
    窗口2：多标签整体指标（雷达图）
    """
    print("\n" + "=" * 70)
    print("3. 可视化图表生成")
    print("=" * 70)
    
    br_single = model_results["BR+随机森林"]["single_metrics"].set_index("标签")
    mlknn_single = model_results["ML-kNN"]["single_metrics"].set_index("标签")
    y_prob_br = model_results["BR+随机森林"]["y_prob"]
    y_prob_mlknn = model_results["ML-kNN"]["y_prob"]
    
    # ====================== 窗口1：单标签指标合集 ======================
    fig1 = plt.figure(figsize=(18, 12))
    fig1.suptitle("单标签指标可视化窗口", fontsize=16, fontweight="bold", y=0.98)
    
    # 子图1：两个模型各标签AUC-ROC对比柱状图（上方跨两列）
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    auc_compare = pd.DataFrame({
        "BR+随机森林": br_single["AUC-ROC"],
        "ML-kNN": mlknn_single["AUC-ROC"]
    }).reset_index().melt(id_vars="标签", var_name="模型", value_name="AUC-ROC")
    
    sns.barplot(data=auc_compare, x="标签", y="AUC-ROC", hue="模型", ax=ax1, palette="Set2", width=0.6)
    ax1.set_title("两个模型各标签 AUC-ROC 对比", fontsize=14, fontweight="bold")
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="随机基线")
    ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax1.bar_label(ax1.containers[0], fmt="%.3f", padding=3, fontsize=9)
    ax1.bar_label(ax1.containers[1], fmt="%.3f", padding=3, fontsize=9)
    
    # 子图2：BR+随机森林各标签ROC曲线（左下）
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    for i, label in enumerate(LABEL_COLS):
        fpr, tpr, _ = roc_curve(y_val[:, i], y_prob_br[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.4f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', label='随机基线', alpha=0.7)
    ax2.set_xlabel('假阳性率 (FPR)', fontsize=11)
    ax2.set_ylabel('真阳性率 (TPR)', fontsize=11)
    ax2.set_title('BR+随机森林 各标签ROC曲线', fontsize=13, fontweight="bold")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 子图3：ML-kNN各标签ROC曲线（右下）
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    for i, label in enumerate(LABEL_COLS):
        fpr, tpr, _ = roc_curve(y_val[:, i], y_prob_mlknn[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.4f})')
    
    ax3.plot([0, 1], [0, 1], 'k--', label='随机基线', alpha=0.7)
    ax3.set_xlabel('假阳性率 (FPR)', fontsize=11)
    ax3.set_ylabel('真阳性率 (TPR)', fontsize=11)
    ax3.set_title('ML-kNN 各标签ROC曲线', fontsize=13, fontweight="bold")
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # ====================== 窗口2：多标签整体指标雷达图 ======================
    fig2 = plt.figure(figsize=(8, 8))
    fig2.suptitle("多标签整体指标可视化窗口", fontsize=16, fontweight="bold", y=0.98)
    ax4 = plt.subplot(111, polar=True)
    
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
    
    # 归一化到0-1区间
    max_values = np.max([br_values, mlknn_values], axis=0)
    br_norm = np.array(br_values) / max_values
    mlknn_norm = np.array(mlknn_values) / max_values
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    # 修复：使用np.concatenate闭合数组，保证首尾相连、长度一致
    angles = np.concatenate([angles, [angles[0]]])
    br_norm = np.concatenate([br_norm, [br_norm[0]]])
    mlknn_norm = np.concatenate([mlknn_norm, [mlknn_norm[0]]])
    
    ax4.plot(angles, br_norm, "o-", linewidth=2, label="BR+随机森林", color="#1f77b4")
    ax4.fill(angles, br_norm, alpha=0.25, color="#1f77b4")
    ax4.plot(angles, mlknn_norm, "o-", linewidth=2, label="ML-kNN", color="#ff7f0e")
    ax4.fill(angles, mlknn_norm, alpha=0.25, color="#ff7f0e")
    
    ax4.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=11)
    ax4.set_ylim(0, 1.1)
    ax4.set_title("两个模型整体核心指标对比", fontsize=13, fontweight="bold", pad=20)
    ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print("✅ 可视化窗口已弹出：")
    print("  窗口1：单标签指标（AUC对比柱状图 + 两个模型ROC曲线）")
    print("  窗口2：多标签整体指标（核心指标雷达图）")
    plt.show()


# -------------------------- 4. 测试集预测与Excel导出 --------------------------
def predict_test_set(model_results, X_test, test_smiles):
    """预测测试集全部样本，控制台预览50个结果，导出Excel"""
    print("\n" + "=" * 70)
    print("4. 测试集50个样本预测结果预览")
    print("=" * 70)
    
    br_model = model_results["BR+随机森林"]["model"]
    mlknn_model = model_results["ML-kNN"]["model"]
    
    # BR+随机森林预测
    br_pred = br_model.predict(X_test)
    br_prob = np.array([prob[:, 1] for prob in br_model.predict_proba(X_test)]).T
    
    # ML-kNN预测
    mlknn_pred = mlknn_model.predict(X_test).toarray()
    mlknn_prob = mlknn_model.predict_proba(X_test).toarray()
    
    # 构建结果表
    result_list = []
    for i, smi in enumerate(test_smiles):
        row = {"SMILES": smi}
        # BR+随机森林结果
        for j, label in enumerate(LABEL_COLS):
            row[f"BR+随机森林_{label}_预测类别"] = int(br_pred[i, j])
            row[f"BR+随机森林_{label}_阳性概率"] = round(float(br_prob[i, j]), 4)
        # ML-kNN结果
        for j, label in enumerate(LABEL_COLS):
            row[f"ML-kNN_{label}_预测类别"] = int(mlknn_pred[i, j])
            row[f"ML-kNN_{label}_阳性概率"] = round(float(mlknn_prob[i, j]), 4)
        result_list.append(row)
    
    test_result_df = pd.DataFrame(result_list)
    
    # 控制台预览全部50个样本的预测类别
    preview_cols = ["SMILES"] + [col for col in test_result_df.columns if "预测类别" in col]
    print("（仅展示预测类别，完整概率见Excel文件）")
    print(test_result_df[preview_cols].to_string(index=False))
    
    # 保存Excel
    save_path = r"C:\Users\Gensokyo\Desktop\机器学习大作业\测试集预测结果.xlsx"
    # save_path = "/mnt/测试集预测结果.xlsx"
    test_result_df.to_excel(save_path, index=False)
    print(f"\n✅ 完整预测结果已保存至：{save_path}")
    print("  文件包含：两个模型5个标签的预测类别 + 阳性概率，共21列数据")
    
    return test_result_df


# -------------------------- 主函数 --------------------------
def main():
    # 1. 数据加载与特征提取
    X_train, X_val, y_train, y_val, X_test, test_smiles = load_and_process_data()
    
    # 2. 模型训练与评估
    model_results = train_and_evaluate_models(X_train, X_val, y_train, y_val)
    
    # 3. 可视化（传入验证集真实标签）
    plot_all_results(model_results, y_val)
    
    # 4. 测试集预测与导出
    test_result_df = predict_test_set(model_results, X_test, test_smiles)
    
    # 5. 最终总结
    print("\n" + "=" * 70)
    print("🏁 实验运行完成！输出内容汇总")
    print("=" * 70)
    print("1. 控制台输出：样本统计、单标签指标、多标签指标、模型对比、50个测试集预览")
    print("2. 可视化窗口：单标签指标窗口 + 多标签整体指标窗口")
    print("3. 输出文件：测试集预测结果.xlsx（含两个模型的预测类别与阳性概率）")


if __name__ == "__main__":
    main()