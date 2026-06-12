# 机器学习小组项目

## 📋 项目简介

本项目是机器学习课程的小组综合项目，包含分类、回归、聚类三种算法大类的实现与展示。

## 🎯 项目目标

1. 实现三种机器学习算法大类（分类、回归、聚类）
2. 搭建交互式展示系统，支持数据上传、模型训练、结果可视化
3. 对比不同算法的性能指标
4. 完成实验报告和答辩PPT

## 🛠 技术栈

- **Python 3.8+**
- **scikit-learn** - 机器学习算法库
- **pandas** - 数据处理
- **numpy** - 数值计算
- **matplotlib** - 静态可视化
- **plotly** - 交互式可视化
- **streamlit** - Web展示系统
- **joblib** - 模型保存/加载

## 📁 项目结构

```
ml_group_project/
├── app.py                         # Streamlit 展示系统主入口
├── requirements.txt               # 项目依赖清单
├── README.md                      # 本文件
│
├── data/                          # 数据集存放目录
│   ├── classification/            # 分类任务数据
│   ├── regression/                # 回归任务数据
│   └── clustering/                # 聚类任务数据
│
├── models/                        # 训练好的模型文件（.pkl）
│   ├── classification/
│   ├── regression/
│   └── clustering/
│
├── src/                           # 源代码目录
│   ├── classification/            # 分类任务模块
│   │   ├── base.py                # 分类算法基类（统一接口）
│   │   ├── svm.py                 # SVM 实现
│   │   ├── knn.py                 # KNN 实现
│   │   └── evaluate.py            # 分类评估工具
│   │
│   ├── regression/                # 回归任务模块
│   │   ├── base.py                # 回归算法基类（统一接口）
│   │   ├── linear.py              # 线性回归实现
│   │   └── evaluate.py            # 回归评估工具
│   │
│   ├── clustering/                # 聚类任务模块
│   │   ├── base.py                # 聚类算法基类（统一接口）
│   │   ├── kmeans.py              # K-Means 实现
│   │   └── evaluate.py            # 聚类评估工具
│   │
│   └── utils/                     # 公共工具模块
│       ├── preprocess.py          # 数据预处理工具
│       ├── visualization.py       # 可视化工具
│       └── io.py                  # 数据加载/保存工具
│
├── pages/                         # Streamlit 多页面应用
│   ├── 1_📊_分类任务.py
│   ├── 2_📈_回归任务.py
│   ├── 3_🔍_聚类任务.py
│   └── 4_⚖️_结果对比.py
│
├── results/                       # 实验结果输出目录
│   ├── classification/
│   ├── regression/
│   └── clustering/
│
└── report/                        # 报告和PPT存放目录
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd ml_group_project

# 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行展示系统

```bash
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

### 3. 使用流程

1. 在左侧菜单选择任务类型（分类/回归/聚类）
2. 上传 CSV 格式的数据文件
3. 选择算法并设置参数
4. 点击"开始训练"按钮
5. 查看评估指标和可视化结果

## 📐 统一接口规范

### 分类算法接口

所有分类算法必须继承 `BaseClassifier` 基类，并实现以下方法：

```python
from src.classification.base import BaseClassifier

class MyClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(model_name="MyClassifier")

    def train(self, X_train, y_train, **kwargs) -> Dict:
        """训练模型"""
        pass

    def predict(self, X) -> np.ndarray:
        """预测标签"""
        pass

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """预测概率"""
        pass
```

### 回归算法接口

所有回归算法必须继承 `BaseRegressor` 基类：

```python
from src.regression.base import BaseRegressor

class MyRegressor(BaseRegressor):
    def __init__(self):
        super().__init__(model_name="MyRegressor")

    def train(self, X_train, y_train, **kwargs) -> Dict:
        """训练模型"""
        pass

    def predict(self, X) -> np.ndarray:
        """预测值"""
        pass
```

### 聚类算法接口

所有聚类算法必须继承 `BaseClusterer` 基类：

```python
from src.clustering.base import BaseClusterer

class MyClusterer(BaseClusterer):
    def __init__(self):
        super().__init__(model_name="MyClusterer")

    def train(self, X, **kwargs) -> Dict:
        """训练模型"""
        pass

    def predict(self, X) -> np.ndarray:
        """预测聚类标签"""
        pass
```

### 评估指标返回格式

所有评估函数返回统一的字典格式：

```python
# 分类指标
{
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
    "auc": float or None
}

# 回归指标
{
    "mae": float,
    "mse": float,
    "rmse": float,
    "r2": float
}

# 聚类指标
{
    "silhouette": float,
    "calinski_harabasz": float,
    "davies_bouldin": float
}
```

## 👥 小组分工

| 成员 | 负责任务 | 算法 |
|------|---------|------|
| 成员1 | 分类任务 | SVM |
| 成员2 | 分类任务 | KNN |
| 成员3 | 回归任务 | 线性回归 |
| 成员4 | 聚类任务 | K-Means |

## 📊 评价指标说明

### 分类指标
- **Accuracy（准确率）**：正确预测的比例
- **Precision（精确率）**：预测为正类中实际为正类的比例
- **Recall（召回率）**：实际为正类中被正确预测的比例
- **F1-score**：精确率和召回率的调和平均
- **AUC**：ROC曲线下面积

### 回归指标
- **MAE（平均绝对误差）**：预测值与真实值差的绝对值的平均
- **MSE（均方误差）**：预测值与真实值差的平方的平均
- **RMSE（均方根误差）**：MSE的平方根
- **R²（决定系数）**：模型解释数据变异的比例

### 聚类指标
- **轮廓系数**：衡量聚类的紧密度和分离度（-1到1，越大越好）
- **CH指数**：类间方差与类内方差的比值（越大越好）
- **DB指数**：类内距离与类间距离的比值（越小越好）

## 📝 注意事项

1. 数据文件必须是 CSV 格式
2. 分类和回归任务的数据最后一列应为标签/目标值
3. 聚类任务不需要标签列
4. 模型文件保存在 `models/` 目录下
5. 实验结果保存在 `results/` 目录下

## 📄 许可证

本项目仅用于课程学习，请勿用于商业用途。
