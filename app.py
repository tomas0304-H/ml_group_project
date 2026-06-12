"""
机器学习小组项目 - Streamlit 展示系统主入口

运行方式：
    streamlit run app.py
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="机器学习小组项目",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 首页内容
st.title("🤖 机器学习小组项目 - 展示系统")
st.markdown("---")

st.markdown("""
## 📋 项目简介

本系统用于展示和比较小组成员的机器学习算法实验结果。

### 🎯 支持的任务类型

| 任务类型 | 算法 | 评价指标 |
|---------|------|---------|
| **分类任务** | SVM、KNN | Accuracy、Precision、Recall、F1、AUC |
| **回归任务** | 线性回归、Ridge回归 | MAE、MSE、RMSE、R² |
| **聚类任务** | K-Means | 轮廓系数、CH指数、DB指数 |

### 🚀 使用方法

1. 在左侧菜单中选择任务类型
2. 上传数据文件（CSV格式）或使用内置样例数据
3. 选择算法并设置参数
4. 点击运行按钮查看结果
5. 查看预测结果、评估指标和可视化图表

### 📁 项目结构

```
ml_group_project/
├── app.py                 # 本文件 - 系统主入口
├── pages/                 # 各任务页面
├── src/                   # 源代码
│   ├── classification/    # 分类算法
│   ├── regression/        # 回归算法
│   ├── clustering/        # 聚类算法
│   └── utils/             # 工具模块
├── data/                  # 数据集
├── models/                # 训练好的模型
└── results/               # 实验结果
```

### 👥 小组成员

| 成员 | 负责任务 |
|------|---------|
| 成员1 | 分类任务 - SVM |
| 成员2 | 分类任务 - KNN |
| 成员3 | 回归任务 - 线性回归 |
| 成员4 | 聚类任务 - K-Means |

---

**提示**：请从左侧菜单选择要查看的任务类型。
""")

# 侧边栏信息
with st.sidebar:
    st.markdown("## 📖 关于")
    st.markdown("""
    本系统是机器学习课程的小组项目展示平台。

    **技术栈**：
    - Streamlit
    - scikit-learn
    - pandas
    - plotly
    """)
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    st.success("✅ 系统运行正常")
