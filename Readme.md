# 🛡️ ML-DDoS-Detection System

基于机器学习（随机森林）的 DDoS 流量检测系统，包含交互式 Web 界面。

## 📊 项目简介
本项目利用 **CIC-IDS2017** 数据集，通过 **Random Forest** 算法训练模型，能够高效识别正常流量与 DDoS 攻击流量。
项目包含一个基于 **Streamlit** 的可视化 Web 界面，支持实时流量模拟检测与特征重要性分析。

## 🚀 功能特性
- **高准确率**：模型测试准确率达到 99.9%。
- **可视化界面**：无需代码基础即可通过 Web 界面查看检测结果。
- **实时模拟**：支持从数据集中随机抽取样本模拟抓包检测。
- **可解释性 AI**：展示模型决策的关键特征权重（如包大小、端口号）。

## 🛠️ 技术栈
- **Python 3.9+**
- **Machine Learning:** Scikit-learn (Random Forest)
- **Data Processing:** Pandas, Numpy
- **Visualization:** Streamlit, Matplotlib, Seaborn

