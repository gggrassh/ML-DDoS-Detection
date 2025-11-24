import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. 页面基础配置 ===
st.set_page_config(
    page_title="DDoS 智能检测系统",
    page_icon="🛡️",
    layout="wide"
)

# 自定义一点点 CSS 让界面更好看 (可选)
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)


# === 2. 加载模型和数据 (使用缓存，防止每次刷新都重新加载) ===
@st.cache_resource
def load_model():
    try:
        model = joblib.load('ddos_model.pkl')
        return model
    except:
        st.error("找不到模型文件 'ddos_model.pkl'，请先运行 step2_train_model.py")
        return None


@st.cache_data
def load_sample_data():
    # 我们只加载前 1000 行数据用于演示，避免内存爆炸
    try:
        df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=1000)
        # 记得清洗列名空格，和训练时保持一致
        df.columns = df.columns.str.strip()
        return df
    except:
        st.error("找不到 CSV 数据文件，请确认文件路径正确")
        return None


model = load_model()
data = load_sample_data()

# === 3. 侧边栏设计 ===
st.sidebar.title("控制面板 ⚙️")
app_mode = st.sidebar.selectbox("选择功能", ["系统概览", "实时流量检测模拟", "模型黑盒揭秘"])

# === 4. 主页面逻辑 ===

if app_mode == "系统概览":
    st.title("🛡️ 基于机器学习的 DDoS 流量检测系统")
    st.markdown("### 📊 项目背景")
    st.info(
        "本项目利用 **随机森林 (Random Forest)** 算法，通过分析网络流量特征（如包大小、流持续时间等），实时识别 DDoS 攻击。")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="模型准确率", value="99.99%", delta="High")
    with col2:
        st.metric(label="支持检测类型", value="DDoS / Normal")
    with col3:
        st.metric(label="算法核心", value="Random Forest")

    Random_forest_img = "./images/Random_forest.png"
    st.image(
        Random_forest_img,
        caption="随机森林算法示意图", use_column_width=True)

elif app_mode == "实时流量检测模拟":
    st.title("🚨 实时流量检测模拟")
    st.write("点击下方按钮，从数据集中随机抽取一条网络流量，模拟'抓包'并检测。")

    if st.button("🔍 抓取并分析一条流量"):
        if model is not None and data is not None:
            # 1. 随机抽取一行数据
            random_index = np.random.randint(0, len(data))
            sample = data.iloc[[random_index]]

            # 获取真实标签（用于对比）
            true_label = sample['Label'].values[0]

            # 2. 准备输入数据 (去掉 Label 列)
            input_data = sample.drop('Label', axis=1)
            # 处理无穷大 (和训练时一样的预处理)
            input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            input_data.fillna(0, inplace=True)  # 简单填充，防止报错

            # 3. 模型预测
            prediction = model.predict(input_data)[0]
            # 预测概率
            prob = model.predict_proba(input_data)[0]

            # 4. 展示结果
            st.divider()
            col_res, col_detail = st.columns([1, 2])

            with col_res:
                if prediction == 1:  # 1 代表 DDoS
                    st.error("⚠️ 警告：检测到 DDoS 攻击！")
                    st.metric("攻击置信度", f"{prob[1] * 100:.2f}%")
                else:
                    st.success("✅ 安全：正常流量")
                    st.metric("安全置信度", f"{prob[0] * 100:.2f}%")

                st.caption(f"真实标签: {true_label}")

            with col_detail:
                st.write("📝 **流量特征详情:**")
                # 展示几个关键特征
                important_cols = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
                st.dataframe(sample[important_cols].T)

elif app_mode == "模型黑盒揭秘":
    st.title("🧠 模型是如何思考的？")
    st.write("机器学习不是魔法。下面的图表展示了模型认为哪些特征对于判断 DDoS 最重要。")

    if model is not None and data is not None:
        # 提取特征重要性
        importances = model.feature_importances_
        feature_names = data.drop('Label', axis=1).columns

        # 创建 DataFrame 并排序
        feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

        # 画图
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis', ax=ax)
        ax.set_title('Top 10 Most Important Features for DDoS Detection')
        st.pyplot(fig)

        st.info("""
        **图表解读：**
        如果 'Bwd Packet Length Mean' (平均包长度) 排在前面，说明攻击者发送的数据包大小非常规律（或异常），这是模型判断的主要依据。
        """)