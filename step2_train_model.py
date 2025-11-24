import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time

# 设置显示选项
pd.set_option('display.max_columns', None)

print("=== 1. 开始读取数据 ===")
filename = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(filename)

# 去除列名空格
df.columns = df.columns.str.strip()

print(f"原始数据形状: {df.shape}")

# === 2. 数据清洗 (Data Cleaning) ===
print("\n=== 2. 处理脏数据 ===")
# 替换无穷大为 NaN (空值)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# 删除所有包含 NaN 的行
df.dropna(inplace=True)

print(f"清洗后数据形状: {df.shape} (已删除包含空值/无穷大的行)")

# === 3. 标签编码 (Label Encoding) ===
# 机器更喜欢数字。我们将 BENIGN 设为 0，DDoS 设为 1
print("\n=== 3. 标签数字化 ===")
df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})
print("映射完成：BENIGN -> 0, DDoS -> 1")

# === 4. 特征与目标分离 ===
# X 是特征 (把 Label 列去掉，剩下的都是特征)
# y 是目标 (只保留 Label 列)
X = df.drop('Label', axis=1)
y = df['Label']

# === 5. 划分训练集和测试集 ===
# test_size=0.2 表示 20% 的数据用来做考试测试
print("\n=== 4. 划分训练集与测试集 (8:2) ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集数量: {len(X_train)}, 测试集数量: {len(X_test)}")

# === 6. 模型训练 (Model Training) ===
print("\n=== 5. 开始训练随机森林模型 (这可能需要几十秒) ===")
start_time = time.time()

# 初始化模型 (n_estimators=100 表示用了100棵决策树)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 调用你电脑所有CPU核心
clf.fit(X_train, y_train)

end_time = time.time()
print(f"训练完成！耗时: {end_time - start_time:.2f} 秒")

# === 7. 模型评估 (Evaluation) ===
print("\n=== 6. 模型评估结果 ===")
y_pred = clf.predict(X_test)

# 准确率
acc = accuracy_score(y_test, y_pred)
print(f"检测准确率 (Accuracy): {acc*100:.4f}%")

# 详细报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'DDoS (1)']))

# === 8. 保存模型 ===
print("\n=== 7. 保存模型 ===")
model_filename = 'ddos_model.pkl'
joblib.dump(clf, model_filename)
print(f"模型已保存为: {model_filename}")
print("你可以用于 Web 界面展示了！")