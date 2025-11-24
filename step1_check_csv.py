import pandas as pd
import numpy as np

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("正在读取 CSV 文件 (Friday-DDoS)...")

# 1. 读取 CSV
filename = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(filename)

# === 去除列名中的空格 ===
# 原始数据的列名是 " Label" 去除前面的空格变成 "Label"
original_cols = df.columns.tolist()
df.columns = df.columns.str.strip()
print(f"\n已清理列名空格。")

print("\n=== 1. 数据概览 ===")
print(f"数据形状: {df.shape}")

# === 查看标签分布 ===
print("\n=== 流量类别统计 (Label) ===")
# 查看里面是否只有 BENIGN 和 DDoS
print(df['Label'].value_counts())

# === 坑点2：检查脏数据 (非常重要) ===
print("\n=== 脏数据检查 ===")
# 检查是否有无穷大 (inf) 和 空值 (NaN)
# 这一步决定了我们下一阶段清洗数据的代码怎么写
count_inf = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
count_null = df.isnull().sum().sum()

print(f"无穷大数值 (Infinity) 数量: {count_inf}")
print(f"空值 (Null/NaN) 数量: {count_null}")

if count_inf > 0 or count_null > 0:
    print("\n 警告：数据不干净，需要清洗它")
else:
    print("\n 数据很干净！")