import numpy as np
from numpy import dot, floor, argsort
from numpy.random import random, uniform
import numba as nb
import pandas as pd
import time

# 加载数据集
try:
    data = np.loadtxt('corel', usecols=range(1, 33))
    print(f"数据加载成功，形状: {data.shape}")
except FileNotFoundError:
    print("错误: 文件 'corel' 未找到，请确保文件路径正确。")
    exit()


# 欧氏距离计算（Numba加速）
@nb.jit(nopython=True)
def calc_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 哈希映射函数
def hash_and_fill(inputs, R, b, a, bucket_num):
    buckets = [{} for _ in range(bucket_num)]
    mapped_idxes = floor((dot(inputs, R) + b) / a)
    for i, hash_keys in enumerate(mapped_idxes):
        for j, hash_key in enumerate(hash_keys):
            buckets[j].setdefault(hash_key, []).append(i)
    return buckets


# 并集法查询
def find(q, k, R, b, a, buckets):
    hash_keys = np.floor((dot(q, R) + b) / a)[0]
    for i, hash_key in enumerate(hash_keys):
        if i == 0:
            candi_set = set(buckets[0][hash_key])
        else:
            candi_set = candi_set.union(buckets[i][hash_key])
    candi_set = list(candi_set)
    dist = [calc_dist(data[i], q) for i in candi_set]
    set_idxes = argsort(dist)[1:k + 1]
    res = [candi_set[i] for i in set_idxes]
    return res


# 超参数设置
bucket_num = 15  # 哈希表数量
a = 0.1  # 桶宽
R = random([32, bucket_num])
b = uniform(0, a, [1, bucket_num])
print("正在构建LSH索引...")
tic = time.perf_counter()
buckets = hash_and_fill(data, R, b, a, bucket_num)
toc = time.perf_counter()
print(f"索引构建完成，耗时: {toc - tic:.4f}秒")

# 加载或计算真实近邻
try:
    true_idxes = pd.read_csv('true_idxes.csv', header=None)
    print("已加载预计算的真实近邻")
except FileNotFoundError:
    print("未找到预计算的真实近邻，开始计算...")
    tic = time.perf_counter()
    true_idxes = []
    for i in range(1000):
        if i % 100 == 0:
            print(f"计算真实近邻: {i}/1000")
        dist_real = [calc_dist(data[j], data[i]) for j in range(data.shape[0])]
        true_idx = argsort(dist_real)[1:11].tolist()
        true_idxes.append(true_idx)
    toc = time.perf_counter()
    print(f"真实近邻计算完成，耗时: {toc - tic:.4f}秒")
    pd.DataFrame(true_idxes).to_csv('true_idxes.csv', index=False, header=False)
    true_idxes = pd.DataFrame(true_idxes)

# 执行LSH近邻搜索
print("开始LSH近邻搜索...")
tic = time.perf_counter()
pred_idx = []
for i in range(1000):
    if i % 100 == 0:
        print(f"搜索进度: {i}/1000")
    res = find(data[i], 10, R, b, a, buckets)
    pred_idx.append(res)
toc = time.perf_counter()
print(f"LSH搜索完成，总耗时: {toc - tic:.4f}秒")


# 计算性能指标
def count_true_num(pred, true):
    return sum(1 for x in pred if x in true)


n = 1000
precisions, recalls, accuracies = [], [], []

for i in range(n):
    TP = count_true_num(pred_idx[i], true_idxes.iloc[i].tolist())
    FP = 10 - TP
    FN = 10 - TP
    TN = data.shape[0] - 10 - FP

    precisions.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
    recalls.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    accuracies.append((TP + TN) / (TP + TN + FP + FN))

p, r, acc = np.mean(precisions), np.mean(recalls), np.mean(accuracies)
print(f"\n性能指标:")
print(f"查准率(Precision): {p * 100:.4f}%")
print(f"召回率(Recall):    {r * 100:.4f}%")
print(f"准确率(Accuracy):  {acc * 100:.4f}%")




# LSH 适用场景
# 高维数据：当数据维度超过 10 时，LSH 的随机投影能有效降维并保持相似性。
# 大规模数据集：索引构建和查询效率高，适合百万级以上数据。

# iDistance 适用场景
# 精确搜索需求：如地理信息系统（GIS）中的最近设施查询。
# 低维数据：在 2D/3D 空间中，iDistance 的空间划分和剪枝更高效。
# 查询频率低但精度要求高：如科学计算中的精确近邻分析。