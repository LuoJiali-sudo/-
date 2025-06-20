import numpy as np


def pca(data, n_components=5):
    # 数据中心化
    mean = np.mean(data, axis=0)
    # print("mean:",mean,"\n")
    centered_data = data - mean

    # 计算协方差矩阵
    cov_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)

    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 对特征值进行排序，并获取对应的索引
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前 n_components 个特征向量
    top_eigenvectors = sorted_eigenvectors[:, :n_components]

    # 进行降维
    reduced_data = np.dot(centered_data, top_eigenvectors)

    return reduced_data, sorted_eigenvalues


# 读取数据
try:
    # 先读取完整数据
    all_data = np.loadtxt('ColorHistogram.asc')
    # 排除第一列
    data = all_data[:, 1:]
except FileNotFoundError:
    print("错误: 文件 'ColorHistogram.asc' 未找到。")
else:
    # 计算 PCA 之前的数据方差
    variance_before = np.var(data, axis=0)

    # 进行 PCA 降维
    reduced_data, eigenvalues = pca(data, n_components=5)

    # 计算 PCA 之后的数据方差
    variance_after = eigenvalues[:5]

    print("原始数据规模：", data.shape, "\n")
    print("PCA 之前的数据方差:")
    print(variance_before, "\n")
    print("PCA 降维后的数据（降至 5 维）:")
    print(reduced_data, "\n")
    print("PCA 之后数据的方差:")
    print(variance_after)