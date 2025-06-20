import numpy as np
import time
import heapq
import bisect
import csv
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os


def load_data(filename):
    """加载数据集，返回包含ID和特征的元组列表"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            id = int(parts[0])
            features = np.array(parts[1:])
            data.append((id, features))
    return data


def normalize_data(data):
    """将数据归一化到[0,1]范围"""
    features = np.array([x[1] for x in data])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return [(data[i][0], normalized_features[i]) for i in range(len(data))]


def euclidean_distance(a, b):
    """计算欧氏距离"""
    return np.sqrt(np.sum((a - b) ** 2))


class iDistanceIndex:
    """iDistance索引结构实现"""

    def __init__(self, data, num_partitions=20, max_ref_points=1000):
        """
        初始化iDistance索引
        :param data: 数据集，格式为[(id, feature), ...]
        :param num_partitions: 分区数量
        :param max_ref_points: 用于选择参考点的最大样本数
        """
        self.data = data
        self.num_partitions = num_partitions
        self.reference_points = []
        self.partitions = defaultdict(list)
        self.max_ref_points = max_ref_points
        self.build_index()

    def select_reference_points(self):
        """使用k-means聚类选择参考点，对大数据集进行采样"""
        features = np.array([x[1] for x in self.data])

        # 如果数据量太大，采样部分数据用于聚类
        if len(features) > self.max_ref_points:
            indices = np.random.choice(len(features), self.max_ref_points, replace=False)
            sample_features = features[indices]
        else:
            sample_features = features

        kmeans = KMeans(n_clusters=self.num_partitions, random_state=42, n_init=10)
        kmeans.fit(sample_features)
        self.reference_points = kmeans.cluster_centers_

    def build_index(self):
        """构建iDistance索引"""
        print("Selecting reference points...")
        self.select_reference_points()

        print("Building partitions...")
        # 并行处理数据点分配（这里简化实现）
        for id, point in self.data:
            # 找到最近的参考点
            min_dist = float('inf')
            best_ref = 0
            for i, ref in enumerate(self.reference_points):
                dist = euclidean_distance(point, ref)
                if dist < min_dist:
                    min_dist = dist
                    best_ref = i

            # 存储格式：(一维键值, 数据点ID, 原始特征)
            one_dim_key = best_ref + min_dist  # 分区编号+距离作为一维键
            self.partitions[best_ref].append((one_dim_key, id, point))

        print("Sorting partitions...")
        # 对每个分区内的数据按一维键排序
        for ref in self.partitions:
            self.partitions[ref].sort(key=lambda x: x[0])

        print("Index construction complete.")

    def range_query(self, query_point, radius):
        """范围查询，返回距离query_point在radius内的所有点"""
        results = []

        for ref_idx, ref_point in enumerate(self.reference_points):
            dist_to_ref = euclidean_distance(query_point, ref_point)

            # 计算一维键的查询范围
            lower = ref_idx + max(0, dist_to_ref - radius)
            upper = ref_idx + dist_to_ref + radius

            # 在分区内进行一维范围查询
            partition = self.partitions[ref_idx]

            # 使用二分查找确定范围
            left = bisect.bisect_left([x[0] for x in partition], lower)
            right = bisect.bisect_right([x[0] for x in partition], upper)

            for i in range(left, right):
                _, id, point = partition[i]
                actual_dist = euclidean_distance(query_point, point)
                if actual_dist <= radius:
                    results.append((id, actual_dist))

        return results

    def knn_query(self, query_point, k, max_radius=10.0, max_iter=10):
        """
        KNN查询
        :param query_point: 查询点特征
        :param k: 需要的最近邻数量
        :param max_radius: 最大搜索半径
        :param max_iter: 最大迭代次数
        :return: 前k个最近邻的(id, distance)列表
        """
        radius = 0.1  # 初始搜索半径
        iter_count = 0
        results = []

        while iter_count < max_iter and radius <= max_radius:
            candidates = self.range_query(query_point, radius)

            if len(candidates) >= k:
                # 按距离排序并取前k个
                candidates.sort(key=lambda x: x[1])
                return candidates[:k]
            else:
                # 扩大搜索半径
                radius *= 1.5  # 1.5倍增长比2倍更平滑
                iter_count += 1

        # 如果达到最大迭代次数或半径，返回当前找到的最佳结果
        candidates.sort(key=lambda x: x[1])
        return candidates[:min(k, len(candidates))]


def compute_ground_truth(data, query_points, k=10, cache_file="ground_truth.csv"):
    """暴力计算真实最近邻，支持结果缓存"""
    if os.path.exists(cache_file):
        print(f"Loading ground truth from cache: {cache_file}")
        ground_truth = []
        with open(cache_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # 转换为整数列表并跳过第一个元素（查询点自身ID）
                neighbors = [int(x) for x in row[1:k + 1]]
                ground_truth.append([(id, 0.0) for id in neighbors])  # 距离设为0.0，仅用于兼容格式
        return ground_truth

    print("Computing ground truth (this may take a while)...")
    ground_truth = []
    total = len(query_points)

    for i, (query_id, query_point) in enumerate(query_points):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{total} queries...")

        distances = []
        for id, point in data:
            if id == query_id:
                continue  # 排除查询点本身
            dist = euclidean_distance(query_point, point)
            distances.append((id, dist))

        # 按距离排序并取前k个
        distances.sort(key=lambda x: x[1])
        ground_truth.append(distances[:k])

    # 保存结果到缓存文件
    with open(cache_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, (query_id, _) in enumerate(query_points):
            # 写入查询点ID和其k个最近邻ID
            row = [query_id] + [x[0] for x in ground_truth[i]]
            writer.writerow(row)

    print(f"Ground truth saved to cache: {cache_file}")
    return ground_truth


def compute_predicted_neighbors(index, query_points, k=10, cache_file="predicted_neighbors.csv"):
    """计算预测的最近邻，支持结果缓存"""
    if os.path.exists(cache_file):
        print(f"Loading predicted neighbors from cache: {cache_file}")
        predicted = []
        with open(cache_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # 转换为整数列表
                neighbors = [int(x) for x in row[:k]]
                predicted.append([(id, 0.0) for id in neighbors])  # 距离设为0.0，仅用于兼容格式
        return predicted

    print("Computing predicted neighbors (this may take a while)...")
    predicted = []
    total = len(query_points)

    for i, (query_id, query_point) in enumerate(query_points):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{total} queries...")

        neighbors = index.knn_query(query_point, k)
        predicted.append(neighbors)
    return predicted


def evaluate_performance_with_tptnfpfn(predicted, ground_truth, data_size):
    """
    使用TP/TN/FP/FN评估性能
    :param predicted: 预测的近邻列表
    :param ground_truth: 真实近邻列表
    :param data_size: 数据集总大小
    :return: 平均precision, recall, accuracy
    """
    print("Evaluating performance with TP/TN/FP/FN metrics...")

    all_precision = []
    all_recall = []
    all_accuracy = []

    total_queries = len(predicted)

    for i in range(total_queries):
        # 提取预测和真实的近邻ID集合
        pred_ids = set([x[0] for x in predicted[i]])
        true_ids = set([x[0] for x in ground_truth[i]])

        # 计算TP, TN, FP, FN
        tp = len(pred_ids & true_ids)
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)
        tn = data_size - len(pred_ids | true_ids) - 1  # 减去查询点本身

        # 计算性能指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        all_precision.append(precision)
        all_recall.append(recall)
        all_accuracy.append(accuracy)

    # 计算平均值
    avg_precision = sum(all_precision) / total_queries
    avg_recall = sum(all_recall) / total_queries
    avg_accuracy = sum(all_accuracy) / total_queries

    return avg_precision, avg_recall, avg_accuracy


def main():
    # 加载和预处理数据
    print("Loading data...")
    data = load_data('corel')
    print(f"Loaded {len(data)} data points.")

    print("Normalizing data...")
    data = normalize_data(data)

    # 前1000个点作为查询点
    query_points = data[:1000]
    print(f"Using {len(query_points)} query points.")

    # 计算真实最近邻（使用缓存机制）
    ground_truth = compute_ground_truth(data, query_points, cache_file="ground_truth.csv")

    # 构建iDistance索引
    print("\nBuilding iDistance index...")
    index = iDistanceIndex(data, num_partitions=20, max_ref_points=2000)

    # 计算预测的最近邻（使用缓存机制）
    predicted = compute_predicted_neighbors(index, query_points)

    # 评估性能（使用TP/TN/FP/FN）
    print("\nEvaluating performance...")
    precision, recall, accuracy = evaluate_performance_with_tptnfpfn(
        predicted, ground_truth, len(data))

    print("\nPerformance Results:")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

# Performance Results:
# Precision@10: 0.9000
# Recall@10: 0.9000
# Accuracy: 1.0000