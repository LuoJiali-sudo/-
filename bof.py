import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
import scipy.cluster.vq as cluster
import numpy as np
import cv2
import time

# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据集参数
image_nums = 1000  # 1000张图片
class_nums = 10  # 10个类别
images_per_class = 100  # 每类100张


# 1. 提取所有图像的SIFT特征点
def extract_sift_features():
    print("开始提取SIFT特征...")
    start_time = time.time()

    # 初始化描述子矩阵
    descriptors = np.empty((0, 128), dtype=np.float32)
    desc_list = []  # 保存每张图片的描述子

    for i in range(class_nums):
        for j in range(images_per_class):
            img_path = f'corel/{i}/{i * 100 + j}.jpg'
            img = cv2.imread(img_path)

            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                desc_list.append(None)
                continue

            # 提取SIFT特征
            sift = cv2.SIFT_create()
            _, features = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

            if features is not None:
                desc_list.append(features)
                descriptors = np.vstack((descriptors, features))
            else:
                desc_list.append(None)
                print(f"警告: 图像 {img_path} 未检测到特征点")

    print(f"SIFT特征提取完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"总描述子数量: {descriptors.shape[0]}, 每个描述子维度: {descriptors.shape[1]}")
    return descriptors, desc_list


# 2. 对所有关键点描述子进行k-means聚类构建视觉词典
def build_visual_vocabulary(descriptors, k=1000, sampling_rate=10):
    print("\n开始构建视觉词典...")
    start_time = time.time()

    # 对描述子进行采样以减少计算量
    sampled_descriptors = descriptors[::sampling_rate, :]
    print(f"采样后描述子数量: {sampled_descriptors.shape[0]}")

    # 使用k-means聚类
    print(f"开始k-means聚类(k={k})...")
    voc, _ = cluster.kmeans(sampled_descriptors.astype(np.float64), k, 1)  # 转换为float64提高精度

    print(f"视觉词典构建完成，耗时: {time.time() - start_time:.2f}秒")
    return voc


# 3. 为每张图像生成直方图表示
def generate_image_histograms(desc_list, voc, words_num):
    print("\n开始生成图像直方图表示...")
    start_time = time.time()

    imhists = np.zeros((image_nums, words_num))

    for i in range(image_nums):
        if i < len(desc_list) and desc_list[i] is not None:
            # 计算每个描述子到最近视觉单词的距离
            codes, _ = vq(desc_list[i], voc)

            # 生成直方图
            imhist, _ = np.histogram(codes, bins=range(words_num + 1))
            imhists[i] = imhist

    # 计算IDF权重
    occurence_num = np.sum(imhists > 0, axis=0)
    IDF = np.log((image_nums) / (occurence_num + 1))

    # 应用TF-IDF
    for i in range(image_nums):
        if np.sum(imhists[i]) > 0:  # 避免除以零
            imhists[i] = imhists[i] / np.sum(imhists[i]) * IDF

    print(f"图像直方图生成完成，耗时: {time.time() - start_time:.2f}秒")
    return imhists


# 4. 图像检索函数
def image_retrieval(query_id, imhists, top_k=10):
    print(f"\n开始检索与图像{query_id}最相似的{top_k}张图片...")
    start_time = time.time()

    # 获取查询图像的直方图
    query_hist = imhists[query_id]

    # 计算余弦相似度
    similarities = []
    for i in range(image_nums):
        if i != query_id:  # 排除查询图像本身
            norm_query = np.linalg.norm(query_hist)
            norm_target = np.linalg.norm(imhists[i])

            if norm_query > 0 and norm_target > 0:  # 避免除以零
                sim = np.dot(query_hist, imhists[i]) / (norm_query * norm_target)
                similarities.append((i, sim))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_k]

    print(f"检索完成，耗时: {time.time() - start_time:.2f}秒")
    return top_results


# 5. 显示结果
def display_results(query_id, results):
    # 显示查询图像
    query_img = plt.imread(f'corel/{query_id // 100}/{query_id}.jpg')
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title(f"Query Image (ID: {query_id}, Class: {query_id // 100})")
    plt.axis('off')
    plt.show()

    # 显示检索结果
    plt.figure(figsize=(20, 10))
    for i, (img_id, sim) in enumerate(results):
        result_img = plt.imread(f'corel/{img_id // 100}/{img_id}.jpg')
        plt.subplot(2, 5, i + 1)
        plt.imshow(result_img)
        plt.title(f"Rank {i + 1}\nID: {img_id}\nClass: {img_id // 100}\nSimilarity: {sim:.3f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 计算准确率(前10中同类别的数量)
    query_class = query_id // 100
    correct = sum(1 for (img_id, _) in results if img_id // 100 == query_class)
    accuracy = correct / len(results)
    print(f"Retrieval Accuracy (Same class in top {len(results)}): {correct}/{len(results)} = {accuracy:.2%}")


# 主函数
def main():
    # 1. 提取SIFT特征
    descriptors, desc_list = extract_sift_features()

    if descriptors.shape[0] == 0:
        print("错误: 未提取到任何特征描述子!")
        return

    # 2. 构建视觉词典(1000个视觉单词)
    voc = build_visual_vocabulary(descriptors, k=1000)

    # 3. 生成图像直方图表示
    imhists = generate_image_histograms(desc_list, voc, words_num=1000)

    # 4. 执行图像检索示例
    query_id =320  # 可以修改为任意0-999的ID
    results = image_retrieval(query_id, imhists)
    display_results(query_id, results)


if __name__ == "__main__":
    main()


# class0：Retrieval Accuracy (Same class in top 10): 9/10 = 90.00%
# class7：Retrieval Accuracy (Same class in top 10): 8/10 = 80.00%
