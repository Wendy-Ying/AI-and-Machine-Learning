import numpy as np
from collections import Counter

# 计算熵
def calculate_entropy(labels):
    total_count = len(labels)
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        probability = count / total_count
        entropy -= probability * np.log2(probability)
    return entropy

# 计算信息增益
def information_gain(data, labels, threshold):
    total_entropy = calculate_entropy(labels)  # 计算总熵
    # 根据阈值分割数据
    left_indices = [i for i, x in enumerate(data) if x <= threshold]
    right_indices = [i for i, x in enumerate(data) if x > threshold]
    
    # 计算分割后的熵
    left_labels = [labels[i] for i in left_indices]
    right_labels = [labels[i] for i in right_indices]
    left_entropy = calculate_entropy(left_labels)
    right_entropy = calculate_entropy(right_labels)
    
    # 计算加权后的熵
    n = len(labels)
    weighted_entropy = (len(left_labels) / n) * left_entropy + (len(right_labels) / n) * right_entropy
    
    # 计算信息增益
    info_gain = total_entropy - weighted_entropy
    return info_gain

# 示例数据
data = [1, 2, 3, 4, 5]
labels = ['A', 'A', 'B', 'B', 'B']

# 计算所有可能阈值的信息增益
thresholds = [(data[i] + data[i + 1]) / 2 for i in range(len(data) - 1)]
info_gains = [information_gain(data, labels, th) for th in thresholds]

# 选择信息增益最大的阈值
best_threshold = thresholds[np.argmax(info_gains)]
print(f"Best threshold: {best_threshold}")
