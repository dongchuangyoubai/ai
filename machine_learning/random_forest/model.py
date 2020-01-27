import pandas as pd
import numpy as np
import math

df = pd.read_csv('train.csv')

train_size = len(df['age'])
# print(train_size)

# 有放回采样
def random_choice(dataset, k):
    index = np.random.choice(range(train_size), size=k)
    # print(index)
    return df.ix[index, :].values

# print(random_choice(df, 10))

# 节点分裂
def data_split(dataset, feature, value):
    left = []
    right = []
    for i in dataset:
        # print(type(i[feature]))
        # print(type(value))
        if i[feature] < value:
            left.append(list(i))
        else:
            right.append(list(i))
    return np.array(left), np.array(right)

# 计算信息熵
def cal_entropy(dataset, target):
    tmp = dataset[:, target]
    size = len(tmp)
    label_1 = sum(tmp)
    label_0 = size - label_1
    p0 = label_0 / size
    p1 = label_1 / size
    if p0 == 0 or p1 == 0:
        ent = 0
    else:
        ent = - p0 * math.log2(p0) - p1 * math.log2(p0)
    return ent

# 选择最佳分裂点
def choice_best_point(dataset, target, feature):
    tmp = sorted(list(set(dataset[:, feature])))
    cand_points = [(tmp[i] + tmp[i + 1]) / 2 for i in range(len(tmp) - 1)]
    # print(cand_points)
    best_point = 0
    max_gain = 0
    for i in cand_points:
        left, right = data_split(dataset, feature, i)
        tmp_ent = cal_entropy(left, target) + cal_entropy(right, target)
        # print(tmp_ent)
        # break
        tmp_gain = cal_entropy(dataset, target) - tmp_ent
        if tmp_gain > max_gain:
            max_gain = tmp_gain
            best_point = i
    return best_point, max_gain



# 计算信息增益
def cal_gain(dataset, target, feature):
    best_point, max_gain = choice_best_point(dataset, target, feature)
    return best_point, max_gain


if __name__ == '__main__':
    # 遍历所以特征选取信息增益最大的分裂点
    idx = 0
    max_gain = 0
    for i in range(39):
        best_point, tmp_gain = cal_gain(random_choice(df, 200), 39, i)
        if tmp_gain > max_gain:
            idx = i
    print(idx)
