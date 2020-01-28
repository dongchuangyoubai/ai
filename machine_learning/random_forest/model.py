import pandas as pd
import numpy as np
import math

df = pd.read_csv('train.csv')
feature_names = df.columns.values
print(feature_names)
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
def cal_entropy(dataset):
    if len(dataset) == 0:
        return 0
    tmp = dataset[:, -1]
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
def choice_best_point(dataset, feature):
    tmp = sorted(list(set(dataset[:, -1])))
    cand_points = [(tmp[i] + tmp[i + 1]) / 2 for i in range(len(tmp) - 1)]
    # print(cand_points)
    best_point = 0
    max_gain = 0
    for i in cand_points:
        left, right = data_split(dataset, feature, i)
        tmp_ent = cal_entropy(left) + cal_entropy(right)
        # print(tmp_ent)
        # break
        tmp_gain = cal_entropy(dataset) - tmp_ent
        if tmp_gain > max_gain:
            max_gain = tmp_gain
            best_point = i
    return best_point, max_gain



# 计算信息增益
def cal_gain(dataset, feature):
    best_point, max_gain = choice_best_point(dataset, feature)
    return best_point, max_gain

# 计算比例最高的类别
def major_class(labels):
    dic = {}
    for i in labels:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
    res_sorted = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    # print(res_sorted)
    return res_sorted[0][0]

# 选取当前数据集下，用于划分的最佳分割点特征索引
def choose_best_to_split(dataset):
    feature_nums = len(dataset[0]) - 1
    best_feature_idx = 0
    max_gain = 0
    for i in range(feature_nums):
        tmp_point, tmp_gain = choice_best_point(dataset, i)
        if tmp_gain > max_gain:
            best_feature_idx = i
    return best_feature_idx


# 构建决策树
def creat_tree(dataset):
    labels = dataset[:, -1]
    print(list(labels).count(0))
    # 当前类别完全相同，无需划分
    if list(labels).count(0) == len(labels):
        return 0
    if list(labels).count(1) == len(labels):
        return 1
    # 特征遍历完毕仍然分不出纯净分组，返回当前类别下最多的类别
    if len(dataset[0]) == 1:
        return major_class(labels)
    best_feature_idx = choose_best_to_split(dataset)
    print(feature_names[best_feature_idx])




if __name__ == '__main__':
    # 遍历所以特征选取信息增益最大的分裂点
    # idx = 0
    # max_gain = 0
    # for i in range(39):
    #     best_point, tmp_gain = cal_gain(random_choice(df, 200), -1, i)
    #     if tmp_gain > max_gain:
    #         idx = i
    # print(idx, best_point)
    creat_tree(random_choice(df, 200))

