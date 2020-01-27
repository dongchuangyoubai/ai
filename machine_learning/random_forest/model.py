import pandas as pd
import numpy as np
import math

df = pd.read_csv('train.csv')

train_size = len(df['age'])
print(train_size)

# 有放回采样
def random_choice(dataset, k):
    index = np.random.choice(range(train_size), size=k)
    # print(index)
    return df.ix[index, :]

# print(random_choice(df, 10))

# 节点分裂
def data_split(dataset, row_idx, value):
    left = right = []
    for i in dataset:
        if i[row_idx] < value:
            left.append(i)
        else:
            right.append(i)
    return left, right

# 计算信息熵
def cal_entropy(dataset, target):
    tmp = dataset[target].values
    print(tmp)
    size = len(tmp)
    label_1 = sum(tmp)
    label_0 = size - label_1
    p0 = label_0 / size
    p1 = label_1 / size
    ent = - p0 * math.log2(p0) - p1 * math.log2(p0)
    return ent

# 选择最佳分裂点
def

# 计算信息增益
def cal_gain(dataset, target, feature):
    ent_org = cal_entropy(dataset, target)


print(cal_entropy(df, 'target'))
