import jieba
from sklearn.model_selection import train_test_split

fr = open('train', 'r', encoding='utf-8')

data_x = []
y = []
for line in fr.readlines():
    tmp = line.strip().split('\t')
    data_x.append('\t'.join(tmp[1: 3]))
    y.append(tmp[3])

print(data_x[10], y[10])

