import pickle
import numpy as np
import collections

label_dict = {'O': 0,
     'B-LOC': 1,
     'M-LOC': 2,
     'E-LOC': 3,
     'B-PER': 4,
     'M-PER': 5,
     'E-PER': 6,
     'B-ORG': 7,
     'M-ORG': 8,
     'E-ORG': 9}

def build_vocab(path, vocab_size):
    fr = open(path, 'r', encoding='utf-8')
    char2count = {}
    for line in fr.readlines():
        tmp = [i.split('/')[0] for i in line.strip().split(' ')]

        for i in tmp:
            if i not in char2count:
                char2count[i] = 0
            char2count[i] += 1
    a = sorted(char2count.items(), key=lambda x: x[1], reverse=True)
    char = []
    for (i, _) in a:
        if vocab_size - 1 != 0:
            char.append(i)
            vocab_size -= 1
    char.append('UNK')
    char2id = {}
    for i in range(len(char)):
        char2id[char[i]] = i
    # print(char2id)
    dataset_x = []
    dataset_y = []
    seq_len = 0
    fr = open(path, 'r', encoding='utf-8')
    for line in fr.readlines():
        tmp = [i.split('/') for i in line.strip().split(' ')]
        data_x = [str(char2id[i[0]]) if i[0] in char2id else str(3999) for i in tmp]
        data_y = [str(label_dict[i[1]]) for i in tmp]
        dataset_x.append(data_x)
        dataset_y.append(data_y)
        tmp = len(data_y)
        if tmp > seq_len:
            seq_len = tmp

    print(seq_len)

    with open('train_data_x', 'w', encoding='utf-8') as fw:
        for i in dataset_x:
            fw.writelines(' '.join(i) + '\n')

    with open('train_data_y', 'w', encoding='utf-8') as fw:
        for i in dataset_y:
            fw.writelines(' '.join(i) + '\n')



build_vocab('train_data', 4000)