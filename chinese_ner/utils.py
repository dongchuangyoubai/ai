import pickle
import numpy as np

def build_vocab(path):
    fr = open(path, 'r', encoding='utf-8')
    char2count = {}
    for line in fr.readlines():
        tmp = [i.split('/')[0] for i in line.strip().split(' ')]
        print(tmp)

        for i in tmp:
            if i not in char2count:
                char2count[i] = 0
            char2count[i] += 1
        print(char2count)
        break

build_vocab('train_data')