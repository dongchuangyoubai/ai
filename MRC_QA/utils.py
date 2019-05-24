import pickle

def getVocab(dim):
    filename = "glove.6B.%dd.txt" % dim
    print(filename)
    fr = open(filename, 'r', encoding='utf-8')
    vocab_list = []
    vocab_embs = []
    for i in fr.readlines():
        word, emb = i.strip().split(maxsplit=1)
        print(word)
        if len(emb.split()) == dim:
            vocab_list.append(word)
            vocab_embs.append(emb)

    return vocab_list, vocab_embs


def word2Id(vocab_list):
    word2id = {}
    for i in range(len(vocab_list)):
        word2id[vocab_list[i]] = i
    return word2id

def buildTrainData(filename, word2id):
    fr = open(filename, 'r', encoding='utf-8')
    train_data = []
    q_max_len = 0
    p_max_len = 0
    for i in fr.readlines():
        if len(i.strip().split('|||')) != 3:
            print('yqh:line:', i)
            continue
        else:
            [para, ques, ans] = i.strip().split('|||')
        para = para.strip().split()
        para = [str(word2id[w]) if w in word2id else '201534' for w in para]
        ques = ques.strip().split()
        ques = [str(word2id[w]) if w in word2id else '201534' for w in ques]
        train_data.append(' '.join(para) + "|||" + ' '.join(ques) + "|||" + ans)

    with open('train_data_ready', 'w', encoding='utf-8') as fw:
        for i in train_data:
            fw.writelines(str(i) + "\n")


if __name__ == '__main__':
    vocab_list, vocab_embs = getVocab(50)

    # print(vocab_list.index('unk'))
    word2id = word2Id(vocab_list)
    pickle.dump([vocab_list, vocab_embs, word2id], open('data.pickle', 'wb'))
    [vocab_list, vocab_embs, word2id] = pickle.load(open('data.pickle', 'rb'))
    # print(vocab_list.index('unk'))
    buildTrainData('train_data', word2id)
