import os
fr = open(os.path.join('original', 'train1.txt'), 'r', encoding='utf-8')
clean_data = []
count = 0
for line in fr.readlines():
    cont = line.strip().split(' ')
    cont = [i.split('/') for i in cont]
    tmp_str = ''
    for i in cont:
        # if i[1] != 'o':
        #     continue
        label = i[1];
        if label == 'o':
            for j in i[0]:
                tmp_str += j + '/' + 'O '
        elif label == 'ns':
            tmp_str += i[0][0] + '/' + 'B-LOC '
            for j in i[0][1: -1]:
                tmp_str += j + '/' + 'M-LOC '
            if len(i[0]) > 1:
                tmp_str += i[0][-1] + '/' + 'E-LOC '
        elif label == 'nr':
            tmp_str += i[0][0] + '/' + 'B-PER '
            for j in i[0][1: -1]:
                tmp_str += j + '/' + 'M-PER '
            if len(i[0]) > 1:
                tmp_str += i[0][-1] + '/' + 'E-PER '
        elif label == 'nt':
            tmp_str += i[0][0] + '/' + 'B-ORG '
            for j in i[0][1: -1]:
                tmp_str += j + '/' + 'M-ORG '
            if len(i[0]) > 1:
                tmp_str += i[0][-1] + '/' + 'E-ORG '
        else:
            continue
    clean_data.append(tmp_str + '\n')
    count += 1
    if count == 10:
        break
fr.close()

fw = open('train_data', 'w', encoding='utf-8')
for i in clean_data:
    fw.writelines(i)


fw.close()
