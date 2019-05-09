import json

fr = open('train-v2.0.json', 'r')

js = json.loads(fr.read())
print(js["version"])
print(len(js['data']))
print(js['data'][1]['paragraphs'][0])
print(js['data'][1]['paragraphs'][0]['qas'])
print(js['data'][1]['paragraphs'][0]['context'])
data = js['data']
clean_data = []
for i in data:
    for j in i['paragraphs']:
        context = j['context']
        for k in j['qas']:
            question = k['question']
            if k['answers'] != []:
                answer_start = k['answers'][0]['answer_start']
                answer_end = answer_start + len(k['answers'][0]['text'])
                # print(len(context))
                # print(answer_start, answer_end)
                # print(context[answer_start: answer_end])
                res = context + "|||" + question + "|||" + str(answer_start) + " " + str(answer_end)
                if len(res.split('|||')) != 3:
                    print(res)
                clean_data.append(res)
    #         break
    #     break
    # break

with open('train_data', 'w', encoding='utf-8') as fw:
    for i in clean_data:
        if len(i.strip().split("|||")) == 3:
            fw.writelines(i + '\n')