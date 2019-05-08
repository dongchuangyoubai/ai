import json

fr = open('train-v2.0.json', 'r')

js = json.loads(fr.readline())
print(type(js))
print(js["version"])
# print(js['data'][0])
print(js['data'][0]['paragraphs'][0])
print(js['data'][0]['paragraphs'][0]['qas'])
print(js['data'][0]['paragraphs'][0]['context'])
data = js['data']
for i in data:

