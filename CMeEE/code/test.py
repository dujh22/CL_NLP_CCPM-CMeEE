import requests, json

url = 'http://127.0.0.1:1234/getMedicalNER'
data = {'inputstring': '患者40年前发现血压升高，最高血压160/100mmHg，规律服用苯磺酸氨氯地平片2.5mgQd控制血压，血压波动在130/80mmHg左右；2015年于我院诊断为“脑梗死”，恢复尚可，生活不能自理；否认肝炎、结核、疟疾病史，否认糖尿病、精神疾病史，否认外伤史，否认食物、药物过敏史，预防接种史不详。'
    }


# data_json = json.dumps(data)   #dumps：将python对象解码为json数据
r_json = requests.post(url,data)



r = requests.post(url, data)
s = r.text

s = json.loads(s)
s = s['data']['MedicalNER']['entities']

for entity in s:
    if entity['type']=='sym':
        symptom_word1 = entity['entity']
        # 对symptom_word1进行词典转换
        # 使用编辑距离在已有词表中召回50个词
        # 使用bert进行排序

        print(symptom_word1)

print(r)
# print(r.content)