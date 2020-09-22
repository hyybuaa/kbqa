import requests
import json
import time

def get_web_sim(data):
    url = r'http://192.168.1.28:6768/link'
    result = requests.post(url=url, data=json.dumps({"data": data}))
    return json.loads(result.text)['data']

def get_web_ner(data):
    url = r'http://192.168.1.28:6768/ner'
    result = requests.post(url=url, data=json.dumps({"data": data}))
    return json.loads(result.text)['data']  

def get_web_path(data):
    url = r'http://192.168.1.28:6768/rank'
    result = requests.post(url=url, data=json.dumps({"data": data}))
    return json.loads(result.text)['data']

def get_web_same_entity(data):
    url = r'http://192.168.1.28:6768/dupli_entity'
    result = requests.post(url=url, data=json.dumps({"data": data}))
    return json.loads(result.text)['data']



if __name__ == "__main__":
    start_time = time.time()
    # data = [['俄罗斯的首都有多少人口', '首都'], ['俄罗斯的首都有多少人口', '所属洲'], ['发明显微镜的人是什么职业？', '职业']]
    # result = get_web_path(data)
    query = r'尔赫斯的国家首都在哪里？'
    result = get_web_ner(query)
    # data = [['特朗普老婆的国籍', '妻子国籍'], ['特朗普老婆的国籍', '妻子'], ['特朗普老婆的国籍', '国籍']]
    # result = get_web_path(data)
    print(result)
    print("时间：", str(time.time()-start_time))