import re
import json
from py2neo import Graph
import requests
import time
import pandas as pd
from copy import deepcopy


graph = Graph("bolt://127.0.0.1:7688", username="neo4j", password="123456")


class PathExample():

    def __init__(self, cypher,  sub_template='1a'):
        self.paths = []
        if sub_template == '1aa':
            self.path =  cypher['r.name']
            self.answer = cypher['m.name']
        elif sub_template == '1ab':
            self.path = cypher['m.name']
            self.answer = cypher['r.name']
        elif sub_template == '2aa':
            self.path = cypher['r1.name'] + '\t' + cypher['r2.name']
            self.answer = cypher['m.name']
        elif sub_template == '2ab':
            self.path = cypher['r1.name'] + '\t' + cypher['r2.name'] + cypher['m.name']
            self.answer = cypher['q.name']
        elif sub_template == '2da':
            self.path = cypher['r1.name']  + '\t' + cypher['r2.name'] + cypher['m.name']
            self.answer = cypher['q.name']
        elif sub_template == '2db':
            self.path =  cypher['r1.name'] + '\t' + cypher['r2.name']
            self.answer = cypher['m.name']


class PathFeature():

    def __init__(self):
        self.answer = {}

    def add(self, example):
        self.answer.setdefault(example.path, []).append(example.answer)


def sparql2cypher(sparql):
    entity = []
    # print("spaeql: ", sparql)
    target = re.findall(r"(select.+?where)", sparql.lower())[0].replace('select', '').replace('where', '').strip()
    # print("target: ", target)
    sparql = sparql.replace(target, '<label>')
    sql = re.sub('\?[a-z]', '<target>', sparql)
    sql = re.findall(r"([<\"].+?[>\"])", sql)
    if len(sql) % 3 != 1:
        return [], ''
    for i in range(len(sql)):
        if i in [1, 3, 4, 6, 7, 9] and sql[i] != '<target>' and sql[i] != '<label>':
            sql[i] = sql[i].replace('<', '').replace(">", '').replace('\"', '')
            entity.append(sql[i].strip())
    unknow_entity = ['<target>', '<label>']
    if len(sql) == 4:
        if sql[1] not in unknow_entity:
            if sql[3] == '<label>':
                template = '1aa' # 1052
            elif sql[2] == '<label>':
                template = '1ab' # 5
        elif sql[3] not in unknow_entity:
            if sql[1] == '<label>':
                template = '1ba' # 116
            elif sql[2] == '<label>':
                template = '1bb' # 0
        else :
            template = '1other' # 0
    elif len(sql) == 7:
        
        if sql[1] not in unknow_entity and sql[4] not in unknow_entity: # 向中间约束
            if sql[3] == '<label>' and sql[6] == '<label>':
                template = '2ca' # 14
            else:
                template = '2cother' # 8

        elif sql[1] not in unknow_entity and sql[4] in unknow_entity:
            if sql[6] == '<label>':
                template = '2aa' # 342
            elif sql[3] == '<label>' and sql[4] == '<label>':
                template = '2ab' # 44
            elif sql[3] != '<label>' and sql[4] == '<label>':
                template = '2cb' # 25
            else:
                template = '2aother' # 4
        elif sql[4] not in unknow_entity and sql[1] in unknow_entity:
            if sql[3] == '<label>':
                template = '2aa'
            elif sql[3] not in unknow_entity:
                template = '2ab'
            else:
                template = '2aother' 

        elif sql[1] in unknow_entity and sql[4] in unknow_entity:
            if sql[1] == '<label>' and sql[4] == '<label>':
                template = '2da' # 264
            elif sql[3] == '<label>' or sql[6] == '<label>':
                template = '2db' # 233
            else:
                template = '2dother' # 12
        else:
            template = '2other' # 0 
    else:
        entity = []
        template = '3'
    return entity, template

class Template():
    def __init__(self,  templates = ['1a', '2a', '2d']):
        self.templates = templates 
    
    def get_cyphers(self, entity):
        cyphers = []
        for template in self.templates:
            if template == '1a':
                cypher_1aa = "match(n:baike)-[r]->(m:baike) \
                            where n.name ='{}' and r.name<>'标签' \
                            return n.name, r.name, m.name".format(entity)
                cyphers.append({"template": "1a", "cypher": cypher_1aa, "sub_templates": ['1aa']})
            elif template == '2a':
                cypher_2aa = "match(n:baike)-[r1]->(q:baike)-[r2]->(m:baike) \
                            where n.name ='{}'  and r1.name<>'标签' and r2.name<>'标签' and r2.name<>'描述'\
                            return n.name, r1.name, m.name, r2.name, q.name".format(entity)
                cypher_2ab = "match(n:baike)-[r1]->(p:baike)<-[r3]-(q:baike)-[r2]->(m:baike)  \
                            where n.name ='{}' and r3.name='中文名' and r1.name<>'标签' and r2.name<>'标签' and r2.name<>'描述'\
                            return n.name, r1.name, m.name, r2.name, q.name".format(entity)
                cypher_2ac = "match(n:baike)<-[r3]-(q:baike)-[r1]->(q:baike)-[r2]->(m:baike) \
                            where n.name ='{}' and r3.name='中文名' and r1.name<>'标签' and r2.name<>'标签' and r2.name<>'描述'\
                            return n.name, r1.name, m.name, r2.name, q.name".format(entity)
                cyphers.append({"template": "2a", "cypher": cypher_2aa, "sub_templates": ['2aa']})
                cyphers.append({"template": "2a", "cypher": cypher_2ab, "sub_templates": ['2aa']})
                cyphers.append({"template": "2a", "cypher": cypher_2ac, "sub_templates": ['2aa']})
            elif template == '2d':
                cypher_2da = "match(n:baike)<-[r1]-(q:baike)-[r2]->(m:baike)  \
                            where n.name ='{}' and r1.name<>'标签' and r2.name<>'标签' and r2.name<>'描述' \
                            return n.name, r1.name, m.name, r2.name, q.name".format(entity)
                cyphers.append({"template": "2d", "cypher": cypher_2da, "sub_templates": ['2da']})
        return cyphers


def get_graph_template(entitys, used_template=['1a', '2a', '2d']):
    result = []
    template_cypher = Template(used_template)
    for entity in entitys:
        cyphers = template_cypher.get_cyphers(entity['candiate_entity'])
        for cypher in cyphers:
            cypher_result = graph.run(cypher['cypher']).data()
            if len(cypher_result) > 30000:
                continue
            sub_templates = cypher['sub_templates']
            feature = PathFeature()
            for res in cypher_result:
                for sub_template in sub_templates:
                    example = PathExample(res, sub_template=sub_template)
                    feature.add(example)
            for path in feature.answer:
                result.append(
                    {
                        "answer": '\t'.join(feature.answer[path]),
                         "template": cypher['template'], 
                         "path": path, 
                         "entity": entity['candiate_entity'], 
                         "mention": entity['mention'], 
                         'score': entity['score'] 
                         })
    return result


if __name__ == "__main__":
    start_time = time.time()
    entitys = [{'candiate_entity': '湖上草', 'score': 0.2635486125946045, 'mention': '湖上草'}]
    result = get_graph_template(entitys)
    print(len(result))
    print(result[:4])
    print("时间：", str(time.time()-start_time))