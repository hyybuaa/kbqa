import json
import re
from copy import deepcopy
from process_ccks_data import DataPath
from random import sample
import os
import time
from web import get_web_ner, get_web_sim
from utils import search_graph_feature
from caculate_feature import SimFeture
import pandas as pd
from lightgbm_entity_link import predict

class MentionCandiateEntity():
    def __init__(self, dict_path=None):
        if dict_path == None:
            dict_path = r'构建数据/dict/mention2entity.json'
        self.entity_dict = json.load(open(dict_path))

    def candiate_entitys(self, mention):
        if mention in self.entity_dict:
            candiate_entitys = self.entity_dict[mention]
        else:
            candiate_entitys = []
        return candiate_entitys


class CandiateEntityScorePair():
    def __init__(self, pairs_score, score, mention):
        assert len(pairs_score) == len(score)
        self.candiate_entity_score = [{"candiate_entity": pairs_score[i]['candiate_entity'] , "score":score[i], "mention": mention} for i in range(len(score))]


class QuestionEntityPair():
    def __init__(self, question, entitys):
        self.question = question
        self.entitys = entitys
        self.pairs = [[question, entitys[i]]for i in range(len(entitys))]

    def score(self, score):
        assert len(score) == len(self.pairs)
        self.pairs_score = [{"candiate_entity": self.pairs[j][1], "score": score[j]} for j in range(len(score))]
        return self.pairs_score



def get_candiate_entity_corpus(entity_dir):
    mention_entity = MentionCandiateEntity()
    train_candiate = DataPath('构建数据/ccks_kbqa', type='candiate_entity')
    with open(entity_dir, encoding='utf-8') as f, open(train_candiate.dev_path, 'w', encoding='utf-8') as train_candiate_file:
        lines = f.readlines()
        guid = 0
        for line in lines:
            line = json.loads(line.strip())
            mention = line['mention']
            entity = line['entity']
            query = line['query']
            if entity != mention:
                content = str(guid) + "\t" + query + "\t" + entity + "\t" + "1" +"\n"
                train_candiate_file.write(content)
                guid += 1
            mention_candiate_entitys = mention_entity.candiate_entitys(mention)
            candiate_entity_tok5 = sample(mention_candiate_entitys, 5) if len(mention_candiate_entitys) > 5 else mention_candiate_entitys
            for candiate_entity in candiate_entity_tok5:
                if candiate_entity == entity:
                    continue
                content = str(guid) + "\t" + query + "\t" + candiate_entity + "\t" + "0" +"\n"
                train_candiate_file.write(content)
                guid += 1
    return

def caculate_mention_entity_sim(question_dir):
    train_data = DataPath('构建数据/ccks_kbqa/实体链接数据', 'candiate_entity_sim')
    mention_entity = MentionCandiateEntity()
    with open(question_dir, encoding='utf-8') as f, open(train_data.train_path, 'w', encoding='utf-8') as train_mention_entity_file:
        lines = f.readlines()
        number = 0
        for line in lines:
            if line == '\n':
                continue
            line = json.loads(line.strip())
            question = line['query']
            mentions = get_web_ner(question)
            # print("mention: ", mentions)
            # mention_entity_list = []
            for mention in mentions:
                question_entity = []
                mention_candiate_entitys = mention_entity.candiate_entitys(mention)
                candiate_entity_tok5 = sample(mention_candiate_entitys, 5) if len(mention_candiate_entitys) > 5 else mention_candiate_entitys
                if mention not in mention_candiate_entitys:
                    mention_candiate_entitys.append(mention)
                for candiate_entity in mention_candiate_entitys:
                    question_entity.append(deepcopy([question, candiate_entity]))
                
                # print("question_entity: ", question_entity)
                question_entity_sims = get_web_sim(question_entity)
                question_entity_score = []
                for i in range(len(question_entity)):
                    question_entity_score.append(deepcopy({"question_entity":question_entity[i], "sim": question_entity_sims[i]}))
                question_entity_score = sorted(question_entity_score, key=lambda x: x['sim'], reverse=True)
                question_entity_score_total = question_entity_score[:2]
                for i in range(len(question_entity_score_total)):
                    score = 0
                    candiate_entity_end = deepcopy(question_entity_score_total[i]['question_entity'][1])
                    query = deepcopy(question_entity_score_total[i]['question_entity'][0])
                    sim = deepcopy(question_entity_score_total[i]['sim'])
                    entitys = [entity.replace('_', '') for entity in line['entitys']]
                    if candiate_entity_end in entitys:
                        score = 1
                    graph_hot = search_graph_feature(candiate_entity_end)
                    if 0 in graph_hot:
                        score = 0
                    content = {"score": score, "query": query, "mention": mention, "graph_hot": graph_hot, "entity": entitys, "candiate_entity": candiate_entity_end,  "sim": sim}
                    train_mention_entity_file.write(json.dumps(content, ensure_ascii=False))
                    train_mention_entity_file.write("\n")
            number += 1
            print("number：", number)
    return

def generate_mention_entity_corpus(sim_dir):
    train_mention_entity_rank = DataPath('构建数据/ccks_kbqa/实体链接数据', 'entity_rank')
    with open(sim_dir, encoding='utf-8') as f, open(train_mention_entity_rank.train_path, 'w', encoding='utf-8') as train_mention_entity_rank_file:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            mention = line['mention']
            candiate_entity = line['candiate_entity']
            query = line['query']
            graph_feature = line['graph_hot']
            score = line['score']
            text_sim = line['sim']
            sim_feature = SimFeture(query, candiate_entity , mention)
            features = sim_feature.feature
            features.append(text_sim)
            features.extend(graph_feature)
            content = {"query": query, "candiate_entity": candiate_entity, "feature": features, "score": score}
            train_mention_entity_rank_file.write(json.dumps(content, ensure_ascii=False))
            train_mention_entity_rank_file.write("\n")
    return

def predict_mention_entity_by_path(feature_dir):
    train_data = DataPath('构建数据/ccks_kbqa/实体链接数据', 'link_predict')
    mention_entity = MentionCandiateEntity()
    with open(feature_dir, encoding='utf-8') as f, open(train_data.train_path, 'w', encoding='utf-8') as link_predict_file:
        lines = f.readlines()
        number = 0
        for line in lines:
            if line == '\n':
                continue
            line = line.replace('_', '')
            line = json.loads(line.strip())
            question = line['query']
            entitys = line['entitys']
            candiate_entitys = predict_mention_entity(question)
            score = 0
            for candiate_entity in candiate_entitys:
                if candiate_entity in entitys:
                    score = 1
                    continue
            content = {"query": question, "score": score, "candiate": candiate_entity, "entity": entitys}
            link_predict_file.write(json.dumps(content, ensure_ascii=False))
            link_predict_file.write("\n")
            number += 1
            print("number: ", number)
    return

def predict_mention_entity(question, top_k=2):
    mention_entity = MentionCandiateEntity()
    start_time = time.time()
    mentions = get_web_ner(question)
    ner_time = time.time()
    print("ner时间：", str(ner_time-start_time))
    candiate_entity_list = []
    for mention in mentions:
        mention_time = time.time()
        mention_candiate_entitys = mention_entity.candiate_entitys(mention)
        candiate_time = time.time()
        print("得到候选实体时间：{:.8f}".format(candiate_time-mention_time))
        # candiate_entity_tok5 = sample(mention_candiate_entitys, 5) if len(mention_candiate_entitys) > 5 else mention_candiate_entitys
        if mention not in mention_candiate_entitys:
            mention_candiate_entitys.append(mention)
        question_entity_pair = QuestionEntityPair(question, mention_candiate_entitys)
        # question_entity_pairs = [QuestionEntityPair(question, candiate_entity).pair for candiate_entity in mention_candiate_entitys]
        # for candiate_entity in mention_candiate_entitys:
        #     question_entity.append(deepcopy([question, candiate_entity]))
        question_entity_sims = get_web_sim(question_entity_pair.pairs)
        pairs_score = question_entity_pair.score(question_entity_sims)
        # question_entity_score = []
        # for i in range(len(question_entity_pair)):
        #     question_entity_score.append(deepcopy({"question_entity":question_entity[i], "sim": question_entity_sims[i]}))
        pairs_score = sorted(pairs_score, key=lambda x: x['score'], reverse=True)
        pairs_score = pairs_score[:2]
        text_x = []
        for i in range(len(pairs_score)):
            candiate_entity = pairs_score[i]['candiate_entity']
            # candiate_entity_list.append(candiate_entity)
            graph_hot = search_graph_feature(candiate_entity)
            print("graph: {:.5f}".format(time.time()-candiate_time))
            sim = pairs_score[i]['score']
            sim_feature = SimFeture(question, candiate_entity , mention)
            features = sim_feature.feature
            features.append(sim)
            features.extend(graph_hot)
            text_x.append(features)
        candiate_entity_rank_score = predict(text_x)
        candiate_entity_score_pair = CandiateEntityScorePair(pairs_score, candiate_entity_rank_score, mention)
        # candiate_entity_score = [{"candiate_entity": pairs_score[i]['pair'][1] , "score": candiate_entity_rank_score[i]} for i in range(len(candiate_entity_rank_score))]
        # for i in range(len(candiate_entity_list)):
        #     candiate_entity_score.append(deepcopy({"candiate_entity": candiate_entity_rank_score[i], "score": pred_y[i], "mention": mention}))
        candiate_entity_list.extend(candiate_entity_score_pair.candiate_entity_score)
        print("精排候选实体时间：", str(time.time()-candiate_time))
    candiate_entity_list = sorted(candiate_entity_list, key=lambda x: x['score'], reverse=True)
    candiate_entity_list = candiate_entity_list[:top_k]
    return candiate_entity_list


if __name__ == "__main__":
    start_time = time.time()
    entity_dir = r'构建数据/ccks_kbqa/dev_entity.txt'
    # get_candiate_entity_corpus(entity_dir)
    question_dir = r'构建数据/ccks_kbqa/train_feature.txt'
    # caculate_mention_entity_sim(question_dir)
    sim_dir = r'构建数据/ccks_kbqa/实体链接数据/train_candiate_entity_sim.txt'
    # generate_mention_entity_corpus(sim_dir)
    # predict_mention_entity_by_path(question_dir)
    question = r'香港的英文名'
    result = predict_mention_entity(question)
    print(result)
    print("完成")
    print("时间：", str(time.time()-start_time))
        

    
    