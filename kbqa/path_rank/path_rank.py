import json
import re
from copy import deepcopy

from random import sample
import os
import time

import logging
from .feature import path_rule_score, PathSimFeature
from .path_similarity import predict_sim_rank
import lightgbm as lgb

model_file = r'kbqa/saved_models/lgb_model_path_rank.txt'
gbm_model = lgb.Booster(model_file=model_file)

class QusetionPathScorePair():
    def __init__(self, pairs_score, score):
        assert len(pairs_score) == len(score)
        self.path_rank_score = [{"cypher_result": pairs_score[i]['cypher_result'] , "score":score[i]} for i in range(len(score))]


class QuestionPathPair():
    def __init__(self, question, cypher_result):
        self.question = question
        # self.pairs = [[question, cypher_result[i]['path']]for i in range(len(cypher_result))]
        pairs = []
        for i in range(len(cypher_result)):
            if cypher_result[i]['template'] != '1a':
                pairs.append([question, cypher_result[i]['path'].replace('\t', '')])
                pairs.append([question, cypher_result[i]['path'].split('\t')[0]])
                pairs.append([question, cypher_result[i]['path'].split('\t')[1]])
            else:
                pairs.append([question, cypher_result[i]['path']])
                pairs.append([question, ''])
                pairs.append([question, ''])
        self.pairs = pairs
        self.cypher_result = cypher_result

    def score(self, score):
        assert len(score) == len(self.pairs)
        assert len(score) == len(self.cypher_result) * 3
        # self.pairs_score = [{"cypher_result": self.cypher_result[j], "score": score[j]} for j in range(len(score))]
        self.pairs_score = []
        for j in range(len(self.cypher_result)):
            if self.cypher_result[j]['template'] != '1a':
                self.pairs_score.append({"cypher_result": self.cypher_result[j], "score": [score[j*3], score[j*3+1], score[j*3+2]]})
            else:
                self.pairs_score.append({"cypher_result": self.cypher_result[j], "score": [score[j*3], 0.8, 0.8]} )
        return self.pairs_score


def predict_path_rank(query, cypher_result, use_rank=True):
    test_x = []
    question_path_pair = QuestionPathPair(query, cypher_result)
    question_path_sims = predict_sim_rank(question_path_pair.pairs)
    pairs_score = question_path_pair.score(question_path_sims)
    pairs_score = path_rule_score(pairs_score, query)
    # pairs_score = sorted(pairs_score, key=lambda x: x['score'][0] + x['score'][1] + x['score'][2], reverse=True)
    # pairs_score = pairs_score[:100]
    if len(pairs_score) > 3:
        pairs_score = sorted(pairs_score, key=lambda x: x['score'][0], reverse=True)
    pairs_score = pairs_score[:2]
    # path_entity_pair.extend(pairs_score)
    if use_rank:
        for i in range(len(pairs_score)):
            res = pairs_score[i]['cypher_result']
            path = res['path']
            answer = res['answer']
            mention = res['mention']
            entity_score = res['score']
            sim_feature = PathSimFeature(query, path, mention, answer)
            path_sim = pairs_score[i]['score']
            features = sim_feature.feature
            features.extend(path_sim[:1])
            features.append(entity_score)
            test_x.append(features)
        # path_rank_score = predict_path(text_x)
        path_rank_score = gbm_model.predict(test_x, num_iteration=gbm_model.best_iteration)
        question_path_score_pair = QusetionPathScorePair(pairs_score, path_rank_score)
        candiate_path_list = sorted(question_path_score_pair.path_rank_score , key=lambda x: x['score'], reverse=True)
    else:
        candiate_path_list = pairs_score
    # candiate_entity_list = candiate_entity_list[:top_k]
    print(candiate_path_list[:4])
    return candiate_path_list[0] if len(candiate_path_list) > 0 else {}

def predict_path_rank_list(query, cypher_result):
    test_x = []
    question_path_pair = QuestionPathPair(query, cypher_result)
    question_path_sims = predict_sim_rank(question_path_pair.pairs)
    pairs_score = question_path_pair.score(question_path_sims)
    pairs_score = path_rule_score(pairs_score, query)
    pairs_score = sorted(pairs_score, key=lambda x: x['score'], reverse=True)
    pairs_score = pairs_score[:2]
    # path_entity_pair.extend(pairs_score)
    for i in range(len(pairs_score)):
        res = pairs_score[i]['cypher_result']
        path = res['path']
        answer = res['answer']
        mention = res['mention']
        entity_score = res['score']
        sim_feature = PathSimFeature(query, path, mention, answer)
        path_sim = pairs_score[i]['score']
        features = sim_feature.feature
        features.append(path_sim)
        features.append(entity_score)
        test_x.append(features)
    # path_rank_score = predict_path(text_x)
    path_rank_score = gbm_model.predict(test_x, num_iteration=gbm_model.best_iteration)
    question_path_score_pair = QusetionPathScorePair(pairs_score, path_rank_score)
    candiate_path_list = sorted(question_path_score_pair.path_rank_score , key=lambda x: x['score'], reverse=True)
    # candiate_entity_list = candiate_entity_list[:top_k]
    return candiate_path_list


if __name__ == "__main__":
    start_time = time.time()
    # template_dir = r'构建数据/ccks_kbqa/模版分类数据/train_1aa.txt'
    # entity = r'洪七公'
    # # result = get_graph_template(entity, template='1aa')
    # feature_dir = r'构建数据/ccks_kbqa/train_feature.txt'
    # generate_path_rank_corpus(feature_dir)
    query = r'香港的英文名'
    cypher_result = [
                    {
                        'answer': 'Hong Kong', 
                        'template': '1a', 
                        'path': '外文名称', 
                        'entity': '香港（中国特别行政区）', 
                        'mention': '香港', 
                        'score': 0.17222082860199425}, 
                   
                            ]
    result = predict_path_rank(query, cypher_result)
    print(result)
    print("完成")
    print("时间：", str(time.time()-start_time))