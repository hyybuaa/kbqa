import json
import re
from copy import deepcopy

import os
import time
from .feature import search_graph_feature, SimFeture
# from ..entity_recognition import get_all_entity
from .candiate_entity_similarity import predict_sim
import pandas as pd
import lightgbm as lgb

model_file = r'kbqa/saved_models/lgb_model_link.txt'
gbm_model = lgb.Booster(model_file=model_file)

class MentionCandiateEntity():
    def __init__(self, dict_path=None):
        if dict_path == None:
            dict_path = r'kbqa/entity_link/mention2entity.json'
        self.entity_dict = json.load(open(dict_path))

    def candiate_entitys(self, mention):
        if mention in self.entity_dict:
            candiate_entitys = self.entity_dict[mention]
        else:
            candiate_entitys = []
        return candiate_entitys


class CandiateEntityScorePair():
    def __init__(self, pairs_score, score):
        assert len(pairs_score) == len(score)
        self.candiate_entity_score = [{"candiate_entity": pairs_score[i]['candiate_entity'] , "score":score[i], "mention":pairs_score[i]['mention']} for i in range(len(score))]


class QuestionEntityPair():
    def __init__(self, question, entitys):
        self.question = question
        self.entitys = entitys
        self.pairs = [[question, entitys[i]]for i in range(len(entitys))]

    def score(self, score, mention):
        assert len(score) == len(self.pairs)
        self.pairs_score = [{"candiate_entity": self.pairs[j][1], "score": score[j], "mention": mention} for j in range(len(score))]
        return self.pairs_score


def predict_mention_entity(question, mentions, top_k=3, use_rank=False):
    mention_entity = MentionCandiateEntity()
    start_time = time.time()
    # mentions = get_all_entity(question)
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
        question_entity_sims = predict_sim(question_entity_pair.pairs)
        # question_entity_sims = gbm_model.predict(question_entity_pair.pairs, num_iteration=gbm_model.best_iteration)
        pairs_score = question_entity_pair.score(question_entity_sims, mention)
        # question_entity_score = []
        # for i in range(len(question_entity_pair)):
        #     question_entity_score.append(deepcopy({"question_entity":question_entity[i], "sim": question_entity_sims[i]}))
        pairs_score = sorted(pairs_score, key=lambda x: x['score'], reverse=True)
        pairs_score = pairs_score[:2]
        if use_rank:
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
            candiate_entity_rank_score = gbm_model.predict(text_x, num_iteration=gbm_model.best_iteration)
            # candiate_entity_rank_score = predict(text_x)
            candiate_entity_score_pair = CandiateEntityScorePair(pairs_score, candiate_entity_rank_score)
            # candiate_entity_score = [{"candiate_entity": pairs_score[i]['pair'][1] , "score": candiate_entity_rank_score[i]} for i in range(len(candiate_entity_rank_score))]
            # for i in range(len(candiate_entity_list)):
            #     candiate_entity_score.append(deepcopy({"candiate_entity": candiate_entity_rank_score[i], "score": pred_y[i], "mention": mention}))
            candiate_entity_list.extend(candiate_entity_score_pair.candiate_entity_score)
            print("精排候选实体时间：", str(time.time()-candiate_time))
        else:
            candiate_entity_list.extend(pairs_score)
    candiate_entity_list = sorted(candiate_entity_list, key=lambda x: x['score'], reverse=True)
    candiate_entity_list = candiate_entity_list[:top_k]
    return candiate_entity_list


if __name__ == "__main__":
    start_time = time.time()
    question = r'香港的英文名'
    mentions = ['香港', '英文名']
    result = predict_mention_entity(question, mentions, use_rank=False)
    print(result)
    print("完成")
    print("时间：", str(time.time()-start_time))
        

    
    