import json
import time
from py2neo import Graph
import re
import Levenshtein
import difflib

graph = Graph("bolt://127.0.0.1:7688", username="neo4j", password="123456")


def search_graph_feature(entity):
    cypher_sql = "match(n:baike)-[r]->(m:baike) where n.name ='{}' return r.name".format(entity)
    result_list, cypher_result = [], []
    try:
        cypher_result = graph.run(cypher_sql).data()
        result_list = [res['r.name'] for res in cypher_result]
    except Exception as e:
        print(e)
    
    # result = [len(set(result_list)), len(cypher_result)]
    return  [len(set(result_list)), len(cypher_result)]

def max_common_string(s1,s2):
    if len(s1) < len(s2):
        s1,s2 = s2,s1
    maxstr = s1
    substr_maxlen = max(len(s1),len(s2))
    for sublen in range(substr_maxlen,-1,-1):
        for i in range(substr_maxlen-sublen+1):
            if maxstr[i:i+sublen] in s2:
                return maxstr[i:i+sublen]

question_words = ['谁', '什么', '哪儿', '哪里', '几时',  '多少', '多大', '多高', '何时',
                     '何地', '怎么', '怎的', '怎样', '怎么样', '怎么着', '如何', '为什么', 
                     '哪', '多', '何', '怎', '吗', '呢', '吧', '啊']
def extractor_question_words(query):
    for question_word in question_words:
        question_word_index = query.find(question_word) 
        if question_word_index != -1:
            return [question_word_index, question_word_index+len(question_word)]
    return [0, 0]

def path_rule_score(pairs_score, query):
    discrible_words = ['是什么', '是谁', '怎么样']
    score = 0
    for word in discrible_words:
        if word in query:
            score = 0.9
            continue
    if score > 0:
        discribles = []
        key_value = ''
        for i in range(len(pairs_score)):
            if pairs_score[i]['cypher_result']['path'] == '描述':
                pairs_score[i]['score'] = score
                print(pairs_score[i]['cypher_result'])
                key_value = '描述'
            if key_value != '描述' and pairs_score[i]['cypher_result']['path'] != '标签':
                key_value = pairs_score[i]['cypher_result']['path'] + ":" + pairs_score[i]['cypher_result']['answer']
                discribles.append(key_value)
        if len(discribles) > 0:
            pairs_score.append({"cypher_result":
                                    {"answer": '、'.join(discribles), "template": '1aa', "path": '详细信息', 'entity': pairs_score[0]['cypher_result']['entity']},
                                "score": score})

    # print(pairs_score)
    return pairs_score


class Similarity():
    def __init__(self, str1, str2):
        self.difflib_sim = difflib.SequenceMatcher(None, str1, str2).quick_ratio()
        self.edit_sim = Levenshtein.ratio(str1, str2)
        self.jaro_sim = Levenshtein.jaro_winkler(str1, str2)


def caculate_entity_feature(entity):
    result_special_char = re.findall("[《（“(\"]", entity)
    result_number = re.findall("[0-9]", entity)
    result_english = re.findall("[a-zA-Z]", entity)
    char_flag = 1 if len(result_special_char) > 0 else 0
    english_flag = 1 if len(result_english) > 0 else 0
    number_flag = 1 if len(result_number) > 0 else 0
    result = [char_flag, number_flag, english_flag]
    return result

def caculate_match_feature(query, entity):
    query_length = len(query)
    entity_length = len(entity)
    entity_in_query = 0
    if entity in query:
        entity_in_query = 1
    common_string_entity = len(max_common_string(query, entity))/(len(query) + len(entity))
    result = [query_length, entity_length, entity_in_query, common_string_entity]
    return result

def caculate_match_mention_query(query, mention):
    mention_index = query.find(mention)
    mention_index_feature = [mention_index, mention_index+len(mention)]
    mention_query_feature = mention_index/len(query)
    question_word_index = extractor_question_words(query)
    mention_question_distance = question_word_index[0] - mention_index_feature[1] if mention_index < question_word_index[0] else mention_index - question_word_index[1]
    if question_word_index[0] == mention_index or question_word_index[1] == mention_index_feature[1]:
        mention_question_distance = 0
    mention_start_or_end = 1 if mention_index ==0 or mention_index == len(query) else 0
    common_string_mention = len(mention)/len(query)
    return [mention_query_feature, mention_question_distance, common_string_mention, mention_start_or_end]

def classifier_question(question):
    person = ['谁', '作者', '人']
    location = ['哪国人', '哪里', '在哪', '地址', '位置']
    time = ['时候', '日期', '时代']
    number = ['多少', '多高', '多大', '', '']
    works = ['代表作品', '奖项', '']
    return []

def classifier_answer(answer):
    return []

def caculate_path_query(path, query):
    return

class SimFeture():
    def __init__(self, query, entity, mention):
        self.query = query
        self.entity = entity
        self.entity_feature = caculate_entity_feature(entity)
        self.query_entity = caculate_match_feature(query, entity)
        self.query_mention = caculate_match_mention_query(query, mention)
        self.feature = self.entity_feature + self.query_entity + self.query_mention


class PathSimFeature():
    def __init__(self, query, path, mention, answer):
        path_mention_similarity = Similarity(path, mention)
        path_query_similarity = Similarity(path, query)
        self.path_mention_feature = [path_mention_similarity.edit_sim, 
                                    path_mention_similarity.difflib_sim, 
                                    path_mention_similarity.jaro_sim]
        self.path_query_similarity = [path_query_similarity.edit_sim,
                                    path_query_similarity.difflib_sim,
                                    path_query_similarity.jaro_sim]
        self.answer = classifier_answer(answer)
        self.query = classifier_question(query)
        self.feature = self.path_query_similarity +self.path_mention_feature + self.query + self.answer


if __name__ == "__main__":
    start_time = time.time()
    # print(result)
    print("时间：", str(time.time() - start_time))