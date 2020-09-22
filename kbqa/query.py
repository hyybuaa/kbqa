import logging
import time
import json

from entity_recognition import get_all_entity
from entity_link.candiate_entity_rank import predict_mention_entity
from graph_query import get_graph_template
from path_rank.path_rank import predict_path_rank
from entity_normalize import duplientity_removal


logger = logging.getLogger(__name__)

def kbqa_query(query):
    if query == "":
        result = {
                "score" : 0,
                "query": query,
                "entity": "",
                "intent": "",
                "answer": "你没有向我提问？",
                "templateMatch": None,
                "tag": "智源"
            }
        return result
    logger.info("开始抽取实体...")
    mention_list = get_all_entity(query)
    if len(mention_list) == 0:
        result = {
                "score" : 0,
                "query": query,
                "entity": "",
                "intent": "",
                "answer": "没有抽取到实体",
                "templateMatch": None,
                "tag": "智源"
            }
        return result
    logger.info("开始生成候选实体...")
    candiate_entitys = predict_mention_entity(query, mention_list)
    print(candiate_entitys)
    logger.info("开始neo4j模版查询...")
    cypher_result = get_graph_template(candiate_entitys)
    print(cypher_result[:4])
    if len(cypher_result) == 0:
        result = {
                "score" : 0,
                "query": query,
                "entity": mention_list,
                "intent": "",
                "answer": "没有链接到实体",
                "templateMatch": None,
                "tag": "智源"
            }
        return result
    logger.info("开始查询结果排序...")
    result_rank = predict_path_rank(query, cypher_result)
    logger.info("查询完成！")
    result_dupli = duplientity_removal(result_rank)
    logger.info("查询结果归一化完成！")
    result = {
                "score" : result_dupli['score'],
                "query": query,
                "entity": result_dupli['cypher_result']['entity'],
                "mention": result_dupli['cypher_result']['mention'],
                "intent": result_dupli['cypher_result']['path'],
                "answer": result_dupli['cypher_result']['answer'],
                "templateMatch": result_dupli['cypher_result']['template'],
                "tag": "智源"
            }
    return result

if __name__ == "__main__":
    start_time = time.time()
    query = r'中国的首都有多少人口？'
    # query = r'武汉大学出了多少科学家'
    # query = r'中国和北京什么关系？'
    query = r'马云是谁'
    result = kbqa_query(query)
    print("时间：", str(time.time()-start_time))
    print(result)