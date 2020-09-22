from flask import Flask, Response, request
from flask_cors import CORS
import logging
import json
import os
# Setup logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
from kbqa.entity_recognition import get_all_entity
from kbqa.entity_link.candiate_entity_rank import predict_sim, predict_mention_entity
from kbqa.graph_query import get_graph_template
from kbqa.path_rank.path_rank import predict_path_rank, predict_sim_rank
from kbqa.entity_normalize import duplientity_removal

logging.info("kbqa服务启动成功..")
logging.info("app服务当前工作路径：%s", str(os.getcwd()))
port = 6768
# app_port = os.getenv("app_port")
app = Flask(__name__)
CORS(app, supports_credentials=True)
logger = logging.getLogger(__name__)

def kbqa_query(query):
    logger.info("开始抽取实体...")
    mention_list = get_all_entity(query)
    if len(mention_list) == 0:
        result = {}
        return result
    logger.info("开始生成候选实体...")
    candiate_entitys = predict_mention_entity(query, mention_list)
    logger.info("开始neo4j模版查询...")
    cypher_result = get_graph_template(candiate_entitys)
    # print(cypher_result[:4])
    if len(cypher_result) == 0:
        result = {}
        return result
    logger.info("开始查询结果排序...")
    result = predict_path_rank(query, cypher_result)
    logger.info("查询完成！")
    result = duplientity_removal(result)
    logger.info("查询结果归一化完成！")
    return result


@app.route('/query', methods=["POST", "OPTIONS"])
def query() -> Response:
    data = request.get_json(force=True)['data']
    result = kbqa_query(data)
    # result = correctResult(result)
    result = {'code': 200, 'data': result}
    return Response(json.dumps(result, ensure_ascii=False), mimetype="application/json; charset=UTF-8", status=200)

@app.route('/ner', methods=["POST", "OPTIONS"])
def ner() -> Response:
    data = request.get_json(force=True)['data']
    result = get_all_entity(data)
    # result = correctResult(result)
    result = {'code': 200, 'data': result}
    return Response(json.dumps(result, ensure_ascii=False), mimetype="application/json; charset=UTF-8", status=200)

@app.route('/link', methods=["POST", "OPTIONS"])
def link() -> Response:
    data = request.get_json(force=True)['data']
    result = predict_sim(data)
    # result = correctResult(result)
    result = {'code': 200, 'data': result}
    return Response(json.dumps(result, ensure_ascii=False), mimetype="application/json; charset=UTF-8", status=200)

@app.route('/rank', methods=["POST", "OPTIONS"])
def rank() -> Response:
    data = request.get_json(force=True)['data']
    result = predict_sim_rank(data)
    # result = correctResult(result)
    result = {'code': 200, 'data': result}
    return Response(json.dumps(result, ensure_ascii=False), mimetype="application/json; charset=UTF-8", status=200)

@app.route('/dupli_entity', methods=["POST", "OPTIONS"])
def dupli() -> Response:
    data = request.get_json(force=True)['data']
    result = duplientity_removal(data)
    # result = correctResult(result)
    result = {'code': 200, 'data': result}
    return Response(json.dumps(result, ensure_ascii=False), mimetype="application/json; charset=UTF-8", status=200)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, threaded=True)
    
