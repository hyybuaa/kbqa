import itertools
from copy import deepcopy

def duplientity_removal(result):
    dupliate_entitys = result.split('\t')
    if len(dupliate_entitys) != 1:
        entity_data = list(itertools.combinations(dupliate_entitys, 2))
        entity_query = []
        for data in entity_data:
            entity_query.append(deepcopy([data[0], data[1]]))
        scores = [1, 0.9, 0.95, 0.4]
        scores_index = [index if scores[index]>0.9 else -1 for index in range(len(scores))]
        scores_index = list(set(scores_index))
        scores_index.remove(-1)
        remove_entity = [entity_query[index][1] if len(entity_query[index][0])> len(entity_query[index][1]) else entity_query[index][0] for index in scores_index]
        for entity in remove_entity:
            dupliate_entitys.remove(entity)
    return '\t'.join(dupliate_entitys)

if __name__ == "__main__":
    result = "马云\t阿里巴巴\t看毛"
    print(duplientity_removal(result))
    print('完成')