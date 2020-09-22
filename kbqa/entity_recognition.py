from models.BERT_CRF import BertCrf
from models.NER_main import NerProcessor, CRF_LABELS

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from tqdm import tqdm, trange
import requests
import json
import jieba.analyse
from config  import configs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ner_processor = NerProcessor()
config_file = configs['config']
vocab_file = configs['vocab']
ner_mode_path = configs['ner_model']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_length = 128

tokenizer_inputs = ()
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': max_length,
                    'vocab_file': vocab_file}
tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

def get_ner_model(config_file=config_file, pre_train_model=ner_mode_path, label_num = len(ner_processor.get_labels())):
    model = BertCrf(config_name=config_file,
                    num_tags=label_num, batch_first=True)
    model.load_state_dict(torch.load(pre_train_model))
    model.eval()
    return model.to(device)
ner_model = get_ner_model()
def get_entity(sentence, model=ner_model, tokenizer=tokenizer,  max_len=max_length):
    pad_token = 0
    sentence_list = list(sentence.strip().replace(' ', ''))
    text = " ".join(sentence_list)
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        # We're truncating the first sequence in priority if True
        truncate_first_sequence=True
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    labels_ids = None

    assert len(input_ids) == max_len, "Error with input length {} vs {}".format(
        len(input_ids), max_len)
    assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(
        len(attention_mask), max_len)
    assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(
        len(token_type_ids), max_len)

    input_ids = torch.tensor(input_ids).reshape(1, -1).to(device)
    attention_mask = torch.tensor(attention_mask).reshape(1, -1).to(device)
    token_type_ids = torch.tensor(token_type_ids).reshape(1, -1).to(device)
    labels_ids = labels_ids

    model = model.to(device)
    model.eval()
    # 由于传入的tag为None，所以返回的loss 也是None
    ret = model(input_ids=input_ids,
                tags=labels_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    pre_tag = ret[1][0]
    assert len(pre_tag) == len(sentence_list) or len(pre_tag) == max_len - 2

    pre_tag_len = len(pre_tag)
    b_loc_idx = CRF_LABELS.index('B-LOC')
    i_loc_idx = CRF_LABELS.index('I-LOC')
    o_idx = CRF_LABELS.index('O')

    if b_loc_idx not in pre_tag and i_loc_idx not in pre_tag:
        print("没有在句子[{}]中发现实体".format(sentence))
        return ''
    if b_loc_idx in pre_tag:

        entity_start_idx = pre_tag.index(b_loc_idx)
    else:

        entity_start_idx = pre_tag.index(i_loc_idx)
    entity_list = []
    entity_list.append(sentence_list[entity_start_idx])
    for i in range(entity_start_idx+1, pre_tag_len):
        if pre_tag[i] == i_loc_idx:
            entity_list.append(sentence_list[i])
        else:
            break
    return "".join(entity_list)


def get_entity_keyword(query, top_k=3):
    pos = ['ns', 'nr', 'nt', 'nw']
    result = jieba.analyse.extract_tags(query, topK=top_k, withWeight=False, allowPOS=pos)
    return result

def get_all_entity(query):
    mention = get_entity(query)
    mention_keyword = get_entity_keyword(query)
    mention_list = list(mention_keyword)
    mention_list.append(mention)
    return list(set(mention_list))

if __name__ == "__main__":
    query = r'中国首都'
    result = get_all_entity(query)
    # result = get_entity_keyword(query)
    # result_api = get_ner_api(query)
    print(result)
