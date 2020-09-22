import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')


import argparse
from collections import Counter
import code
import os
import logging
from tqdm import tqdm, trange
import random
import codecs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW,  get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample

import numpy as np
import pandas as pd
from sklearn import metrics
import json
from config import configs

logger = logging.getLogger(__name__)

model_config = configs['config']
vocab_file = configs['vocab']
pre_train_model  = configs['path_similarity_mode']
tokenizer_inputs = ()
max_seq_length = 128
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': max_seq_length,
                    'vocab_file': vocab_file}
tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)
bert_config = BertConfig.from_pretrained(model_config)
bert_config.num_labels = 2
model_kwargs = {'config':bert_config}

model = BertForSequenceClassification.from_pretrained(pre_train_model, **model_kwargs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model = model.to(device)
eval_batch_size = 1024

class SimInputExample(object):
    def __init__(self, guid, question,attribute, label=None):
        self.guid = guid
        self.question = question
        self.attribute = attribute
        self.label = label


class SimInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class SimProcessor(DataProcessor):
    """Processor for the FAQ problem
        modified from https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py#L154
    """
    def get_test_examples(self,datas):
        logger.info("*******  test  ********")
        return self._create_examples(datas)

    def get_labels(self):
        return [0, 1]

    @classmethod
    def _create_examples(cls, datas):
        examples = []
        uid = 0
        for tokens in datas:
            uid += 1
            if 2 == len(tokens):
                examples.append(
                    SimInputExample(guid=int(uid),
                                    question=tokens[0],
                                    attribute=tokens[1],
                                    label=int(0))
                )
        return examples


def sim_convert_examples_to_features(examples,tokenizer,
                                     max_length=512,
                                     label_list=None,
                                     pad_token=0,
                                     pad_token_segment_id = 0,
                                     mask_padding_with_zero = True):

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            text = example.question,
            text_pair= example.attribute,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True  # We're truncating the first sequence in priority if True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)


        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),max_length)

        # label = label_map[example.label]
        label = example.label


        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s " % str(label))

        features.append(
            SimInputFeatures(input_ids,attention_mask,token_type_ids,label)
        )
    return features


def load_and_cache_example(data,tokenizer,processor):
    label_list = processor.get_labels()
    examples = processor.get_test_examples(data)

    features = sim_convert_examples_to_features(examples=examples,tokenizer=tokenizer,max_length=max_seq_length,label_list=label_list)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return dataset


def evaluate(model, eval_dataset):

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_pred_label = []   # 记录所有的预测标签列表
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        label_ids = batch[3].to(device)
        with torch.no_grad():
            inputs = {'input_ids': input_ids,
                      'attention_mask': attention_mask,
                      'token_type_ids': token_type_ids,
                      'labels': label_ids,
            }
            # outputs = model(**inputs)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label_ids)
            _, logits = outputs[0], outputs[1]
            logits = logits.softmax(dim=-1)
            # pred = logits.argmax(dim=-1).tolist()     # 得到预测的label转为list
            # score = logits[:,1:].max().item()
            logits = logits.tolist()
            logits_all = [logit[1] for logit in logits]
            
            all_pred_label.extend(logits_all)                        # 记录预测的 label
    return all_pred_label


def predict_sim_rank(data):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


    processor = SimProcessor()
    eval_dataset = load_and_cache_example(data, tokenizer, processor)
    pred_label = evaluate(model, eval_dataset)
    return pred_label



if __name__ == '__main__':
    data = [['钟南山', '钟南山描述'], ['英国探险家弗朗西斯·德雷克出生地是哪里？', '德雷克（历史人物）出生地'], ['美国独立日设定地区有多少人', '美国独立日（美国国庆日）设定地点费城人口']]
    result = predict_sim_rank(data)
    print(result)

