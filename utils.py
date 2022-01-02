from __future__ import print_function

# these are from trade-dst, https://github.com/jasonwu0731/trade-dst
import os
import csv, json

import logging
from collections import defaultdict
import pdb
logger = logging.getLogger("my")
import pickle
    
    
""" Official evaluation script for v1.1 of the SQuAD dataset. [Changed name for external importing]"""
from collections import Counter
import string
import re
import argparse
import json
import sys



def idx_to_text(tokenizer, idx):
    pass
def dict_to_csv(data, file_name):
    w = csv.writer(open(f'./logs/csvs/{file_name}', "a"))
    for k,v in data.items():
        w.writerow([k,v])
    w.writerow(['===============','================='])
    
    
def dict_to_json(data, file_name):
    with open(f'./logs/jsons/{file_name}', 'a') as fp:
        json.dump(data, fp,  indent=4)



def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_punc(s))


def index_to_text(tokenizer, text):
    print(tokenizer.decode(text))


def cal_EM(answer_list, pred_list):
    good = 0
    for a,p in zip(answer_list, pred_list):
        if normalize_text(a) == normalize_text(p):
            good +=1
    return good*1.0/len(answer_list)

def cal_f1(answer_list, pred_list):
    f1 = 0
    
    for a,p in zip(answer_list, pred_list):
        answer_tokens = normalize_text(a).split()
        pred_tokens = normalize_text(p).split()
        common = Counter(answer_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            mini_f1 = 0
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(answer_tokens)
            mini_f1 = (2 * precision * recall) / (precision + recall)
        f1 += mini_f1
    return f1/len(answer_list)

            
            
