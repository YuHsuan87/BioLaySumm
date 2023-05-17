import os, sys, json

import textstat
import numpy as np
from rouge_score import rouge_scorer
import nltk

from utils import *

def cal_rouge(preds, refs):
    # Get ROUGE F1 scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
    return np.mean([s['rouge1'].fmeasure for s in scores]), \
           np.mean([s['rouge2'].fmeasure for s in scores]), \
           np.mean([s['rougeLsum'].fmeasure for s in scores])

def cal_readability(preds):
    # Get readability scores
    fkgl_scores = []
    dcrs_scores = []
    for pred in preds:
        fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
        dcrs_scores.append(textstat.dale_chall_readability_score(pred))
    return np.mean(fkgl_scores), np.mean(dcrs_scores) 

def read_file_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    if path.endswith('.jsonl'):
        lines = [json.loads(line) for line in lines]

    return lines

def evaluation(pred_path, ref_path):
    ref_dic = read_file_lines(ref_path)
    refs = [d['lay_summary'] for d in ref_dic]
    preds = read_file_lines(pred_path)

    score_dict = {}

    # rouge_score
    rg1, rg2, rgl = cal_rouge(preds, refs)
    score_dict['rouge1'] = round(rg1, 4)
    score_dict['rouge2'] = round(rg2, 4)
    score_dict['rougel'] = round(rgl, 4)
    # readability_score
    fkgl, dcrs = cal_readability(preds)
    score_dict['fkgl'] = round(fkgl, 4)
    score_dict['dcrs'] = round(dcrs, 4)

    return score_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='eLife')
    parser.add_argument("--datatype", type=str, default="val")
    parser.add_argument("--method", type=str, default="null")
    args = parser.parse_args()
    
    ref_path = os.path.join('task1_development', 'val',  f'{args.dataset}_{args.datatype}.jsonl')
    pred_path = args.method

    score_dict = evaluation(pred_path, ref_path)
    print("rouge_score:")
    print([score_dict['rouge1'], score_dict['rouge2'], score_dict['rougel']])
    print("readability(fkgl, dcrs):")
    print([score_dict['fkgl'], score_dict['dcrs']])

    

        