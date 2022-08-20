# -*- coding: utf-8 -*-
# @Time    : 2021/12/26 17:10
# @Author  : Leesure
# @File    : evaluate.py
# @Software: PyCharm
from pprint import pprint
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import config


def load_text(filename):
    print("[*] Loading data")
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split("\n")
    pre_offset = 2
    gold_offset = 3
    gold, pred = [], []
    for i in range(0, len(data), 5):
        gold.append(data[i + gold_offset].split('\t')[1])
        pred.append(data[i + pre_offset].split('\t')[1])
    print("[-] Done")
    return gold, pred


def evaluate():
    predict_path = config.output_file_path + "generated.txt"
    ground_truth, predictions = load_text(predict_path)

    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Rouge": Rouge()
    }

    gts = {}
    res = {}
    if len(predictions) == len(ground_truth):
        for ind, value in enumerate(predictions):
            res[ind] = [value]

        for ind, value in enumerate(ground_truth):
            gts[ind] = [value]
    else:
        Min_Len = min(len(predictions), len(ground_truth))
        for ind in range(Min_Len):
            res[ind] = [predictions[ind]]
            gts[ind] = [ground_truth[ind]]

    print(f'samples:{len(res.keys())} / {len(gts.keys())}')

    scores = {}
    for name, scorer in scorers.items():
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    pprint(scores)


if __name__ == "__main__":
    evaluate()
