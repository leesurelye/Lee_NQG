# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 15:26
# @Author  : Leesure
# @File    : preprocess.py
# @Software: PyCharm
from collections import Counter
import pickle
import os
import config
import json
import torch
from torch.nn.init import uniform
from utils import load_dataset
from utils import UNK_ids, PAD_ids, EOS_ids, BOS_ids, UNK_tokens, EOS_tokens, BOS_tokens, PAD_tokens


def preprocess_dataset(file_path: str, save_folder: str, data_type: str):
    """抽取数据集合中的 S, Q, A"""
    values = []

    def filter_sentence(text: str, qa_pair: list):
        text = text.replace('"', "").replace('\n', "")
        for qa in qa_pair:
            ans_len = len(qa['A'])
            index = text.find(qa['A'])
            if index >= 0:
                start = index
                end = index + ans_len
                while start >= 0 and text[start] != '。':
                    start -= 1
                while end < len(text) and text[end] != '。':
                    end += 1
            else:
                start = 0
                end = len(text)
            values.append({'S': text[start + 1: end], 'Q': qa['Q'], 'A': qa['A'],
                           'S_len': end - start + 2, 'Q_len': len(qa['Q'])})

    file = open(file_path, encoding='utf-8')
    data_raw = json.load(file)
    save_folder = save_folder + '/' + data_type + '.pt'
    print(f"[*] Processing the {data_type} data...")
    for i, item in enumerate(data_raw):
        filter_sentence(item['text'], item['annotations'])
        print("\r", f'{i + 1}/{len(data_raw)}', end="")
    with open(save_folder, 'wb') as f:
        pickle.dump(values, f)
    f.close()

    print(f"\n[-] (Done!) Data saved at folder: {save_folder}")


def tokenize_dataset():
    train_file_path = config.pre_data_folder + 'train.pt'
    test_file_path = config.pre_data_folder + 'test.pt'
    save_train_path = config.pre_data_folder + 'train_tokenized.pt'
    save_test_path = config.pre_data_folder + 'test_tokenized.pt'
    tokenizer = config.tokenizer
    train_tokenized = list()
    test_tokenized = list()
    print('[*] Tokenizing...')
    train_dataset = pickle.load(open(train_file_path, 'rb'))
    total_len = len(train_dataset)
    print("[*] Process Train Dataset")
    for i, data in enumerate(train_dataset):
        S = data['S']
        Q = data['Q']
        tokens = tokenizer([S, Q], tasks='tok')['tok/fine']
        train_tokenized.append({'S': tokens[0], 'Q': tokens[1]})
        print('\r', f'{i}/{total_len} Item', end='')
    with open(save_train_path, 'wb') as f:
        pickle.dump(train_tokenized, f)
    f.close()
    print(' [-] Done!')
    test_dataset = pickle.load(open(test_file_path, 'rb'))
    total_len = len(test_dataset)
    print("[*] Process Test Dataset")
    for i, data in enumerate(test_dataset):
        S = data['S']
        Q = data['Q']
        tokens = tokenizer([S, Q], tasks='tok')['tok/fine']
        test_tokenized.append({'S': tokens[0], 'Q': tokens[1]})
        print('\r', f'{i}/{total_len} Item', end='')
    with open(save_test_path, 'wb') as f:
        pickle.dump(test_tokenized, f)
    f.close()
    print(' [-] Done!')


def build_embed_vector(word2idx: dict = None, cached=True):
    file_path = config.embed_vector_file
    if cached and os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    if word2idx is None:
        tokenize_data = config.vocab_file
        word2idx = pickle.load(open(tokenize_data, 'rb'))
    word2vector = config.w2v_model
    vocab_size = len(word2idx)
    vector = torch.zeros(vocab_size, 300, requires_grad=False)
    # init special Token
    UNK_vector = uniform(torch.empty(1, 300, dtype=torch.float32))
    print("\n[*] Building embedding vector...")
    num_oov = 0
    for token, idx in word2idx.items():
        if token in word2vector.vocab:
            vector[idx] = torch.FloatTensor(word2vector.get_vector(token))
        else:
            vector[idx] = UNK_vector
            num_oov += 1
        print('\r', f"{idx}/{vocab_size}", end="")
    print(f"\n [-] Done! oov_unm={num_oov}")
    if cached:
        with open(file_path, 'wb') as f:
            pickle.dump(vector, f)
        f.close()
    return vector


def build_vocab(cached=True):
    dataset = load_dataset(config.pre_data_folder, 'all', is_tokenizer=True)
    tokens = []
    word2idx = dict()
    word2idx[UNK_tokens] = UNK_ids
    word2idx[PAD_tokens] = PAD_ids
    word2idx[EOS_tokens] = EOS_ids
    word2idx[BOS_tokens] = BOS_ids
    file_path = config.vocab_file
    if cached and os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    # build the vocabulary
    print("[*] Building vocabulary ...")
    for i, item in enumerate(dataset):
        tokens.extend(item['S'])
        tokens.extend(item['Q'])
        print('\r', f'Build Vocab Process : {i}/{len(dataset)}', end="")
    print("\n[-] (Done!) ")
    counter = Counter(tokens)
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # ordered_dict = OrderedDict(sorted_by_freq)
    for idx, (token, freq) in enumerate(sorted_by_freq, start=4):
        word2idx[token] = idx

    if cached:
        with open(file_path, 'wb') as f:
            pickle.dump(word2idx, f)
        f.close()

    return word2idx


if __name__ == '__main__':
    # tokenize_dataset()
    # word_to_idx = build_vocab()
    build_embed_vector()
