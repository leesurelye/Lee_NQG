# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 19:51
# @Author  : Leesure
# @File    : preprocess_drcd.py
# @Software: PyCharm
import torch

import config
import json
import pickle
from collections import Counter
from torch.nn.init import uniform_
from utils import UNK_ids, PAD_ids, EOS_ids, BOS_ids, UNK_tokens, PAD_tokens, EOS_tokens, BOS_tokens
import hanlp
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


tokenizer = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
w2v_model = KeyedVectors.load_word2vec_format(config.embedded_file, binary=False, unicode_errors='ignore')


def tag_answer(sentence: list, ans_start, ans_end):
    length = len(sentence)
    tag = [0] * length
    counter = 0
    for i, token in enumerate(sentence):
        counter += len(token)
        if counter == ans_start - 1:
            tag[i] = 'B'
        elif ans_start < counter <= ans_end:
            tag[i] = 'I'
        else:
            tag[i] = 'O'

    return tag


def mask_answer(sentence: list, ans_start, ans_end):
    length = len(sentence)
    tag = [0] * (length + 2)
    tag[0] = 0
    counter = 0
    for i, token in enumerate(sentence):
        counter += len(token)
        if ans_start <= counter <= ans_end:
            tag[i + 1] = 1
        else:
            tag[i + 1] = 0
    tag[-1] = 0
    return tag


def build_adjacency_matrix(nodes: list, sentence: list, ans_start, ans_end, edge_vocab: dict):
    # node is dependency parsing by HaNLP
    n = len(nodes)
    graph_matrix = [[0] * (n + 2) for _ in range(n + 2)]
    ans_matrix = []
    count = 0
    for i, node in enumerate(nodes, start=1):
        index, edge = node[0], node[1]
        if index < n:
            edge_ids = edge_vocab[edge]
            graph_matrix[index + 1][i] = edge_ids
            graph_matrix[i][index + 1] = edge_ids

    for i, word in enumerate(sentence, start=1):
        count += len(word)
        if ans_start - 1 <= count <= ans_end + 1:
            ans_matrix.append(graph_matrix[i])
    ans_matrix = np.asarray(ans_matrix)
    if ans_matrix.size == 0:
        return np.asarray([0] * (n + 2))
    return np.max(ans_matrix, 0)


def pos_token_to_ids(pos_list: list, vocab: dict):
    n = len(pos_list)
    ans = [0] * (n + 2)
    for i, pos in enumerate(pos_list):
        if pos in vocab.keys():
            ans[i] = vocab[pos]
        else:
            ans[i] = 0
    ans[-1] = 0
    return ans


def process_file(file_name, data_type):
    examples = list()
    edge_vocab = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/edge_vocab.pt', 'rb'))
    pos_vocab = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/pos_vocab.pt', 'rb'))
    with open(file_name, "r") as f:
        source = json.load(f)
        articles = source["data"]
    f.close()
    total = len(articles)
    print("[*] Process Data...")
    for i, article in enumerate(articles):
        for para in article["paragraphs"]:
            S = para["context"]
            S_hanlp = tokenizer(S, tasks=['dep', 'pos'])
            S_tokens = S_hanlp['tok/fine']
            S_dep = S_hanlp['dep']
            S_pos = S_hanlp['pos/ctb']
            S_pos_ids = pos_token_to_ids(S_pos, pos_vocab)
            for qa in para["qas"]:
                Q = qa["question"]
                # change dependency parsing
                Q_tokens = tokenizer(Q, tasks='tok')['tok/fine']
                A = qa["answers"][0]['text']
                answer_start = qa["answers"][0]["answer_start"]
                answer_end = answer_start + len(A)
                mask_ans = build_adjacency_matrix(S_dep, S_tokens, answer_start, answer_end, edge_vocab)
                example = {"S": S_tokens, "Q": Q_tokens, "ans": A, "A": mask_ans, "S_pos": S_pos_ids}
                examples.append(example)

        print('\r', f'Process {i}/{total} item', end='')
    file_name = config.pre_data_folder + f'{data_type}_tokenized.pt'
    with open(file_name, 'wb') as f:
        pickle.dump(examples, f)
    f.close()
    print(f"\n [-] Done! save processed data in folder: {file_name}")
    return examples


def build_pos_vocab(file_name, vocab_file=None):
    if vocab_file is None:
        vocab_file = dict()
    else:
        vocab_file = pickle.load(open(vocab_file, 'rb'))
    with open(file_name, "r") as f:
        source = json.load(f)
        articles = source["data"]
    f.close()
    total = len(articles)
    print("[*] Process Data...")
    for i, article in enumerate(articles):
        for para in article["paragraphs"]:
            S = para["context"]
            S_hanlp = tokenizer(S, tasks='pos')
            S_pos = S_hanlp['pos/ctb']
            for p in S_pos:
                if p in vocab_file.keys():
                    vocab_file[p] += 1
                else:
                    vocab_file[p] = 1
            for qa in para["qas"]:
                Q = qa["question"]
                # change dependency parsing
                Q_pos = tokenizer(Q, tasks='pos')['pos/ctb']
                for q_pos in Q_pos:
                    if q_pos in vocab_file.keys():
                        vocab_file[q_pos] += 1
                    else:
                        vocab_file[q_pos] = 1
        print('\r', f'Process {i}/{total} item', end='')
    file_name = config.pre_data_folder + f'pos_vocab.pt'
    with open(file_name, 'wb') as f:
        pickle.dump(vocab_file, f)
    f.close()
    print(f"\n [-] Done! pos vocab size: {len(vocab_file)}")


def build_vocab(examples=None):
    tokens = []
    word2idx = dict()
    idx2word = dict()
    word2idx[UNK_tokens] = UNK_ids
    word2idx[PAD_tokens] = PAD_ids
    word2idx[EOS_tokens] = EOS_ids
    word2idx[BOS_tokens] = BOS_ids

    idx2word[UNK_ids] = UNK_tokens
    idx2word[PAD_ids] = PAD_tokens
    idx2word[EOS_ids] = EOS_tokens
    idx2word[BOS_ids] = BOS_tokens

    file_path = config.vocab_file

    if examples is None:
        # only vocab train vocab
        file_name = config.pre_data_folder + 'train_tokenized.pt'
        examples = pickle.load(open(file_name, 'rb'))
    total = len(examples)
    print("[*] Load Data...")
    for i, example in enumerate(examples):
        tokens.extend(example['S'])
        tokens.extend(example['Q'])
        print('\r', f'Process {i}/{total} item', end='')
    print("\n [-] Done!")
    counter = Counter(tokens)
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print("[*] Build vocab...")
    for idx, (token, freq) in enumerate(sorted_by_freq, start=4):
        word2idx[token] = idx
        idx2word[idx] = token
        if idx == config.vocab_size - 1:
            break
        print('\r', f'{idx}/{config.vocab_size} item', end='')
    with open(file_path, 'wb') as f:
        pickle.dump({'token2idx': word2idx, 'idx2token': idx2word}, f)
    f.close()
    print(f'\n [-] Done! vocab file saved at folder: {file_path}')
    return {'token2idx': word2idx, 'idx2token': idx2word}


def build_embedded_from_vocab(vocab=None):
    if vocab is None:
        vocab = pickle.load(open(config.vocab_file, 'rb'))
    word2idx = vocab['token2idx']

    embedded = torch.zeros(config.vocab_size, 300, requires_grad=False)
    UNK_vector = uniform_(torch.empty(1, 300, dtype=torch.float32))
    print("\n[*] Building embedding vector...")
    num_oov = 0
    for token, idx in word2idx.items():
        try:
            embedded[idx] = torch.FloatTensor(w2v_model.get_vector(token))
        except KeyError:
            embedded[idx] = UNK_vector
            num_oov += 1
        print('\r', f"{idx}/{config.vocab_size}", end="")
    print(f"\n [-] Done! oov_unm={num_oov}")
    with open(config.embed_vector_file, 'wb') as f:
        pickle.dump(embedded, f)
    f.close()
    return embedded


def combine_dataset():
    train_data_file = config.pre_data_folder + 'train_tokenized.pt'
    dev_data_file = config.pre_data_folder + 'dev_tokenized.pt'
    train_data = pickle.load(open(train_data_file, 'rb'))
    valid_data = pickle.load(open(dev_data_file, 'rb'))
    i = 1
    for d in valid_data:
        train_data.append(d)
        i += 1
        print(f'{i} processed')
    with open(train_data_file, 'wb') as f:
        pickle.dump(train_data, f)
    f.close()


def build_edge_vocab():
    file_path = '/home1/liyue/Lee_NQG/dataset/preprocess/edge_vocab.pt'
    counter = pickle.load(open(file_path, 'rb'))
    edge2idx = dict()
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print("[*] Build vocab...")
    for idx, (token, freq) in enumerate(sorted_by_freq):
        edge2idx[token] = idx
        print('\r', f'{idx} item', end='')
    with open(file_path, 'wb') as f:
        pickle.dump(edge2idx, f)
    f.close()


if __name__ == '__main__':
    # process_file('/home1/liyue/Lee_NQG/dataset/DRCD/train.json', 'train')
    # vocabulary = build_vocab()
    # build_embedded_from_vocab(vocabulary)
    # process_file('/home1/liyue/Lee_NQG/dataset/DRCD/dev.json', 'dev')
    # process_file('/home1/liyue/Lee_NQG/dataset/DRCD/test.json', 'test')
    # combine_dataset()
    # vocab = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/pos_vocab.pt', 'rb'))
    vocab = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/DRCD/pos_vocab.pt', 'rb'))
    print(len(vocab))
