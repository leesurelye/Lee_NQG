# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 14:34
# @Author  : Leesure
# @File    : preprocess_squad.py
# Info :This is file about preprocess AQuAD dataset
# @Software: PyCharm
import torch
import config_squad as config
import json
import pickle
from torch.nn.init import uniform_
from utils import UNK_ids, PAD_ids, EOS_ids, BOS_ids, UNK_tokens, PAD_tokens, EOS_tokens, BOS_tokens
import spacy

# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models import KeyedVectors

tokenizer = spacy.load('en_core_web_sm')


def transf_glove():
    glove_input_file = config.glove_file
    word2vec_file = config.word2vec_file
    word_dict = dict()
    # 2196018, 300
    f = open(glove_input_file, 'r')
    i = 0
    lines = f.readlines()
    for line in lines:
        doc = line.split()
        vector = doc[-300:]
        vector = [float(x) for x in vector]
        token = ''.join(doc[:-300])
        word_dict[token] = torch.FloatTensor(vector)
        i += 1
        print('\r', f'{i} process', end='')
    with open(word2vec_file, 'wb') as f:
        pickle.dump(word_dict, f)
    f.close()


def process_file(file_folder):
    vocab_dit = dict()
    pos_dict = dict()
    dep_dict = dict()

    def tokenize_sentence(sentence: str, data_type: str):
        doc = tokenizer(sentence)
        n = len(doc)
        token_list = [0] * n
        pos_list = [0] * n
        dep_list = [0] * n
        for index, token in enumerate(doc):
            # tokens
            token_list[index] = token.text
            pos_list[index] = token.pos_
            dep_list[index] = token.dep_
            if data_type == 'dev':
                continue
            if token.text in vocab_dit.keys():
                vocab_dit[token.text] += 1
            else:
                vocab_dit[token.text] = 1
            # pos tagging
            if token.pos_ in pos_dict.keys():
                pos_dict[token.pos_] += 1
            else:
                pos_dict[token.pos_] = 1
            # dependence parsing
            if token.dep_ in dep_dict.keys():
                dep_dict[token.dep_] += 1
            else:
                dep_dict[token.dep_] = 1
        return token_list, pos_list, dep_list

    def tokenize_question(question: str, data_type: str):
        doc = tokenizer(question)
        n = len(doc)
        token_list = [0] * n
        for index, token in enumerate(doc):
            # tokens
            token_list[index] = token.text
            if data_type == 'dev':
                continue
            if token.text in vocab_dit.keys():
                vocab_dit[token.text] += 1
            else:
                vocab_dit[token.text] = 1
        return token_list

    def tag_ans_position(S_tokens: list, S_dep: list, answer_start: int, answer_end: int):
        n = len(S_tokens)
        tag_ans = [0] * n
        count = 0
        for i, token in enumerate(S_tokens):
            if answer_start <= count <= answer_end:
                tag_ans[i] = S_dep[i]
            else:
                tag_ans[i] = '[PAD]'
            count += len(token)
        return tag_ans

    def process_article(articles: list, data_type: str):
        print(f'[*] Process {data_type} data...')
        examples = list()
        for i, article in enumerate(articles):
            for para in article["paragraphs"]:
                S = para["context"]
                S_doc = tokenizer(S)
                S_tokens, S_pos, S_dep = tokenize_sentence(S_doc, data_type)
                for qa in para["qas"]:
                    if qa['is_impossible']:
                        continue
                    Q = qa["question"]
                    # change dependency parsing
                    Q_tokens = tokenize_question(Q, data_type)
                    A = qa["answers"][0]['text']
                    answer_start = qa["answers"][0]["answer_start"]
                    answer_end = answer_start + len(A)
                    A_dep = tag_ans_position(S_tokens, S_dep, answer_start, answer_end)
                    example = {"S": S_tokens, "Q": Q_tokens, "ans": A, "A": A_dep, "S_pos": S_pos}
                    examples.append(example)
            print('\r', f'Process {i}/{total} item', end='')
        file_name = config.preprocess_folder + f'{data_type}_tokenized.pt'
        print(f'\n[*] {data_type} size ', len(examples))
        with open(file_name, 'wb') as ex_f:
            pickle.dump(examples, ex_f)
        ex_f.close()
        print('[-] Done')

    def build_vocab(counter: dict, vocab_type: str):
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
        sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for idx, (token, freq) in enumerate(sorted_by_freq, start=4):
            word2idx[token] = idx
            idx2word[idx] = token
            if idx == config.vocab_size - 1:
                break
            print('\r', f'{idx}/{config.vocab_size} item', end='')
        save_path = config.preprocess_folder + f'{vocab_type}_vocab.pt'
        print(f'[*] the {vocab_type} size {len(word2idx)}')
        with open(save_path, 'wb') as vocab_f:
            pickle.dump({'token2idx': word2idx, 'idx2token': idx2word}, vocab_f)
        vocab_f.close()
        print(f'\n [-] Done! vocab file saved at folder: {save_path}')

    file_path = file_folder + '/' + 'train-v2.0.json'
    with open(file_path, "r") as f:
        source = json.load(f)
        train = source["data"]
    f.close()
    total = len(train)
    process_article(train, 'train')
    dev_path = file_folder + '/' + 'dev-v2.0.json'
    with open(dev_path, 'r') as f:
        source = json.load(f)
        dev = source["data"]
    f.close()
    process_article(dev, 'dev')

    print('[*] Build vocabulary')
    build_vocab(vocab_dit, 'token')
    build_vocab(pos_dict, 'pos')
    build_vocab(dep_dict, 'dep')


def build_embedded_from_vocab(vocab=None):
    if vocab is None:
        vocab_path = config.preprocess_folder + 'token_vocab.pt'
        vocab = pickle.load(open(vocab_path, 'rb'))
    word2idx = vocab['token2idx']
    word2vector = pickle.load(open(config.word2vec_file, 'rb'))
    embedded = torch.zeros(config.vocab_size, 300, requires_grad=False)
    UNK_vector = uniform_(torch.empty(1, 300, dtype=torch.float32))
    print("\n[*] Building embedding vector...")
    num_oov = 0
    for token, idx in word2idx.items():
        try:
            embedded[idx] = torch.FloatTensor(word2vector[token])
        except KeyError:
            embedded[idx] = UNK_vector
            num_oov += 1
        print('\r', f"{idx}/{config.vocab_size}", end="")
    print(f"\n [-] Done! oov_unm={num_oov}")
    with open(config.embed_vector_file, 'wb') as f:
        pickle.dump(embedded, f)
    f.close()
    return embedded


def tokens_to_ids():
    file_path = config.preprocess_folder + 'train_tokenized.pt'
    pos_vocab = pickle.load(open(config.preprocess_folder + 'pos_vocab.pt', 'rb'))
    pos_vocab = pos_vocab['token2idx']
    dep_vocab = pickle.load(open(config.preprocess_folder + 'dep_vocab.pt', 'rb'))
    dep_vocab = dep_vocab['token2idx']
    examples = pickle.load(open(file_path, 'rb'))
    total = len(examples)
    for i, example in enumerate(examples):
        A_dep = example['A']
        S_pos = example['S_pos']
        n = len(A_dep)
        A_dep_ids = [0] * (n + 2)
        S_pos_ids = [0] * (n + 2)
        A_dep_ids[0] = BOS_ids
        S_pos_ids[0] = BOS_ids
        for j, a_ in enumerate(A_dep):
            A_dep_ids[j] = dep_vocab[a_]
        for j, s_ in enumerate(S_pos):
            S_pos_ids[j] = pos_vocab[s_]
        A_dep_ids[-1] = EOS_ids
        S_pos_ids[-1] = EOS_ids
        example['A'] = A_dep_ids
        example['S_pos'] = S_pos_ids
        examples[i] = example
        print('\r', f'{i + 1}/{total} process', end='')
    with open(file_path, 'wb') as f:
        pickle.dump(examples, f)
    f.close()
    print('[*] Done!')


if __name__ == '__main__':
    # process_file('/home1/liyue/Lee_NQG/dataset/SQuAD')
    # tokens_to_ids()
    res = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/SQuAD/test_tokenized.pt', 'rb'))
    print(res[0])