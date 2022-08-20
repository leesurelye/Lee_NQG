# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 19:46
# @Author  : Leesure
# @File    : utils.py
# @Software: PyCharm
from nltk.translate import bleu_score
# from rouge import Rouge
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as torch_data_utils
from torch.utils.data.dataset import T_co
import torch.nn.functional as F
import torch
import pickle
import config
import config_squad

BOS_tokens = '[BOS]'
EOS_tokens = '[EOS]'
PAD_tokens = '[PAD]'
UNK_tokens = '[UNK]'

PAD_ids = 0
BOS_ids = 1
EOS_ids = 2
UNK_ids = 3

special_tokens = {
    PAD_tokens: 0,
    BOS_tokens: 1,
    EOS_tokens: 2,
    UNK_tokens: 3
}

if config.dataset == 'squad':
    vocab = pickle.load(open(config_squad.preprocess_folder + 'token_vocab.pt', 'rb'))
else:
    vocab = pickle.load(open(config.vocab_file, 'rb'))

token2idx_vocab = vocab['token2idx']


class DataSet(torch_data_utils.Dataset):
    def __init__(self, data: list, is_test=False):
        super(DataSet, self).__init__()
        # self.data = load_dataset(data_folder, data_type, is_tokenizer=True)
        self.data = data
        self.len = len(self.data)
        self.vocab = token2idx_vocab
        self.is_test = is_test

    def __getitem__(self, index) -> T_co:
        # translate tokens into ids
        # since test didn't need the gold question
        S = self.data[index]['S']
        Q = self.data[index]['Q']
        A = self.data[index]['A']
        S_pos = self.data[index]['S_pos']
        ans_ids = torch.tensor(A, dtype=torch.int64)
        S_pos = torch.tensor(S_pos, dtype=torch.int64)
        s_ids, s_extended_ids, oov_list = self.sentence2idx(S)
        # train
        if not self.is_test:
            q_ids, q_extended_ids = self.question2idx(Q, oov_list)
            return s_ids, s_extended_ids, q_ids, q_extended_ids, ans_ids, S_pos, oov_list
        else:
            A_str = self.data[index]['ans']
            return s_ids, s_extended_ids, ans_ids, S, Q, A_str, S_pos, oov_list

    def sentence2idx(self, src_seq: list):
        ids = [BOS_ids]
        # tag_ids = [0]
        extended_ids = [BOS_ids]
        oov_list = []
        for token in src_seq:
            if token in self.vocab:
                ids.append(self.vocab[token])
                extended_ids.append(self.vocab[token])
            else:
                ids.append(UNK_ids)
                if token not in oov_list:
                    oov_list.append(token)
                extended_ids.append(len(self.vocab) + oov_list.index(token))
            # tag_ids.append(tag2idx[tag])
        # tag_ids.append(0)
        ids.append(EOS_ids)
        extended_ids.append(EOS_ids)
        ids = torch.tensor(ids, dtype=torch.int64)
        extended_ids = torch.tensor(extended_ids, dtype=torch.int64)
        # tag_ids = torch.tensor(tag_ids, dtype=torch.int64)
        return ids, extended_ids, oov_list

    def question2idx(self, tgt_seq: list, oov_list: list):
        ids = [BOS_ids]
        extended_ids = [BOS_ids]
        for token in tgt_seq:
            if token in self.vocab:
                ids.append(self.vocab[token])
                extended_ids.append(self.vocab[token])
            else:
                ids.append(UNK_ids)
                if token in oov_list:
                    extended_ids.append(len(self.vocab) + oov_list.index(token))
                else:
                    extended_ids.append(UNK_ids)
        ids.append(EOS_ids)
        extended_ids.append(EOS_ids)

        return torch.tensor(ids, dtype=torch.int64), torch.tensor(extended_ids, dtype=torch.int64)

    def __len__(self):
        return self.len


def load_DRCD_dataset(file_folder: str, data_type: str):
    file_path = file_folder + f'{data_type}_tokenized.pt'
    return pickle.load(open(file_path, 'rb'))


def load_SQuAD_dataset(file_folder: str, data_type: str):
    file_path = file_folder + f'{data_type}_tokenized.pt'
    return pickle.load(open(file_path, 'rb'))


def BLEU_N(reference, hypotheses, n=-1):
    if isinstance(reference, torch.Tensor):
        reference = reference.unsqueeze(1).tolist()
    if isinstance(hypotheses, torch.Tensor):
        hypotheses = hypotheses.tolist()
    if n == -1:
        weights = [0.25, 0.25, 0.25, 0.25]
    else:
        weights = [0, 0, 0, 0]
        weights[n - 1] = 1
    score = bleu_score.corpus_bleu(reference, hypotheses, weights,
                                   smoothing_function=bleu_score.SmoothingFunction().method7)
    return score


def pad_graph(graphs: list, max_len: int):
    # batch_size, seq_len, seq_len
    graphs = list(graphs)
    for i, graph in enumerate(graphs):
        off_set = max_len - graph.size(0)
        graphs[i] = F.pad(graph, (0, off_set, 0, off_set))
    return torch.stack(graphs)


def collection_function(batch: list):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    s_ids, s_extended_ids, q_ids, q_extended_ids, ans_ids, s_pos, oov_list = zip(*batch)

    padded_s_ids = pad_sequence(s_ids, batch_first=True)
    s_extended_ids = pad_sequence(s_extended_ids, batch_first=True)
    # max_len = s_extended_ids.size(1)
    padded_q_ids = pad_sequence(q_ids, batch_first=True)
    q_extended_ids = pad_sequence(q_extended_ids, batch_first=True)
    ans_ids = pad_sequence(ans_ids, batch_first=True)
    s_pos = pad_sequence(s_pos, batch_first=True)
    # graph = pad_graph(graph, max_len)
    return padded_s_ids, s_extended_ids, padded_q_ids, q_extended_ids, ans_ids, s_pos, oov_list


def collection_function_for_test(batch: list):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    s_ids, s_extended_ids, ans_ids, S_tokens, Q_tokens, A_str, s_pos, oov_list = zip(*batch)

    padded_s_ids = pad_sequence(s_ids, batch_first=True)
    s_extended_ids = pad_sequence(s_extended_ids, batch_first=True)
    # max_len = s_extended_ids.size(1)
    a_padded_ids = pad_sequence(ans_ids, batch_first=True)
    s_pos = pad_sequence(s_pos, batch_first=True)
    # graph = pad_graph(graph, max_len)
    return padded_s_ids, s_extended_ids, a_padded_ids, S_tokens, Q_tokens, A_str, s_pos, oov_list


def output_ids2words(id_list, idx2word, oov_list=None):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param oov_list: list of oov words
    :return: list of words
    """
    words = []
    for idx in id_list:
        word = None
        try:
            word = idx2word[idx]
        except KeyError:
            if oov_list is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = oov_list[article_oov_idx]
                except IndexError:
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ids]
        words.append(word)

    return words
