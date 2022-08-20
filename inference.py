# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 18:26
# @Author  : Leesure
# @File    : inference.py
# @Software: PyCharm
import config
from utils import EOS_ids, BOS_ids, UNK_ids
import torch
from torch.utils.data import DataLoader
from utils import output_ids2words, DataSet, collection_function_for_test
import os
from model import LeeNQG
import torch.nn.functional as F
from utils import load_DRCD_dataset, load_SQuAD_dataset
from utils import vocab


class Generator(object):
    def __init__(self, checkpoint, output_folder, trail=0, print_question=False, example=None):
        """

        """
        self.vocab = vocab
        self.token2idx = self.vocab['token2idx']
        self.idx2token = self.vocab['idx2token']
        self.print_question = print_question
        if not print_question:
            self.dataset = config.dataset
            if self.dataset == 'squad':
                self.output_dir = output_folder + 'SQuAD/'
                test_example = load_SQuAD_dataset(config.pre_data_folder + 'SQuAD/', 'test')
            else:
                self.output_dir = output_folder + 'DRCD/'
                test_example = load_DRCD_dataset(config.pre_data_folder + 'DRCD/', 'test')
            test_data = DataSet(test_example, is_test=True)
            print('[*] the test data size: ', len(test_data))
            self.data_loader = DataLoader(test_data,
                                          batch_size=1,
                                          shuffle=False,
                                          collate_fn=collection_function_for_test)

            self.pred_dir = os.path.join(self.output_dir, "generated_{key}.txt".format(key=trail))
            self.golden_dir = os.path.join(self.output_dir, 'gold_{key}.txt'.format(key=trail))
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            data_set = DataSet(example, is_test=True)
            self.data_loader = DataLoader(data_set, batch_size=1, shuffle=False,
                                          collate_fn=collection_function_for_test)

        self.model = LeeNQG(config.vocab_size, config.emb_dim, config.hid_dim,
                            config.n_layer, config.cov_loss_hyper, config.generate_max_len,
                            config.device).to(config.device)
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        if not self.print_question:
            pred_fw = open(self.pred_dir, "w")
            golden_fw = open(self.golden_dir, "w")
            total_len = len(self.data_loader)
        for i, test_data in enumerate(self.data_loader):
            # src_seq, ext_src_seq, _, _, tag_seq, oov_lst = test_data
            src_seq, ext_src_seq, ans_seg, S_tokens, Q_tokens, A_str, s_pos, oov_lst = test_data
            best_question = self.beam_search(src_seq, ext_src_seq, ans_seg, s_pos)
            # discard START  token
            output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
            decoded_words = output_ids2words(
                output_indices, self.idx2token, oov_lst[0])
            try:
                fst_stop_idx = decoded_words.index(EOS_ids)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            # calculate bleu_4
            decoded_words = ' '.join(decoded_words)
            if not self.print_question:
                S = ' '.join(S_tokens[0])
                Q = ' '.join(Q_tokens[0])
                pred_fw.write(decoded_words+'\n')
                golden_fw.write(Q + '\n')

            # golden_question = self.untokenizer_data[i]['Q']
            # pred_fw.write(f"--------------{i} Item -----------------\n")
            # pred_fw.write(f'[Sen ]:\t{S}\n')
            # pred_fw.write(f'[Pred]:\t{decoded_words}\n')
            # pred_fw.write(f'[Gold]:\t{Q}\n')
            # pred_fw.write(f'[Ans]:\t{A_str[0]}\n')
            # pred_fw.write(f'[SCORE]: bleu_4={bleu_4:.4f}\n')
            # pred_fw.write(f'[Gold] : {golden_question}\n')
                print("\r", f" [-] Generate {i + 1}/{total_len} question", end='')
            else:
                print(decoded_words)
        # avg_BLEU_4 = total_bleu_4 / total_len
        # pred_fw.write(f'BLEU_4{avg_BLEU_4}')
        # print(f"\n[-] Done. The average BLEU score is {avg_BLEU_4}")

    def beam_search(self, src_seq, ext_src_seq, ans_seg, s_pos):
        enc_mask = torch.sign(src_seq)
        src_len = torch.sum(enc_mask, 1).to(config.device)
        prev_context = torch.zeros(1, 1, 2 * config.hid_dim)

        src_seq = src_seq.to(config.device)
        ext_src_seq = ext_src_seq.to(config.device)
        enc_mask = enc_mask.to(config.device)
        prev_context = prev_context.to(config.device)
        ans_seg = ans_seg.to(config.device)
        s_pos = s_pos.to(config.device)
        # forward encoder
        enc_outputs, enc_states = self.model.encoder(src_seq, src_len, ans_seg, s_pos)
        # [2, batch_size, d] but b = 1
        hypotheses = [Hypothesis(tokens=[BOS_ids],
                                 log_probs=[0.0],
                                 state=enc_states[:, 0, :],
                                 context=prev_context[0]) for _ in range(config.beam_size)]
        # tile enc_outputs, enc_mask for beam search
        ext_src_seq = ext_src_seq.repeat(config.beam_size, 1)
        enc_outputs = enc_outputs.repeat(config.beam_size, 1, 1)
        enc_features = self.model.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(config.beam_size, 1)
        num_steps = 0
        results = []
        coverage = torch.zeros(1, src_seq.size(1)).to(config.device)
        while num_steps < config.max_decode_step and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(self.token2idx) else UNK_ids for idx in latest_tokens]
            prev_y = torch.tensor(latest_tokens, dtype=torch.long, device=config.device).view(-1)

            # make batch of which size is beam size
            all_state_h = []
            all_context = []
            for h in hypotheses:
                state_h = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            # [beam_size, |V|]
            logits, states, context_vector, coverage = self.model.decoder.decode(prev_y, ext_src_seq, enc_features,
                                                                                 prev_context, prev_h, enc_mask,
                                                                                 coverage)
            # h_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids = torch.topk(log_probs, config.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                # state_i = (h_state[:, i, :], c_state[:, i, :])
                state_i = states[:, i, :]
                context_i = context_vector[i]
                for j in range(config.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == EOS_ids:
                    if num_steps >= config.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == config.beam_size or len(results) == config.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context

    def extend(self, token, log_prob, state, context=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context)
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)
