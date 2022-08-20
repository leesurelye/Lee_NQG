# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 19:38
# @Author  : Lee sure
# @File    : model.py
# @Software: PyCharm
# import random
import torch
import torch.nn as nn
import pickle
import config
from utils import PAD_ids
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max
import torch.nn.functional as F
from utils import UNK_ids
# from torch.nn.init import orthogonal_

INF = 1e12


class LeeNQG(nn.Module):
    """
    The LeeNQG for Chinese Question Generation
    """

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hid_dim,
                 n_layer,
                 coverage_loss_hyper,
                 tgt_max_len,
                 device,
                 init_emb_vector=None):
        super(LeeNQG, self).__init__()
        self.n_layer = n_layer
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab = vocab_size
        self.device = device
        self.tgt_max_len = tgt_max_len
        if init_emb_vector is None:
            init_emb_vector = pickle.load(open(config.embed_vector_file, 'rb'))
        self.encoder = Encoder(init_emb_vector, vocab_size, emb_dim, hid_dim,
                               n_layer,
                               config.dropout,
                               self.device)
        self.decoder = Decoder(init_emb_vector, vocab_size, emb_dim, hid_dim * 2, n_layer,
                               coverage_loss_hyper,
                               config.dropout,
                               self.device)

        # self.bert = AutoModel.from_pretrained("")
        #
        # #
        # self.liner = nn.Linear()

    def forward(self,
                input_s, ext_src_ids,
                input_q, input_a, s_pos,
                forcing_ratio=0):
        encoder_mask = torch.sign(input_s)
        src_len = torch.sum(encoder_mask, dim=1)
        # encoder_output, ans_output, hidden = self.encoder(input_s, src_len, input_a, ans_len)
        encoder_output, hidden = self.encoder(input_s, src_len, input_a, s_pos)
        return self.decoder(input_q, ext_src_ids, encoder_output, hidden, encoder_mask, forcing_ratio)


class Encoder(nn.Module):
    def __init__(self, embedded_file, vocab, emb_dim, hid_dim, n_layers, dropout,
                 device):
        super(Encoder, self).__init__()
        self.vocab = vocab
        self.hid_dim = hid_dim
        self.device = device
        self.emb_him = emb_dim
        self.n_layers = n_layers
        if embedded_file is not None:
            self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ids).from_pretrained(embedded_file,
                                                                                               freeze=True)
        else:
            self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ids)
        self.ans_embedding = nn.Embedding(config.edge_size, config.edge_emb_size)
        # self.pos_embedding = nn.Embedding(config.pos_size, config.pos_emb_dim)
        self.rnn = nn.GRU(emb_dim + config.edge_emb_size,
                          hid_dim, num_layers=n_layers, bidirectional=True,
                          dropout=dropout,
                          batch_first=True)
        # self.graph_embedding = nn.Linear(emb_dim, hid_dim * 2)
        self.linear_trans = nn.Linear(2 * hid_dim, 2 * hid_dim)
        self.update_layer = nn.Linear(4 * hid_dim, 2 * hid_dim, bias=False)
        self.gate = nn.Linear(4 * hid_dim, 2 * hid_dim, bias=False)

    def gated_self_attn(self, queries, memories, mask):
        # queries: [b,t,d]
        # memories: [b,t,d]
        # mask: [b,t]
        # b, t,
        # mask_ans = input_a.unsqueeze(-1)
        # ans_memories = memories.masked_fill(mask_ans == 0, value=1)
        # memories = torch.mul(memories, ans_memories)

        mask = mask.unsqueeze(1)
        energies = torch.matmul(queries, memories.transpose(1, 2))  # [b, t, t]
        energies = energies.masked_fill(mask == 0, value=-1e12)
        scores = F.softmax(energies, dim=-1)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    @staticmethod
    def ans_sen_attn(sentence, tag, mask):
        ans = sentence.masked_fill(tag.unsqueeze(-1) == 0, value=1)
        energy = torch.mul(sentence, ans)
        # energy = F.bilinear(ans, sentence, )
        mask = mask.unsqueeze(-1)
        energy = torch.masked_fill(energy, mask == 0, value=-1e12)
        energy = torch.softmax(energy, dim=1)
        return torch.mul(energy, sentence)

    def forward(self, input, src_len, input_a, s_pos):
        """
        Encoder of sentence
        graph: batch_size, src_len, src_len
        """
        mask_seq = torch.sign(input)
        # b, src_len, embedding
        embedded = self.embedding(input)
        ans_embedded = self.ans_embedding(input_a)
        # pos_embedded = self.pos_embedding(s_pos)
        embedded = torch.cat([embedded, ans_embedded], dim=-1)
        batch_size, seq_len = input.shape
        packed = pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # batch_size, seq_len, 2 * hid_dim
        memories = self.linear_trans(output)
        self_output = self.gated_self_attn(output, memories, mask_seq)
        ans_output = self.ans_sen_attn(memories, input_a, mask_seq)
        self_output = torch.add(self_output, ans_output)
        hidden = hidden.view(self.n_layers, -1, batch_size, self.hid_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        return self_output, hidden


class Decoder(nn.Module):
    def __init__(self, embedded_file, vocab, emb_dim, hid_dim, n_layers,
                 coverage_loss_hyper, dropout,
                 device):
        super(Decoder, self).__init__()
        self.device = device
        self.vocab = vocab
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.coverage_loss_hyper = coverage_loss_hyper
        if embedded_file is not None:
            self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ids). \
                from_pretrained(embedded_file, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ids)
        self.encoder_trans = nn.Linear(hid_dim, hid_dim)
        self.reduce_layer = nn.Linear(
            emb_dim + hid_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=n_layers, dropout=dropout,
                          bidirectional=False, batch_first=True)
        self.P_vocab = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, vocab)
        )

    @staticmethod
    def attention(memories, query, mask, coverage):
        """
            memories [b, t, d]
            query   [b, 1, d]
            coverage [b, t]
        """
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        sen_energy = energy.squeeze(1).masked_fill(mask == 0, value=-1e12)  # [b, t]

        # ans_energy = torch.matmul(query, ans_memories.transpose(1, 2))
        # ans_energy = ans_energy.squeeze(1).masked_fill(mask == 0, value=-1e12)
        # coverage mechanism
        # energy = torch.add(sen_energy, ans_energy)
        cover_energy = torch.mul(sen_energy, 1 - coverage)
        # cover_energy = torch.mul(energy, coverage)
        attn_dist = F.softmax(cover_energy, dim=1)  # [b, t]
        context = torch.matmul(attn_dist.unsqueeze(dim=1), memories)  # [b, 1, d]

        return attn_dist, cover_energy, context

    # for generate sentence
    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, input_q, ext_src_ids, encoder_output, hidden, enc_mask, forcing_ratio=0):
        input_q = input_q[:, :-1]
        batch_size, src_seq_len = ext_src_ids.shape
        tgt_seq_len = input_q.size(1)
        coverage_vector = torch.zeros(size=(batch_size, src_seq_len), device=self.device)
        # pre_attention = torch.zeros(size=(batch_size, tgt_seq_len, src_seq_len), device=self.device)
        memories = self.encoder_trans(encoder_output)
        context = torch.zeros(batch_size, 1, self.hid_dim, device=self.device)
        logits_output = []
        cov_losses = []
        # pre_time_ids = input_q[:, 0].unsqueeze(1)
        for time_step in range(tgt_seq_len):
            # if random.random() < forcing_ratio:
            #     input_ids = input_q[:, time_step].unsqueeze(1)
            # else:
            #     input_ids = pre_time_ids
            input_ids = input_q[:, time_step].unsqueeze(1)
            embedded = self.embedding(input_ids)
            rnn_input = torch.cat([embedded, context], dim=-1)
            decoder_output, hidden = self.rnn(rnn_input, hidden)
            # coverage mechanism
            attention_dist, energy, context = self.attention(memories, decoder_output, enc_mask,
                                                             coverage_vector)
            # pre_attention[:, time_step] = attention_dist
            cov_loss = torch.sum(torch.min(attention_dist, coverage_vector)) / batch_size * config.cov_loss_hyper
            # update coverage vector
            # TODO change local coverage mechanism
            coverage_vector = coverage_vector + attention_dist
            # if time_step > config.coverage_step:
            #     coverage_vector = coverage_vector - pre_attention[:, time_step]
            # coverage_vector = attention_dist
            logits = self.P_vocab(torch.cat([decoder_output, context], dim=-1)).squeeze(1)
            # _, pre_time_ids = torch.topk(logits, k=1, dim=-1)

            num_oov = max(torch.max(ext_src_ids - self.vocab + 1), 0)
            extend_zeros = torch.zeros(batch_size, num_oov, device=self.device)
            extended_logits = torch.cat([logits, extend_zeros], dim=-1)
            out = torch.zeros_like(extended_logits) - 1e12
            out, _ = scatter_max(energy, ext_src_ids, out=out)
            out = torch.masked_fill(out, out == -1e12, 0)
            logits = extended_logits + out
            logits = torch.masked_fill(logits, logits == 0, -1e12)
            logits_output.append(logits)
            cov_losses.append(cov_loss)
        # tmp = [x.item() for x in cov_losses]
        # print('[coverage loss arr]', tmp)
        return torch.stack(logits_output, dim=1), torch.sum(torch.tensor(cov_losses)) / tgt_seq_len

    # for generate sentence
    def decode(self, input_q, ext_src_ids, encoder_output, context, state, enc_mask, coverage):
        # only forward one step lstm
        # input_q: [batch, ]
        # TODO coverage
        batch_size = input_q.size(0)
        embedded = self.embedding(input_q.unsqueeze(1))  # b, 1, emb_size
        rnn_input = torch.cat([embedded, context], dim=2)
        decoder_output, state = self.rnn(rnn_input, state)
        attention_dist, energy, context = self.attention(encoder_output, decoder_output, enc_mask, coverage)
        logits = self.P_vocab(torch.cat([decoder_output, context], dim=-1)).squeeze(1)  # [b, vocab_size]
        coverage = coverage + attention_dist
        num_oov = max(torch.max(ext_src_ids - self.vocab + 1), 0)
        zeros = torch.zeros(batch_size, num_oov, device=config.device)
        extended_logits = torch.cat([logits, zeros], dim=-1)
        out = torch.zeros_like(extended_logits) - INF
        out, _ = scatter_max(energy, ext_src_ids, out=out)
        out = torch.masked_fill(out, out == -INF, 0)
        logits = extended_logits + out
        logits = torch.masked_fill(logits, logits == -INF, 0)
        # force UNK prob to -inf
        logits[:, UNK_ids] = -INF

        return logits, state, context, coverage
