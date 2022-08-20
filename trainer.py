# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 18:53
# @Author  : Leesure
# @File    : trainer.py
# @Software: PyCharm
import math

import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
import torch
from torch.utils.data import DataLoader
from utils import DataSet
import config
from utils import collection_function, PAD_ids


# import time


class Trainer(object):
    def __init__(self, QG_model: nn.Module, example,
                 device):
        super(Trainer, self).__init__()
        self.device = device
        self.QG_model = QG_model
        self.best_bleu = 0
        self.best_loss = float('inf')
        self.debug_writer = config.writer
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ids)
        self.lr = config.lr
        self.train_step = 0
        self.example = example
        self.example_len = len(example)

    @staticmethod
    def forcing_learning(batch):
        if config.forcing_decay_type == 'linear':
            forcing_ratio = max(0, config.forcing_ratio - config.forcing_decay * batch)
        elif config.forcing_decay_type == 'exp':
            forcing_ratio = config.forcing_ratio * (config.forcing_decay ** batch)
        elif config.forcing_decay_type == 'sigmoid':
            forcing_ratio = config.forcing_ratio * config.forcing_decay / (
                    config.forcing_decay + math.exp(batch / config.forcing_decay))
        else:
            forcing_ratio = config.forcing_ratio
        return forcing_ratio

    def k_fold(self, k: int):
        k = k % config.k_fold
        span = self.example_len // config.k_fold
        start = k * span
        end = (k + 1) * span
        valid_example = self.example[start: end]
        train_start = self.example[0: start]
        train_end = self.example[end:]
        train_example = train_start + train_end
        return DataSet(train_example), DataSet(valid_example)

    def train(self):
        optimizer = optim.Adam(self.QG_model.parameters(), lr=self.lr, weight_decay=config.weight_decay)
        # decay learning rate
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=config.lr_gamma)
        print('[*] Begin training...')
        for epoch in range(config.epochs):
            train_data, valid_data = self.k_fold(epoch)
            train_dl = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collection_function)
            valid_dl = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                    collate_fn=collection_function)
            step = 0
            # Train Stage
            self.QG_model.train()
            for s_ids, s_ext_ids, q_ids, q_ext_ids, a_ids, s_pos, oov_list in train_dl:
                s_ids = s_ids.to(self.device)
                s_ext_ids = s_ext_ids.to(self.device)
                q_ids = q_ids.to(self.device)
                q_ext_ids = q_ext_ids.to(self.device)
                a_ids = a_ids.to(self.device)
                s_pos = s_pos.to(self.device)
                # graph = graph.to(self.device)
                self.QG_model.zero_grad()
                optimizer.zero_grad()
                teach_forcing = self.forcing_learning(self.train_step)
                logits, cov_loss = self.QG_model(s_ids, s_ext_ids, q_ids, a_ids, s_pos, teach_forcing)
                vocab_size = logits.size(-1)
                pred = logits.view(-1, vocab_size)
                target = q_ext_ids[:, 1:].contiguous().view(-1)
                loss = self.criterion(pred, target)
                loss = torch.add(loss, cov_loss)
                loss.backward()
                clip_grad_norm_(self.QG_model.parameters(), max_norm=config.grad_norm)
                optimizer.step()
                step += 1
                print("\r",
                      f"[-] EPOCH {epoch + 1} (train step) {step}/{len(train_data)} || loss:{loss.item()}",
                      end='')
                self.train_step += 1
                # self.debug_writer.add_scalar("avg_batch_loss", loss, global_step=self.train_step)
                # self.debug_writer.add_scalar("avg_coverage loss", cov_loss, global_step=self.train_step)
                # self.debug_writer.add_scalar("teach forcing", teach_forcing, global_step=self.train_step)
                # Valid Stage
                if step % config.train_step == 0:
                    valid_avg_loss = self.valid_step(valid_dl, epoch)
                    # save model
                    if valid_avg_loss < self.best_loss:
                        self.best_loss = valid_avg_loss
                        torch.save(self.QG_model.state_dict(), config.save_model_folder + f'loss_{self.best_loss:.4f}.pt')
                    self.QG_model.train()
            # update learning rate
            # lr_scheduler.step()
            scheduler.step()
            # print("\n[*] Learning rate:", lr_scheduler.get_last_lr())
        # close visualize
        print("[*] Train Done!")
        self.debug_writer.close()

    @torch.no_grad()
    def valid_step(self, valid_data: DataLoader, epoch):
        self.QG_model.eval()
        process = 0
        valid_avg_loss = 0
        total_loss = 0

        print("\n[*] Validate...")
        # metric = Metric(self.vocab, self.config.batch_size)
        for s_ids, s_ext_ids, q_ids, q_ext_ids, a_ids, s_pos, oov_list in valid_data:
            s_ids = s_ids.to(self.device)
            s_ext_ids = s_ext_ids.to(self.device)
            q_ids = q_ids.to(self.device)
            q_ext_ids = q_ext_ids.to(self.device)
            a_ids = a_ids.to(self.device)
            s_pos = s_pos.to(self.device)
            # graph = graph.to(self.device)
            logits, cov_loss = self.QG_model(s_ids, s_ext_ids, q_ids, a_ids, s_pos)
            target = q_ext_ids[:, 1:].contiguous().view(-1)
            vocab_size = logits.size(-1)
            pred = logits.view(-1, vocab_size)
            loss = self.criterion(pred, target)
            loss += cov_loss
            total_loss += loss.item()
            process += 1
            # calculate loss
            valid_avg_loss = total_loss / process

            print('\r',
                  f"[-] EPOCH {epoch + 1} (valid step) {process + 1}/ {len(valid_data)} || loss:{loss.item()}"
                  f"|| valid_avg_loss: {valid_avg_loss}",
                  end='')
            if process % config.valid_step == 0:
                break
        print('\n')
        return valid_avg_loss
