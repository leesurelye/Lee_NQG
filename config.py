# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 21:11
# @Author  : Leesure
# @File    : config.py
# @Software: PyCharm
import torch
from tensorboardX import SummaryWriter

"""
    Notice: This is the config file of training the model and evaluate the predicted sentence 
    if you want to train the model in your own computer, please change the file path according to 
    the parameter meanings
"""
"""train=True for train, Train=False for generate"""
train = False
dataset = 'drcd'
rand_seed = 1234
lr = 0.001
weight_decay = 0
batch_size = 16
k_fold = 10
"""every 200 times for 10 valid """
train_step = 300
# only valid 10 batch data, don't need valid all the valid data set
valid_step = 200
# the learning rate decay ratio
lr_decay_step = 3
lr_gamma = 0.5
epochs = 20
min_decode_step = 8
max_decode_step = 30
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
emb_dim = 300
hid_dim = 256
forcing_decay_type = 'linear'
forcing_ratio = 1
decay_ratio = 0.5
forcing_decay = 0.9999
n_layer = 3
dropout = 0.3
grad_norm = 5.0
valid_percentage = 0.2
cov_loss_hyper = 0.5
generate_max_len = 50
beam_size = 15
coverage_step = 7
# if bleu score is better, save model
save_mode = 'bleu'
raw_train_data_path = '/home1/liyue/Lee_NQG/dataset/raw/round1_train_0907.json'
raw_test_data_path = '/home1/liyue/Lee_NQG/dataset/raw/round1_test_0907.json'
debug_file = './debug/tensorboard'
save_model_folder = '/home1/liyue/Lee_NQG/checkpoint/'

pre_data_folder = '/home1/liyue/Lee_NQG/dataset/preprocess/'
vocab_file = '/home1/liyue/Lee_NQG/dataset/preprocess/DRCD/vocab.pt'
embed_vector_file = '/home1/liyue/Lee_NQG/dataset/preprocess/DRCD/embed.pt'
embedded_file = '/home1/liyue/Lee_NQG/embedding/sgns.zhihu.bigram-char.bz2'
output_file_path = '/home1/liyue/Lee_NQG/predict/'
vocab_size = 45000
edge_size = 49
edge_emb_size = 49
pos_size = 37
pos_emb_dim = 16
# tokenizer don't needed at train and valid step

writer = SummaryWriter(debug_file)
