# -*- coding: utf-8 -*-
# @Time    : 2022/3/7 16:27
# @Author  : Leesure
# @File    : config_squad.py
# @Software: PyCharm
glove_file = '/home1/liyue/Lee_NQG/embedding/glove.840B.300d.txt'
word2vec_file = '/home1/liyue/Lee_NQG/embedding/glove_word2vec.pt'

raw_dataset_folder = '/home1/liyue/Lee_NQG/dataset/SQuAD/'
embed_vector_file = '/home1/liyue/Lee_NQG/dataset/preprocess/SQuAD/embed.pt'
preprocess_folder = '/home1/liyue/Lee_NQG/dataset/preprocess/SQuAD/'

save_model_folder = '/home1/liyue/Lee_NQG/checkpoint/SQuAD/'
output_file_path = '/home1/liyue/Lee_NQG/predict/SQuAD/'
vocab_size = 45000
emb_dim = 300
pos_vocab_size = 22
dep_vocab_size = 49
