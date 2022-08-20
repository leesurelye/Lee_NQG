import pickle
import torch
from model import LeeNQG
import config
import config_squad
from trainer import Trainer
import os
from inference import Generator
from utils import load_DRCD_dataset, load_SQuAD_dataset


def main_for_squad(trial_time):
    config.save_model_folder = config_squad.save_model_folder + f'model_{trial_time}/'
    if not os.path.exists(config.save_model_folder):
        os.makedirs(config.save_model_folder)
    device = config.device
    print("[*] Load Dataset")
    example = load_SQuAD_dataset(config_squad.preprocess_folder, 'train')
    print(f"[*] The train data size is {len(example)}")
    # dev_data = DataSet(config.pre_data_folder, 'dev', token2idx_vocab)
    # print('[*] the dev data size: ', len(dev_data))
    print("[*] Load embedding vector")
    embed_vector = pickle.load(open(config_squad.embed_vector_file, 'rb'))

    model = LeeNQG(config.vocab_size, config.emb_dim, config.hid_dim,
                   config.n_layer, config.cov_loss_hyper, config.generate_max_len,
                   device, embed_vector)
    # TODO use parallel model ??
    model.to(device)
    trainer = Trainer(model, example, device)
    trainer.train()


def main(trial_time):
    config.save_model_folder = config.save_model_folder + f'model_{trial_time}/'
    # config random seed
    if not os.path.exists(config.save_model_folder):
        os.makedirs(config.save_model_folder)
    device = config.device
    print("[*] Load Dataset")
    example = load_DRCD_dataset(config.pre_data_folder + 'DRCD/', 'train')
    print(f"[*] The train data size is {len(example)}")
    print("[*] Load embedding vector")
    embed_vector = pickle.load(open(config.embed_vector_file, 'rb'))

    model = LeeNQG(config.vocab_size, config.emb_dim, config.hid_dim,
                   config.n_layer, config.cov_loss_hyper, config.generate_max_len,
                   device, embed_vector)
    model.to(device)
    trainer = Trainer(model, example, device)
    trainer.train()


if __name__ == '__main__':
    # train model
    torch.manual_seed(config.rand_seed)
    torch.cuda.manual_seed(config.rand_seed)
    trial = 32
    print('trial time', trial)
    if config.train:
        main(trial)
        # main_for_squad(trial)
    else:
        # generate sentence the model you train for
        checkpoint = '/home1/liyue/Lee_NQG/checkpoint/model_26/loss_0.7986.pt'
        beam_searcher = Generator(checkpoint, config.output_file_path, trail=trial)
        beam_searcher.decode()
