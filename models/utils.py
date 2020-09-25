import torch
import torch.nn as nn
from .mlp import MLPGenerator, MLPDiscriminator
import yaml


def get_d(args):
    pass


def get_g(args):
    img_size = args['data']['img_size']
    n_labels = args['data']['n_labels']
    use_label = args['data']['use_label']
    embed_dim = args['data']['embed_dim']
    z_dim = args['z_dist']['dim']
    if args['generator']['name'] == 'mlp':
        g = MLPGenerator(nz=z_dim, nx=img_size ** 2, n_hidden=512, n_hiddenlayer=2,
                         n_labels=n_labels,
                         use_label=use_label, embed_dim=embed_dim)
        return g
    if args['generator']['name'] == 'dcgan':
        pass
    if args['generator']['name'] == 'resnet':
        pass


def load_configs(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    return config


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
