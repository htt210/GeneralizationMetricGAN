import torch
import torch.nn as nn


class Generator(nn.Module):
    def forward(self, x, y=None):
        pass


class Discriminator(nn.Module):
    def forward(self, x, y=None):
        pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.train_data.normal_(0.0, 0.02)
        m.bias.train_data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.train_data.normal_(1.0, 0.02)
        m.bias.train_data.fill_(0)


def weights_init_conv(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.train_data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.train_data.normal_(1.0, 0.02)
        m.bias.train_data.fill_(0)

