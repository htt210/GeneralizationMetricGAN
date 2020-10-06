import torch
import torch.nn as nn
from .base_model import Discriminator, Generator, weights_init


class MLPDiscriminator(Discriminator):
    def __init__(self, nx, n_hidden, n_hiddenlayer, negative_slope=1e-2,
                 use_label=True, n_labels=10, embed_dim=32):
        super(MLPDiscriminator, self).__init__()
        self.use_label = use_label
        if use_label:
            self.embed = nn.Embedding(n_labels, embed_dim)

        self.net = nn.Sequential()
        i = 0
        self.net.add_module('linear_%d' % i, nn.Linear(nx, n_hidden))
        self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        for i in range(1, n_hiddenlayer):
            self.net.add_module('linear_%d' % i, nn.Linear(n_hidden, n_hidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        self.net.add_module('linear_%d' % (i + 1), nn.Linear(n_hidden, 1))
        self.net.apply(weights_init)

    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)  # flatten the image if needed
        if y is not None and self.use_label:
            embedding = self.embed(y)
            x = torch.cat([x, embedding], dim=1)

        return self.net(x)


class MLPGenerator(Generator):
    def __init__(self, nz, nx, n_hidden, n_hiddenlayer, negative_slope=1e-2,
                 use_label=False, n_labels=-1, embed_dim=32):
        super(MLPGenerator, self).__init__()
        self.net = nn.Sequential()
        self.use_label = use_label
        self.n_labels = n_labels
        self.n_embed = embed_dim

        i = 0
        if self.use_label:
            self.net.add_module('linear_%d' % i, nn.Linear(nz + embed_dim, n_hidden))
            self.embed = nn.Embedding(self.n_labels, self.n_embed)
        else:
            self.net.add_module('linear_%d' % i, nn.Linear(nz, n_hidden))
        self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        for i in range(1, n_hiddenlayer):
            self.net.add_module('linear_%d' % i, nn.Linear(n_hidden, n_hidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        self.net.add_module('linear_%d' % (i + 1), nn.Linear(n_hidden, nx))
        self.net.apply(weights_init)

    def forward(self, z, y=None):
        if y is not None and self.use_label:
            embedding = self.embed(y)
            z = torch.cat([z, embedding], dim=1)

        return self.net(z)
