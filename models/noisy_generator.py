import torch
import torch.nn as nn
from .base_model import Generator


class NoisyGenerator(Generator):
    def __init__(self, data, nz):
        super(nn.Module, self).__init__()
        self.data = data  # n_data * nx
        self.n_data = data.size(0)
        self.net = nn.Sequential(nn.Linear(nz, self.n_data), nn.Softmax(dim=1))

    def forward(self, z, y=None):
        selector = self.net(z)  # batch_size * n_data
        x = selector.mm(self.data)  # batch_size * nx
        return x


if __name__ == '__main__':

    G = NoisyGenerator(data, nz)
