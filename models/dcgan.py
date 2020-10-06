import torch.nn as nn
from .base_model import *


class DCDiscriminator(Discriminator):
    def __init__(self, img_size, nc, ndf):
        super(DCDiscriminator, self).__init__()
        self.img_size = img_size
        self.nc = nc
        self.ndf = ndf
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()  # remove to make it consistent with other discriminators
        )
        self.apply(weights_init_conv)

    def forward(self, x, y=None):
        # print(inputs.size())
        # if len(inputs) > 1:
        #     labels = inputs[1]
        #     inputs = inputs[0]
        # else:
        #     inputs = inputs[0]
        x = x.view(x.size(0), self.nc, self.img_size, self.img_size)
        return self.net(x)


class DCGenerator(Generator):
    def __init__(self, nz, ngf, img_size, nc):
        super(DCGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.img_size = img_size
        self.nc = nc

        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init_conv)

    def forward(self, z, y=None):
        z = z.view(z.size(0), self.nz, 1, 1)
        return self.net(z)
