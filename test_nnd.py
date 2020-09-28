import os
import yaml
from matplotlib import pyplot as plt
from metrics import nnd, nnd_iter, MNISTMNLPClassifier, MNISTCNNClassifier, VGG
import torchvision.datasets as dsets
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import argparse
import configs.utils as cutils
import models
from time import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-base_config', type=str,
    # default='configs/mnist/mnist_basic.yaml', help='path to base config')
    parser.add_argument('-model', type=str, default='mlp', help='classifier architecture: mlp | cnn')
    parser.add_argument('-data', type=str, default='mnist', help='dataset to use: cifar10 | mnist')
    parser.add_argument('-epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('-lr', type=float, default=0.005, help='learning rate')
    # parser.add_argument('-beta1', type=float, default=0.5)
    # parser.add_argument('-beta2', type=float, default=0.999)
    parser.add_argument('-noise_weight', type=float, default=0., help='noise weight')
    parser.add_argument('-noise_dist', type=str, default='uniform', help='noise distribution: gauss | uniform')
    parser.add_argument('-gp_weight', type=float, default=10., help='gp weight')
    # parser.add_argument('-gp_center', type=float, default=1., help='gp center')
    # parser.add_argument('-gp_inter', type=float, default=-1., help='gp interpolation, -1 for random')

    parser.add_argument('-device', type=str, default='cuda:0', help='device to run on')

    args = parser.parse_args()
    # with open(args.base_config, 'r') as f:
    #     config = yaml.load(f)
    #     print(config)
    # cutils.update_config(config['nnd'], args.__dict__)
    # print(config)

    print(args)

    config_str = ''
    for k, v in args.__dict__.items():
        if k != 'device':
            config_str += '_' + k + '_' + str(v)

    print(config_str)
    if not torch.cuda.is_available():
        exit()

    n_exps = 10
    size = 28 if args.data == 'mnist' else 32
    nc = 1 if args.data == 'mnist' else 3
    lr = args.lr
    batch_size = 128
    device = args.device
    n_epoch = args.epochs

    transformMnist = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
    ])

    if args.data == 'mnist':
        train_data = dsets.MNIST(root='~/github/data/mnist/', train=True, transform=transformMnist, download=True)
        test_data = dsets.MNIST(root='~/github/data/mnist/', train=False, transform=transformMnist, download=True)
    else:
        train_data = dsets.CIFAR10(root='~/github/data/cifar10/', train=True, transform=transform, download=True)
        test_data = dsets.CIFAR10(root='~/github/data/cifar10/', train=False, transform=transform, download=True)

    train_x = []
    train_y = []
    for (x, y) in train_data:
        train_x.append(x)
        train_y.append(y)

    train_x = torch.stack(train_x, dim=0)
    train_y = torch.LongTensor(train_y)
    print(train_x.size())
    print(train_y.size())

    noise_weights = [args.noise_weight]  # [0., 0.1, 0.5, 1]

    train_sizes = [1000, 2500, 5000, 7500, 10000, 30000, 60000]
    test_data = DataLoader(dataset=test_data, batch_size=100, shuffle=True,
                           drop_last=True, num_workers=16, pin_memory=True)

    nnds = [[[] for j in range(len(train_sizes))] for i in range(len(noise_weights))]

    fp = 'results/nnd/' + str(time()) + '/'
    os.makedirs(fp, exist_ok=True)

    with open(fp + 'nnd_configs.txt', 'w') as cf:
        cf.write(str(args))

    with open(fp + 'nnd_score.txt', 'w') as rf:
        rf.write('train_sizes: ' + str(train_sizes) + '\n')
        rf.write('noise_weights: ' + str(noise_weights) + '\n')

        for i, noise_weight in enumerate(noise_weights):
            for j, train_size in enumerate(train_sizes):
                for eidx in range(n_exps):
                    if args.arch == 'mlp':
                        net = models.MLPDiscriminator(nx=size * size, n_hidden=512,
                                                      n_hiddenlayer=3, use_label=False)
                    else:
                        net = models.DCDiscriminator(img_size=size, nc=nc, ndf=128)

                    train_data = TensorDataset(train_x[:train_size], train_y[:train_size])
                    train_data = DataLoader(dataset=train_data, batch_size=128, shuffle=True,
                                            drop_last=True, num_workers=16, pin_memory=True)

                    nndij = nnd_iter(C=net, gan_loss='wgangp', real_data=test_data.__iter__(),
                                     fake_data=train_data.__iter__(), lr=args.lr, betas=(0.5, 0.999),
                                     noise_weight=noise_weight, noise_dist=args.noise_dist,
                                     n_iter=args.n_iter, device=args.device)
                    nnds[i][j].append(nndij)

                    rf.write(str(nndij) + ' ')
                    rf.flush()
                rf.write('\n')
                rf.flush()  # end of a experiment
            rf.write('\n')  # end of a noise_weight
