from matplotlib import pyplot as plt
from metrics import TSTR, MNISTMNLPClassifier, MNISTCNNClassifier, VGG
import torchvision.datasets as dsets
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-arch', type=str, default='mlp', help='classifier architecture')
    parser.add_argument('-data', type=str, default='mnist', help='dataset to use')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('-device', type=str, default='cuda:0', help='device to run on')
    parser.add_argument('-noise', action='store_true', default=False, help='use noise')

    args = parser.parse_args()
    config_str = ''
    for k, v in args.__dict__.items():
        if k != 'device':
            config_str += '_' + k + '_' + str(v)

    print(config_str)
    if not torch.cuda.is_available():
        exit()

    n_exps = 10
    size = 28 if args.data == 'mnist' else 32
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
    print(len(train_x))
    train_x = torch.stack(train_x, dim=0)
    train_y = torch.LongTensor(train_y)
    print(train_x.size())
    print(train_y.size())

    train_sizes = [1000, 10000, 20000, 30000, 40000, 50000, 60000]
    test_data = DataLoader(dataset=test_data, batch_size=100, shuffle=True,
                           drop_last=True, num_workers=16, pin_memory=True)

    acc_list = [[] for _ in range(len(train_sizes))]
    loss_list = [[] for _ in range(len(train_sizes))]
    for i, train_size in enumerate(train_sizes):
        for eidx in range(n_exps):
            if args.arch == 'mlp':
                print('using MLP')
                classifier = MNISTMNLPClassifier()
            else:
                print('using CNN')
                if args.data == 'mnist':
                    print('CNN')
                    classifier = MNISTCNNClassifier()
                else:
                    print('VGG')
                    classifier = VGG('VGG16', img_size=size)
            optimizer = optim.Adam(lr=lr, betas=(0.9, 0.999), params=classifier.parameters())

            train_data = TensorDataset(train_x[:train_size], train_y[:train_size])
            train_data = DataLoader(dataset=train_data, batch_size=128, shuffle=True,
                                    drop_last=True, num_workers=16, pin_memory=True)

            tstr = TSTR(classifier=classifier, optimizer=optimizer, train_data=train_data,
                        test_data=test_data, loss=nn.NLLLoss(), device=device, n_epoch=n_epoch)

            tstr.train()
            acc, loss = tstr.test()
            acc_list[i].append(acc)
            loss_list[i].append(loss)
            print('\n', 'Train size', train_size, 'Acc:', acc, 'Loss:', loss)

    acc_list = torch.tensor(acc_list)
    loss_list = torch.tensor(loss_list)
    acc_mean = acc_list.mean(dim=1)
    acc_std = acc_list.std(dim=1)
    loss_mean = loss_list.mean(dim=1)
    loss_std = loss_list.std(dim=1)

    with open('results/tstr/tstr_train_sizes%s.txt' % config_str, 'w') as f:
        f.write('Train sizes\n')
        f.write(str(train_sizes))
        f.write('\nAcc_list\n')
        f.write(str(acc_list))
        f.write('\nLoss_list\n')
        f.write(str(loss_list))

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x=train_sizes, y=acc_mean, yerr=acc_std)
    fig.savefig('results/tstr/TSTR accuracy%s.pdf' % config_str,
                bbox_inches='tight')
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x=train_sizes, y=loss_mean, yerr=loss_std)
    fig.savefig('results/tstr/TSTR loss%s.pdf' % config_str,
                bbox_inches='tight')
