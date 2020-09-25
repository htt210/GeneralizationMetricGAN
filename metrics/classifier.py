import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import os


class Classifier(nn.Module):

    def nclasses(self):
        pass

    def forward(self, *inputs):
        pass

    def get_features(self, x, layer_name=None):
        pass


class MNISTCNNClassifier(Classifier):
    def __init__(self):
        super(MNISTCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def nclasses(self):
        return 10

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        # print('forward size', x.size())
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_features(self, x, layer_name=None):
        x = x.view(x.size(0), 1, 28, 28)
        # print('imput size', x.size())
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # print('feature size', x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MNISTMNLPClassifier(Classifier):
    def __init__(self):
        super(MNISTMNLPClassifier, self).__init__()
        nx = 28 * 28
        nhidden = 256
        self.nx = nx
        self.nhidden = nhidden

        self.net = nn.ModuleList([
            nn.Linear(nx, nhidden),
            nn.LeakyReLU(negative_slope=1e-1),
            nn.Linear(nhidden, nhidden),
            nn.LeakyReLU(negative_slope=1e-1),
            nn.Linear(nhidden, nhidden),
            nn.LeakyReLU(negative_slope=1e-1),
            nn.Linear(nhidden, 10),
            nn.LogSoftmax(dim=1)
        ])

    def nclasses(self):
        return 10

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.net:
            x = layer(x)
        return x

    def get_features(self, x, layer_name=None):
        x = x.view(x.size(0), -1)
        for layer in self.net[:-3]:
            x = layer(x)
        return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Classifier):
    def __init__(self, vgg_name, img_size):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        scale = img_size / 32.
        self.classifier = nn.Linear(int(512 * scale * scale), 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

    def get_features(self, x, layer_name=None):
        return self.features(x).view(x.size(0), -1)

    def nclasses(self):
        return 10

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        # print(data.size())
        if args.arch == 'mlp':
            data = data.view(batch_size, 28 * 28)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            # print(data.size())
            if args.arch == 'mlp':
                data = data.view(batch_size, 28 * 28)
            output = model(data)
            # print('output.size()', output.size())
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch classifier')
    parser.add_argument('--arch', type=str, default='vgg', help='architecture: mlp | cnn')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--tests-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = True

    torch.manual_seed(args.seed)

    device = args.device
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    dataset = 'mnist'
    if args.arch == 'cnn':
        print('CNN')
        train_loader = torch.utils.train_data.DataLoader(
            datasets.MNIST('~/github/data/mnist/', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))  # (0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.train_data.DataLoader(
            datasets.MNIST('~/github/data/mnist/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # (0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = MNISTCNNClassifier().to(device)
    elif args.arch == 'mlp':
        print('MLP')
        train_loader = torch.utils.train_data.DataLoader(
            datasets.MNIST('~/github/data/mnist/', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))  # (0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.train_data.DataLoader(
            datasets.MNIST('~/github/data/mnist/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # (0.1307,), (0.3081,))
            ])),
            batch_size=100, shuffle=False, **kwargs)
        model = MNISTMNLPClassifier().to(device)
    elif args.arch == 'vgg':
        print('VGG')
        train_loader = torch.utils.train_data.DataLoader(
            datasets.CIFAR10('~/github/data/cifar10/', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(args.img_size),
                                 transforms.RandomCrop(args.img_size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            drop_last=True,
            batch_size=128,
            pin_memory=True,
            num_workers=4,
            shuffle=True
        )
        test_loader = torch.utils.train_data.DataLoader(
            datasets.CIFAR10('~/github/data/cifar10/', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(args.img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            drop_last=True,
            batch_size=100,
            pin_memory=True,
            num_workers=4,
            shuffle=False
        )
        model = VGG('VGG16', img_size=args.img_size)
        dataset = 'cifar10'
    else:
        print('Architecture not supported')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.cpu().state_dict(),
                   os.path.expanduser("~/github/data/" + dataset + "/" + dataset + "_%s_%d.t7" % (args.arch,
                                                                                                  args.img_size)))


if __name__ == '__main__':
    main()
