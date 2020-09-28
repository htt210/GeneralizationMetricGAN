import torch
from torch import distributions
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pickle
import os


def get_dataset(name, data_dir, train=True, size=64, lsun_categories=None):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
    ])

    transformMnist = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if name == 'image':
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
        nlabels = len(dataset.classes)
    elif name == 'imagenet32':
        dataset = ImageNetDataset(dir=data_dir, img_size=size)
        nlabels = 1000
    elif name == 'imagenet64':
        dataset = ImageNetDataset(dir=data_dir, img_size=size)
        nlabels = 1000
    elif name == 'mnist':
        dataset = datasets.MNIST(root=data_dir, train=train, download=True,
                                 transform=transformMnist)
        nlabels = 10
    elif name == 'cifar10':
        if train:
            transform_train = transforms.Compose([
                transforms.Resize(size),
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = datasets.CIFAR10(root=data_dir, train=train, download=True,
                                       transform=transform_train)
        else:
            transform_test = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = datasets.CIFAR10(root=data_dir, train=train, download=True,
                                       transform=transform_test)
        nlabels = 10
    elif name == 'fashionmnist':
        dataset = datasets.FashionMNIST(root=data_dir, train=train, download=True,
                                        transform=transformMnist)
        nlabels = 10
    elif name == 'lsun':
        if lsun_categories is None:
            lsun_categories = 'train'
        dataset = datasets.LSUN(data_dir, lsun_categories, transform)
        nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir, transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    elif name == 'celeba':
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        nlabels = 1
    else:
        raise NotImplemented

    return dataset, nlabels


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img / 127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img


# Note that this will work with Python3
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x / np.float16(255)
    mean_image = mean_image / np.float16(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=torch.HalfTensor(X_train),
        Y_train=torch.LongTensor(Y_train),
        mean=torch.HalfTensor(mean_image))


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir, img_size=32):
        n_files = len(os.listdir(dir))
        self.dicts = []
        self.batch_sizes = [0]
        self.n_samples = 0
        for bidx in range(1, n_files + 1):
            print(bidx)
            dicti = load_databatch(dir, bidx, img_size)
            self.n_samples += dicti['X_train'].size(0)
            self.batch_sizes.append(self.n_samples)
            self.dicts.append(dicti)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        for bidx, start_idx in enumerate(self.batch_sizes):
            if idx >= start_idx:
                iidx = idx - start_idx
            else:
                break
        sample = (self.dicts[bidx - 1]['X_train'][iidx].float(), self.dicts[bidx - 1]['Y_train'][iidx])

        return sample


def get_zdist(dist_name, dim, device):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def get_ydist(nlabels, device=None):
    logits = torch.zeros(nlabels, device=device)
    ydist = distributions.categorical.Categorical(logits=logits)
    # Add nlabels attribute
    ydist.nlabels = nlabels

    return ydist

# imagenet = ImageNetDataset(dir='/media/htt210/Data/github/data/imagenet_32x32/Imagenet32_train/', img_size=32)
