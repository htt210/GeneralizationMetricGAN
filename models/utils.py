import torch
import torch.optim as optim
import torch.utils.data as udata
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import models
import torch.autograd as ag
from .mlp import MLPGenerator, MLPDiscriminator
import yaml


def get_d(args):
    img_size = args['data']['img_size']
    n_labels = args['data']['n_labels']
    use_label = args['data']['use_label']
    embed_dim = args['data']['embed_dim']
    z_dim = args['z_dist']['dim']
    if args['generator']['name'] == 'mlp':
        d = MLPDiscriminator(nx=img_size ** 2, n_hiddenlayer=2, n_hidden=512,
                             n_labels=n_labels, use_label=use_label, embed_dim=embed_dim)
        return d
    if args['generator']['name'] == 'dcgan':
        pass
    if args['generator']['name'] == 'resnet':
        pass


def get_g(args):
    img_size = args['data']['img_size']
    n_labels = args['data']['n_labels']
    use_label = args['data']['use_label']
    embed_dim = args['data']['embed_dim']
    z_dim = args['z_dist']['dim']
    if args['generator']['name'] == 'mlp':
        g = MLPGenerator(nz=z_dim, nx=img_size ** 2, n_hidden=512,
                         n_hiddenlayer=2, n_labels=n_labels,
                         use_label=use_label, embed_dim=embed_dim)
        return g
    if args['generator']['name'] == 'dcgan':
        pass
    if args['generator']['name'] == 'resnet':
        pass


def get_c(args):
    img_size = args['data']['img_size']
    n_channels = args['data']['n_channels']
    if args['nnd']['model'] == 'mlp':
        net = models.MLPDiscriminator(nx=img_size * img_size, n_hidden=512,
                                      n_hiddenlayer=3, use_label=False)
    else:
        net = models.DCDiscriminator(img_size=img_size, nc=n_channels, ndf=128)
    return net


def get_optims(param_g, param_d, args):
    lr_g = args['training']['lr_g']
    lr_d = args['training']['lr_d']
    lr_anneal = args['training']['lr_anneal']
    lr_anneal_every = args['training']['lr_anneal_every']

    optim_g = optim.Adam(lr=lr_g, betas=(0.5, 0.999), params=param_g)
    optim_d = optim.Adam(lr=lr_d, betas=(0.5, 0.999), params=param_d)

    return optim_g, optim_d


def build_lr_scheduler(optimizer, config, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_anneal_every'],
        gamma=config['training']['lr_anneal'],
        last_epoch=last_epoch
    )
    return lr_scheduler


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def load_data(args):
    dataset = args['data']['name']
    size = args['data']['img_size']
    batch_size = args['training']['batch_size']
    train_size = args['data']['train_size']
    test_size = args['data']['test_size']

    transform_mnist = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    transform_cifar = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
    ])

    if dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(root=args['data']['root'], train=True,
                                                transform=transform_mnist, download=True)
        test_data = torchvision.datasets.MNIST(root=args['data']['root'], train=False,
                                               transform=transform_mnist, download=True)
        train_x = torch.stack([x[0] for x in train_data], dim=0)[:train_size]
        train_y = torch.LongTensor([x[1] for x in train_data])[:train_size]
        test_x = torch.stack([x[0] for x in test_data], dim=0)[:test_size]
        test_y = torch.LongTensor([x[1] for x in test_data])[:test_size]
        train_data = udata.TensorDataset(train_x, train_y)
        test_data = udata.TensorDataset(test_x, test_y)

        n_labels = 10
    elif dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args['data']['root'], train=True,
                                                  transform=transform_cifar, download=True)
        test_data = torchvision.datasets.CIFAR10(root=args['data']['root'], train=True,
                                                 transform=transform_cifar, download=True)
        train_x = torch.stack([x[0] for x in train_data], dim=0)[:train_size]
        train_y = torch.LongTensor([x[1] for x in train_data])[:train_size]
        test_x = torch.stack([x[0] for x in test_data], dim=0)[:test_size]
        test_y = torch.LongTensor([x[1] for x in test_data])[:test_size]
        train_data = udata.TensorDataset(train_x, train_y)
        test_data = udata.TensorDataset(test_x, test_y)

        n_labels = 10
    else:
        train_data = torchvision.datasets.ImageFolder(root=args['data']['root'], transform=transform_cifar)
        test_data = torchvision.datasets.ImageFolder(root=args['data']['root'], transform=transform_cifar)
        n_labels = 1

    train_data = udata.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=16, pin_memory=True, drop_last=True)
    test_data = udata.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                 num_workers=16, pin_memory=True, drop_last=True)

    return train_data, test_data, n_labels


def get_z_dist(args, device):
    if args['z_dist']['type'] == 'gauss':
        loc = torch.zeros(args['z_dist']['dim'], device=device)
        scale = loc + 1.
        z_dist = torch.distributions.Normal(loc=loc, scale=scale)
    if args['z_dist']['type'] == 'uniform':
        low = torch.zeros(args['z_dist']['dim'], device=device) - 1.
        high = low + 2.
        z_dist = torch.distributions.Uniform(low=low, high=high)
    z_dist.dim = args['z_dist']['dim']
    return z_dist


def get_y_dist(args, device):
    n_labels = args['data']['n_labels']
    logits = torch.zeros(n_labels, device=device)
    y_dist = torch.distributions.categorical.Categorical(logits=logits)

    # Add n_labels attribute
    y_dist.n_labels = n_labels

    return y_dist


def compute_loss(gan_type, d_out, target):
    """
    :param gan_type: name of the loss function: wgan | standard
    :param d_out: discriminator's output
    :param target: 1 for real, 0 for fake
    :return:
    """
    targets = d_out.new_full(size=d_out.size(), fill_value=target)

    if gan_type == 'standard':
        loss = F.binary_cross_entropy_with_logits(d_out, targets)
    elif gan_type == 'wgan':
        loss = (2 * target - 1) * d_out.mean()
    else:
        raise NotImplementedError

    return loss


def cal_grad_pen(D, real_batch, fake_batch, gp_weight, gp_center, gp_inter, label_batch=None) -> torch.Tensor:
    # print(args.gp_inter)
    device = real_batch.device
    batch_size = real_batch.size(0)
    if gp_inter == 'real':
        alpha = torch.tensor(1., device=device)  # torch.rand(real_data.size(0), 1, device=device)
    elif gp_inter == 'fake':
        alpha = torch.tensor(0., device=device)
    else:
        alpha = torch.rand(batch_size, 1, device=device)
    if not isinstance(D, models.MLPDiscriminator):
        alpha = alpha.view(-1, 1, 1, 1)
    else:
        real_batch = real_batch.view(batch_size, -1)
        fake_batch = fake_batch.view(batch_size, -1)
    alpha = alpha.expand(real_batch.size())

    interpolates = alpha * real_batch + ((1 - alpha) * fake_batch)
    interpolates.requires_grad_(True)
    if label_batch is None:
        disc_interpolates = D(interpolates)
    else:
        disc_interpolates = D(interpolates, label_batch)
    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - gp_center) ** 2).mean() * gp_weight
    return gradient_penalty

