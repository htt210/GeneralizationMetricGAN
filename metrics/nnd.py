import torch
import torch.optim as optim
import tqdm
from .utils import *
from gans import compute_loss, cal_grad_pen


def nnd(C, gan_loss, real_data, fake_data, args):
    """
    Compute NND between 2 data sets of the same size

    :param C: critic
    :param gan_loss: currently support only wgan-1gp loss: wgan
    :param real_data:
    :param fake_data:
    :param args:
    :return: nnd between real and fake data
    """
    if gan_loss != 'wgan':
        raise NotImplementedError('Only wgan-1gp loss is supported. Expected `wgan`, found `' + gan_loss + '`')

    device = args['device']
    for p in C.parameters():
        p.requires_grad_()
    C.train()
    C.to(device)

    optimizer = optim.Adam(C.parameters(), lr=args['nnd']['lr'], betas=(args['nnd']['beta1'], args['nnd']['beta2']))
    for eidx in tqdm(range(args['nnd']['epoch'])):
        for idx, (real, fake) in enumerate(zip(real_data, fake_data)):
            optimizer.zero_grad()
            # print(real)
            if isinstance(real, list):
                real = real[0]
            if isinstance(fake, list):
                fake = fake[0]
            real = real.to(device)
            fake = fake.to(device)
            nnd_noise = args['nnd']['noise_weight']
            if args['nnd']['noise_dist'] == 'gauss':
                noise = nnd_noise * torch.randn_like(fake)
            else:
                noise = nnd_noise * (torch.rand_like(fake) * 2 - 1)
            fake += noise
            pred_real = C(real)
            loss_real = compute_loss(gan_loss, pred_real, 1)
            pred_fake = C(fake)
            loss_fake = compute_loss(gan_loss, pred_fake, 0)
            grad_pen = cal_grad_pen(C, real_batch=real, fake_batch=fake,
                                    gp_weight=args['nnd']['gp_weight'],
                                    gp_center=1.,
                                    gp_inter=-1.)
            loss_reg = -(loss_real + loss_fake) + grad_pen
            loss_reg.backward()
            optimizer.step()
        # print(loss_real.item(), loss_fake.item(), grad_pen.item(), loss_reg.item())

    loss = torch.tensor(0.)
    with torch.no_grad():
        for idx, (real, fake) in enumerate(zip(real_data, fake_data)):
            if isinstance(real, list):
                real = real[0]
            if isinstance(fake, list):
                fake = fake[0]
            real = real.to(device)
            fake = fake.to(device)
            pred_real = C(real)
            loss_real = compute_loss(gan_loss, pred_real, 1)
            pred_fake = C(fake)
            loss_fake = compute_loss(gan_loss, pred_fake, 0)
            loss += loss_real + loss_fake
        loss /= idx
    return loss.item()


def nnd_iter(C, gan_loss, real_data, fake_data, lr, betas, noise_weight, noise_dist, gp_weight, n_iter, device):
    """
    compute nnd between 2 dataset of arbitrary sizes

    :param C:
    :param gan_loss:
    :param real_data:
    :param fake_data:
    :param lr:
    :param betas:
    :param noise_weight:
    :param noise_dist:
    :param gp_weight:
    :param n_iter:
    :param device:
    :return:
    """
    if gan_loss != 'wgan':
        raise NotImplementedError('Only wgan-1gp loss is supported. Expected `wgan`, found `' + gan_loss + '`')

    for p in C.parameters():
        p.requires_grad_()
    C.train()
    C.to(device)

    optimizer = optim.Adam(C.parameters(), lr=lr, betas=betas)
    for iidx in tqdm(range(n_iter)):
        # for idx, (real, fake) in enumerate(zip(real_data, fake_data)):
        real = next(real_data)
        fake = next(fake_data)
        optimizer.zero_grad()
        if isinstance(real, list):
            real = real[0]
        if isinstance(fake, list):
            fake = fake[0]
        real = real.to(device)
        fake = fake.to(device)

        if noise_dist == 'gauss':
            noise = noise_weight * torch.randn_like(fake)
        else:  # uniform [-1, 1]
            noise = noise_weight * (torch.rand_like(fake) * 2 - 1)
        fake += noise

        pred_real = C(real)
        loss_real = compute_loss(gan_loss, pred_real, 1)
        pred_fake = C(fake)
        loss_fake = compute_loss(gan_loss, pred_fake, 0)
        grad_pen = cal_grad_pen(C, real_batch=real,
                                fake_batch=fake,
                                gp_weight=gp_weight,
                                gp_center=1.,
                                gp_inter=-1.)
        loss_reg = -(loss_real + loss_fake) + grad_pen
        loss_reg.backward()
        optimizer.step()
        # print(loss_real.item(), loss_fake.item(), grad_pen.item(), loss_reg.item())

    with torch.no_grad():
        loss_real = torch.tensor(0.)
        for ridx, real in enumerate(real_data):
            real = real.to(device)
            pred_real = C(real)
            loss_real += compute_loss(gan_loss, pred_real, 1)
        loss_real /= ridx

        loss_fake = torch.tensor(0.)
        for fidx, fake in enumerate(fake_data):
            fake = fake.to(device)
            pred_fake = C(fake)
            loss_fake += compute_loss(gan_loss, pred_fake, 0)
        loss_fake /= fidx

        loss = loss_real + loss_fake
        return loss.item()

