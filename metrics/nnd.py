from models.utils import cal_grad_pen, compute_loss
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm, trange
from .utils import *
import os



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
            if not isinstance(real, torch.Tensor):
                real = real[0]
            if not isinstance(fake, torch.Tensor):
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
            if not isinstance(real, torch.Tensor):
                real = real[0]
            if not isinstance(fake, torch.Tensor):
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


def nnd_iter(C, gan_loss, real_data, fake_data, lr, betas,
             noise_weight, noise_dist, gp_weight, n_iter, device):
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

    # writer = SummaryWriter(log_dir=os.path.expanduser('~/github/runs/'))

    for p in C.parameters():
        p.requires_grad_()
    C.train()
    C.to(device)

    real_iter = iter(real_data)
    fake_iter = iter(fake_data)

    optimizer = optim.Adam(C.parameters(), lr=lr, betas=betas)
    for iidx in trange(n_iter):
        # for idx, (real, fake) in enumerate(zip(real_data, fake_data)):
        try:
            real = next(real_iter)
        except:
            real_iter = iter(real_data)
            real = next(real_iter)
        try:
            fake = next(fake_iter)
        except:
            fake_iter = iter(fake_data)
            fake = next(fake_iter)
            
        optimizer.zero_grad()
        if not isinstance(real, torch.Tensor):
            real = real[0]
        if not isinstance(fake, torch.Tensor):
            fake = fake[0]

        real = real.to(device)
        fake = fake.to(device)

        if noise_dist == 'gauss':
            noise = noise_weight * torch.randn_like(fake)
        else:  # uniform [-1, 1]
            noise = noise_weight * (torch.rand_like(fake) * 2 - 1)
        fake += noise

        pred_real = C(real)
        loss_real = compute_loss(gan_loss, pred_real, 1.)
        pred_fake = C(fake)
        loss_fake = compute_loss(gan_loss, pred_fake, 0.)
        grad_pen = cal_grad_pen(C, real_batch=real,
                                fake_batch=fake,
                                gp_weight=gp_weight,
                                gp_center=1.,
                                gp_inter=-1)
        loss_reg = -(loss_real + loss_fake) + grad_pen
        loss_reg.backward()
        optimizer.step()

        # writer.add_scalar('nnd_iter/loss_fake', loss_fake, iidx)
        # writer.add_scalar('nnd_iter/loss_real', loss_real, iidx)
        # writer.add_scalar('nnd_iter/loss_reg', loss_reg, iidx)

    with torch.no_grad():
        loss_real = torch.tensor(0.)
        for ridx, real in enumerate(real_data):
            if not isinstance(real, torch.Tensor):
                real = real[0]
            real = real.to(device)
            pred_real = C(real)
            loss_real += compute_loss(gan_loss, pred_real, 1)
        loss_real /= (ridx + 1)

        loss_fake = torch.tensor(0.)
        for fidx, fake in enumerate(fake_data):
            if not isinstance(fake, torch.Tensor):
                fake = fake[0]
            fake = fake.to(device)
            pred_fake = C(fake)
            loss_fake += compute_loss(gan_loss, pred_fake, 0)
        loss_fake /= (fidx + 1)
        print('nnd_iter ', ridx, loss_real, fidx, loss_fake, loss_real + loss_fake)
        loss = loss_real + loss_fake
        return loss.item()


def nnd_iter_gen(C, G, gan_loss, real_data, z_dist, y_dist, lr, betas,
                 noise_weight, noise_dist, gp_weight, n_iter, device):
    """

    :param C:
    :param G:
    :param gan_loss:
    :param real_data:
    :param z_dist:
    :param y_dist:
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

    # writer = SummaryWriter(log_dir=os.path.expanduser('~/github/runs/'))

    for p in C.parameters():
        p.requires_grad_()
    C.train()
    C.to(device)
    G.to(device)

    real_iter = iter(real_data)
    batch_size = -1

    optimizer = optim.Adam(C.parameters(), lr=lr, betas=betas)
    for iidx in trange(n_iter):
        try:
            real = next(real_iter)
        except:
            real_iter = iter(real_data)
            real = next(real_iter)

        optimizer.zero_grad()
        if not isinstance(real, torch.Tensor):
            real = real[0]

        batch_size = real.size(0)
        with torch.no_grad():
            z = z_dist.sample((batch_size, ))
            y = y_dist.sample((batch_size, ))
            fake = G(z, y)

        real = real.to(device)
        fake = fake.to(device)

        if noise_dist == 'gauss':
            noise = noise_weight * torch.randn_like(fake)
        else:  # uniform [-1, 1]
            noise = noise_weight * (torch.rand_like(fake) * 2 - 1)
        fake += noise

        pred_real = C(real)
        loss_real = compute_loss(gan_loss, pred_real, 1.)
        pred_fake = C(fake)
        loss_fake = compute_loss(gan_loss, pred_fake, 0.)
        grad_pen = cal_grad_pen(C, real_batch=real,
                                fake_batch=fake,
                                gp_weight=gp_weight,
                                gp_center=1.,
                                gp_inter=-1)
        loss_reg = -(loss_real + loss_fake) + grad_pen
        loss_reg.backward()
        optimizer.step()

        # writer.add_scalar('nnd_iter_gen/loss_fake', loss_fake, iidx)
        # writer.add_scalar('nnd_iter_gen/loss_real', loss_real, iidx)
        # writer.add_scalar('nnd_iter_gen/loss_reg', loss_reg, iidx)

    with torch.no_grad():
        loss_real = torch.tensor(0.)
        for ridx, real in enumerate(real_data):
            if not isinstance(real, torch.Tensor):
                real = real[0]
            real = real.to(device)
            pred_real = C(real)
            loss_real += compute_loss(gan_loss, pred_real, 1)
        loss_real /= (ridx + 1)

        loss_fake = torch.tensor(0.)
        for fidx in range(ridx + 1):
            z = z_dist.sample((batch_size,))
            # y = y_dist.sample((batch_size,))
            fake = G(z, y)
            pred_fake = C(fake)
            loss_fake += compute_loss(gan_loss, pred_fake, 0)
        loss_fake /= (fidx + 1)
        print('nnd_iter_gen ', ridx, loss_real, fidx, loss_fake, loss_real + loss_fake)

        loss = loss_real + loss_fake
        return loss.item()


