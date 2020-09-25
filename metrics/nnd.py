import torch
import torch.optim as optim
import tqdm
from .utils import *
from gans import compute_loss, cal_grad_pen


def nnd(C, gan_loss, real_data, fake_data, args):
    """
    Compute NND between real and fake data
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

    optimizer = optim.Adam(C.parameters(), lr=args['nnd_lr'], betas=(args['nnd_beta1'], args['nnd_beta2']))
    for eidx in tqdm(range(args['nnd_epoch'])):
        for idx, (real, fake) in enumerate(zip(real_data, fake_data)):
            optimizer.zero_grad()
            # print(real)
            if isinstance(real, list):
                real = real[0]
            if isinstance(fake, list):
                fake = fake[0]
            real = real.to(device)
            fake = fake.to(device)
            nnd_noise = args.get('nnd_noise', 0)
            # print('noisy')
            noise = nnd_noise * torch.randn_like(fake)
            fake += noise
            pred_real = C(real)
            loss_real = compute_loss(gan_loss, pred_real, 1)
            pred_fake = C(fake)
            loss_fake = compute_loss(gan_loss, pred_fake, 0)
            grad_pen = cal_grad_pen(C, real_batch=real, fake_batch=fake,
                                    gp_weight=args['nnd_gp_weight'],
                                    gp_center=args['nnd_gp_center'],
                                    gp_inter=args['nnd_gp_inter'])
            loss_reg = -(loss_real + loss_fake) + grad_pen
            loss_reg.backward()
            optimizer.step()
        # print(loss_real.item(), loss_fake.item(), grad_pen.item(), loss_reg.item())

    loss = 0.
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
