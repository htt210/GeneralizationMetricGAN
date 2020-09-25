import torch
import torch.nn as nn
import torch.autograd as ag
import models
import torch.nn.functional as F


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


def cal_grad_pen(D, real_batch, fake_batch, gp_weight, gp_center, gp_inter, label_batch=None):
    # print(args.gp_inter)
    alpha = gp_inter
    device = real_batch.device
    batch_size = real_batch.size(0)
    if alpha >= 0:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
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


class GAN(object):
    def __init__(self, args):
        pass

    def train_d(self, real_batch, noise_batch):
        pass

    def train_g(self, noise_batch, label_batch):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

