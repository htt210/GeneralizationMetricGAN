import torch
import torch.nn as nn
import torch.utils.data as udata
import torch.autograd as ag
import torchvision
import models
import torch.nn.functional as F
from tqdm import tqdm, trange
import torch.utils.tensorboard
from time import time
import os
import metrics
from models import cal_grad_pen, compute_loss


class GAN(object):
    def __init__(self, args, device):
        self.G = models.get_g(args)
        self.D = models.get_d(args)
        self.G.to(device)
        self.D.to(device)
        self.optim_g, self.optim_d = models.get_optims(param_g=self.G.parameters(),
                                                       param_d=self.D.parameters(),
                                                       args=args)
        self.z_dist = models.get_z_dist(args, device=device)
        self.y_dist = models.get_y_dist(args, device=device)
        self.img_size = args['data']['img_size']
        self.n_channels = args['data']['n_channels']
        self.loss = args['training']['loss']
        self.args = args
        self.device = device
        self.log_dir = os.path.expanduser(self.args['training']['out_dir'] + '/' + str(time()) + '/')
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_dir + 'config.txt', 'w') as cf:
            cf.write(str(self.args))

    def _train_d(self, real_batch, noise_batch):
        self.optim_d.zero_grad()
        models.toggle_grad(self.G, False)
        models.toggle_grad(self.D, True)

        # train real
        if len(real_batch) == 2:
            real_batch_x, real_batch_y = real_batch
            real_batch_x = real_batch_x.to(self.device)
            real_batch_y = real_batch_y.to(self.device)
        else:
            real_batch_x = real_batch.to(self.device)
            real_batch_y = None
        pred_real = self.D(real_batch_x, real_batch_y)
        loss_real = compute_loss(self.loss, pred_real, 1)
        loss_real.backward()

        # train fake
        fake_batch = self.G(noise_batch, real_batch_y).data
        pred_fake = self.D(fake_batch, real_batch_y)
        loss_fake = compute_loss(self.loss, pred_fake, 0)
        loss_fake.backward()

        # grad pen
        grad_pen = cal_grad_pen(self.D, real_batch=real_batch_x, fake_batch=fake_batch,
                                gp_weight=self.args['training']['gp_weight'],
                                gp_inter=self.args['training']['gp_inter'],
                                gp_center=self.args['training']['gp_center'],
                                label_batch=real_batch_y)
        grad_pen.backward()

        # update params
        self.optim_d.step()

        loss_d = loss_real + loss_fake + grad_pen
        return loss_d.item()

    def _train_g(self, noise_batch, label_batch):
        self.optim_g.zero_grad()
        models.toggle_grad(self.G, True)
        models.toggle_grad(self.D, False)

        fake_batch = self.G(noise_batch, label_batch)
        pred_fake = self.D(fake_batch, label_batch)
        loss_fake = compute_loss(self.loss, pred_fake, 1)
        loss_fake.backward()

        # update params
        self.optim_g.step()

        return loss_fake.item()

    def train(self):
        train_data, test_data, n_labels = models.load_data(self.args)
        n_epochs = self.args['training']['n_epochs']
        batch_size = self.args['training']['batch_size']
        d_steps = self.args['training']['d_steps']
        log_interval = self.args['training']['log_interval']
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.log_dir + '/run/')

        self.fixed_z = self.z_dist.sample((64,))
        self.fixed_y = self.y_dist.sample((64,))
        self.z_start = self.z_dist.sample((32,))
        self.z_end = self.z_dist.sample((32,))
        self.z_inter_list = metrics.slerp(self.z_start, self.z_end, 14)

        for eidx in trange(n_epochs, leave=True, desc='Epoch'):
            for iidx, real_batch in enumerate(tqdm(train_data)):
                noise_batch = self.z_dist.sample((batch_size,))
                loss_d = self._train_d(real_batch=real_batch, noise_batch=noise_batch)
                self.writer.add_scalar('Loss/%d/d' % eidx, loss_d, global_step=iidx)

                if iidx % d_steps == 0:
                    # train G
                    if len(real_batch) == 2:
                        label_batch = real_batch[1]
                    else:
                        label_batch = None
                    loss_g = self._train_g(noise_batch=noise_batch, label_batch=label_batch)
                    self.writer.add_scalar('Loss/%d/g' % eidx, loss_g, global_step=iidx)
            if eidx % log_interval == 0:
                self._log(eidx, test_data)

    def _evaluate(self):
        pass

    def _log(self, eidx, test_data):
        img_size = self.args['data']['img_size']
        n_channels = self.args['data']['n_channels']

        with torch.no_grad():
            fixed_fake = self.G(self.fixed_z, self.fixed_y)
            fixed_fake = fixed_fake.view(fixed_fake.size(0), self.n_channels, self.img_size, self.img_size)
            torchvision.utils.save_image(fixed_fake, self.log_dir + 'fake_%05d.jpg' % eidx)

            inter_fake = [self.G(z) for z in self.z_inter_list]
            inter_fake = torch.cat(inter_fake, dim=0)
            inter_fake = inter_fake.view(inter_fake.size(0), -1,
                                         self.args['data']['img_size'],
                                         self.args['data']['img_size'])
            torchvision.utils.save_image(inter_fake, self.log_dir + 'inter_%05d.jpg' % eidx,
                                         nrow=32, normalize=inter_fake.size(1) > 1)
        if self.args['nnd']['enable']:
            print('computing nnd')

            # generate data
            sample_size = self.args['nnd']['sample_size']
            batch_size = self.args['training']['batch_size']
            n_batches = sample_size // batch_size + 1
            fake_data = []
            with torch.no_grad():
                for _ in range(n_batches):
                    z = self.z_dist.sample((batch_size, ))
                    y = self.y_dist.sample((batch_size, ))
                    fake_batch = self.G(z, y)
                    fake_data.append(fake_batch)
            fake_data = torch.cat(fake_data, dim=0)
            fake_data = udata.TensorDataset(fake_data)
            fake_data = udata.DataLoader(fake_data, batch_size=batch_size, shuffle=True, drop_last=True)

            # compute 2 nnd scores
            # build nets
            net = models.get_c(self.args)
            nnd_fixed = metrics.nnd_iter(C=net, gan_loss='wgan', real_data=test_data,
                                         fake_data=fake_data, lr=self.args['nnd']['lr'], betas=(0.9, 0.999),
                                         noise_weight=self.args['nnd']['noise_weight'],
                                         noise_dist=self.args['nnd']['noise_dist'],
                                         gp_weight=self.args['nnd']['gp_weight'],
                                         n_iter=self.args['nnd']['n_iters'], device=self.device)

            # build nets
            net = models.get_c(self.args)
            nnd_inf = metrics.nnd_iter_gen(C=net, G=self.G, gan_loss='wgan', real_data=test_data,
                                           z_dist=self.z_dist, y_dist=self.y_dist, lr=self.args['nnd']['lr'],
                                           betas=(0.9, 0.999), noise_weight=self.args['nnd']['noise_weight'],
                                           noise_dist=self.args['nnd']['noise_dist'],
                                           gp_weight=self.args['nnd']['gp_weight'],
                                           n_iter=self.args['nnd']['n_iters'], device=self.device)

            # log
            self.writer.add_scalar('nnd_fixed_%05d' % eidx, nnd_fixed, global_step=eidx)
            self.writer.add_scalar('nnd_inf_%05d' % eidx, nnd_inf, global_step=eidx)
            with open(self.log_dir + 'nnd.txt', 'a+') as nnd_file:
                nnd_file.write('%d %f %f\n' % (eidx, nnd_fixed, nnd_inf))
        # end if
