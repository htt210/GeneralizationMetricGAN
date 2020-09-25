import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
from tqdm import tqdm, trange


class TSTR(object):
    def __init__(self, classifier, optimizer, train_data, test_data, loss, n_epoch, device):
        self.classifier = classifier
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.n_epoch = n_epoch
        self.device = device
        self.loss = loss

    def train(self):
        self.classifier.to(self.device)
        for eidx in trange(self.n_epoch, desc='Epoch'):
            for bidx, (x, y) in enumerate(self.train_data):
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                y_hat = self.classifier(x)
                loss_i = self.loss(y_hat, y)
                loss_i.backward()
                self.optimizer.step()

        return self.classifier

    def test(self):
        self.classifier.eval()
        acc_av = 0.
        loss_av = 0.
        with torch.no_grad():
            for bidx, (x, y) in enumerate(tqdm(self.test_data)):
                bidx1 = bidx + 1
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.classifier(x)
                loss_i = self.loss(y_hat, y)
                acc_i = (y_hat.argmax(dim=1) == y.squeeze()).float().mean()
                acc_av = acc_av * (1. - 1. / bidx1) + acc_i / bidx1
                loss_av = loss_av * (1. - 1. / bidx1) + loss_i / bidx1

            return acc_av.item(), loss_av.item()

