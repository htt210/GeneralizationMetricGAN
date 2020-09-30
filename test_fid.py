import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

import metrics

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='mnist', help='mnist | cifar10')
    parser.add_argument('-model', type=str, default='~/github/data/mnist/mnist_classifier.pt',
                        help='path to a pretrained classifier')


