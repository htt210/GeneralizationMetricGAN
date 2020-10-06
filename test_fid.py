import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as dist

import numpy as np
import sklearn
from sklearn_extra.cluster import KMedoids

import metrics

import argparse


def get_medians(data, n_medians):
    kmedoids = KMedoids(n_clusters=n_medians).fit(data)
    # print(kmedoids.cluster_centers_)
    return kmedoids.cluster_centers_


if __name__ == '__main__':
    size = 10000
    n_dims = 256
    n_medians = 1000

    fids = []
    fidjs = []
    for i in range(100):
        data1 = torch.randn((size, n_dims)).numpy()
        data2 = torch.randn((size, n_dims)).numpy()

        mu1 = np.mean(data1, axis=0)
        sigma1 = np.cov(data1, rowvar=False)

        mu2 = np.mean(data2, axis=0)
        sigma2 = np.cov(data2, rowvar=False)

        mu3 = mu2
        medians = get_medians(data2, n_medians=n_medians)
        sigma3 = np.cov(medians, rowvar=False)

        print(np.trace(sigma1), np.trace(sigma2), np.trace(sigma3))

        mvn = dist.multivariate_normal.MultivariateNormal(loc=torch.FloatTensor(mu1),
                                                          covariance_matrix=torch.FloatTensor(sigma1))
        log_prog = mvn.log_prob(torch.FloatTensor(data2))
        print(log_prog.min(), log_prog.max())

        fidi = metrics.calculate_frechet_distance(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
        fidj = metrics.calculate_frechet_distance(mu1=mu1, mu2=mu3, sigma1=sigma1, sigma2=sigma3)
        print(fidi)
        print(fidj)
        print()
        fids.append(fidi)
        fidjs.append(fidj)
        break
    # end for
    print(np.mean(fids), np.var(fids))
    print(np.mean(fidjs), np.var(fidjs))

# end main
