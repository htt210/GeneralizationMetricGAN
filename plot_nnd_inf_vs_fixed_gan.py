import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse
import re
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='~/github/exps/mdl/mnist/',
                        help='path to the nnd result folder')
    args = parser.parse_args()

    cs = ['r', 'g', 'b', 'y', 'k', 'm']

    folder = os.path.expanduser(args.path)
    train_sizes = ['10k', '30k', '60k']
    nnd_means = {}
    nnd_stds = {}

    fig, ax = plt.subplots(1, 1)
    with open('results/nnd_inf_vs_fixed_gan.txt', 'w+') as resultf:
        for i, ts in enumerate(train_sizes):
            runs = os.listdir(folder + '/' + ts)

            nndts = []
            for run in runs:
                # print(ts, run)
                nndi = []
                with open(folder + '/' + ts + '/' + run + '/nnd.txt') as rf:
                    for line in rf.readlines():
                        vals = line.strip().split(' ')
                        vals = int(vals[0]), float(vals[1]), float(vals[2])
                        nndi.append(vals)
                        # print(vals)
                nndi = torch.tensor(nndi)
                nndts.append(nndi)
                # print(nndi)

            nndts = torch.stack(nndts)
            print(nndts.size())
            nnd_means[ts] = nndts.mean(dim=0)
            nnd_stds[ts] = nndts.std(dim=0)
            print(nnd_means[ts])
            print(nnd_stds[ts])

            resultf.write('ts ' + ts + '\n')
            resultf.write('mean\n')
            resultf.write(str(nnd_means[ts]) + '\n')
            resultf.write('std\n')
            resultf.write(str(nnd_stds[ts]) + '\n')

            ax.errorbar(x=nnd_means[ts][1:, 0].long(), y=nnd_means[ts][1:, 1], yerr=nnd_stds[ts][1:, 1],
                        label='Fix ' + ts, linestyle='--', capsize=4, c=cs[2 * i])
            ax.errorbar(x=nnd_means[ts][1:, 0], y=nnd_means[ts][1:, 2], yerr=nnd_stds[ts][1:, 2],
                        label='Inf ' + ts, linestyle='-', capsize=4, c=cs[2 * i + 1])
        print()
    ax.legend(fontsize=16)
    ax.set_ylabel('NND', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=16)
    fig.savefig('results/nnd_inf_vs_fixed_gan.pdf', bbox_inches='tight')
