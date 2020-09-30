import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='./results/nnd/1601304007.573754/',
                        help='path to the nnd result folder')
    args = parser.parse_args()

    with open(args.path + 'nnd_score.txt', 'r') as f:
        lines = f.readlines()
        train_sizes = lines[0][len('train_sizes:'):].strip()[1:-1].strip().split(', ')
        train_sizes = [int(ts) for ts in train_sizes]
        print(train_sizes)

        noise_weights = lines[1][len('noise_weights:'):].strip()[1:-1].strip().split(', ')
        noise_weights = [float(nw) for nw in noise_weights]
        print(noise_weights)

        nnds = [[[] for j in range(len(train_sizes))] for i in range(len(noise_weights))]

        nidx, sidx = 0, 0
        for i, line in enumerate(lines):
            if line.startswith('noise_weight '):
                scores = [float(score) for score in lines[i + 1].split()]
                nnds[nidx][sidx].append(scores)
                sidx += 1
            elif len(line.strip()) < 1:
                nidx += 1
                sidx = 0

        # for ni in range(len(noise_weights)):
        #     for si in range(len(train_sizes)):
        #         print(nnds[ni][si])
        #     print()

        nnds = torch.tensor(nnds).squeeze()
        print(nnds.size())

        fig, ax = plt.subplots(1, 1)  # , figsize=(5, 5))

        line_styles = [':', '-.', '--', '-']
        for nidx, nw in enumerate(noise_weights):
            nndi = nnds[nidx]
            nndi_mean = nndi.mean(dim=1)
            nndi_std = nndi.std(dim=1)
            (_, caps, _) = ax.errorbar(x=train_sizes, y=nndi_mean, yerr=nndi_std,
                                       marker='.', label='$\lambda = $' + str(nw),
                                       capsize=4, linestyle=line_styles[nidx])
            for cap in caps:
                cap.set_markeredgewidth(1)

        ax.set_ylabel('NND', fontsize=16)
        ax.set_xlabel('Train set size', fontsize=16)
        ax.legend(prop={'size': 16})
        fig.savefig('results/nnd/nnd_noise.pdf', bbox_inches='tight')
        plt.show()
