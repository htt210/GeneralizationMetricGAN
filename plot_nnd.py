import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='./results/nnd/1601543121.0444973/',
                        help='path to the nnd result folder')
    args = parser.parse_args()

    with open(args.path + 'nnd_score.txt', 'r') as f, \
            open(args.path + 'nnd_mean_std.txt', 'w') as mvf, open(args.path + 'nnd_configs.txt') as configf:

        config_line = configf.readlines()
        test_size_start = config_line[0].find('test_size=')
        if test_size_start < 0:
            test_size = 10000
        else:
            test_size_start += len('test_size=')

            test_size = int(re.findall(r'[0-9]+', config_line[0][test_size_start:])[0])
            # print(test_size)
        print('test_size ', test_size)

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
        max_nnd = 0
        for nidx, nw in enumerate(noise_weights):
            nndi = nnds[nidx]
            nndi_mean = nndi.mean(dim=1)
            nndi_std = nndi.std(dim=1)
            max_nnd = max(nndi.max().item(), max_nnd)
            (_, caps, _) = ax.errorbar(x=train_sizes, y=nndi_mean, yerr=nndi_std,
                                       marker='.', label='$\epsilon = $' + str(nw),
                                       capsize=4, linestyle=line_styles[nidx])
            for cap in caps:
                cap.set_markeredgewidth(1)

            mvf.write('noise_weight: ' + str(nw) + '\n')
            mvf.write('mean_line: ' + str(nndi_mean) + '\n')
            mvf.write('std_line: ' + str(nndi_std) + '\n')

        ax.set_xticks([0, 5000, 10000, 30000, 60000])
        ax.set_xticklabels(['0', '5', '10', '30', '60'])
        ax.plot([test_size, test_size], [0, max_nnd], linestyle='--', c='k', alpha=0.5)
        ax.annotate('$|\mathcal{D}| = |\mathcal{D}_{test}| = %d$' % test_size, xy=(test_size, max_nnd - 1),
                    xycoords='data', xytext=(0.3, 0.55), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, headwidth=6, width=1),
                    horizontalalignment='left', verticalalignment='top', fontsize=16)

        ax.set_ylabel('NND', fontsize=16)
        ax.set_xlabel('Size of $\mathcal{D}\ (x1000)$', fontsize=16)
        ax.legend(prop={'size': 16})
        fig.savefig('results/nnd/nnd_noise_test_size_%d.pdf' % test_size, bbox_inches='tight')
        plt.show()
