import argparse
import gans
from configs import load_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda:0', help='device to run')
    parser.add_argument('-config', type=str, default='configs/mnist/mnist_basic.yaml', help='path to config file')
    args = parser.parse_args()
    device = args.device
    config = load_configs(args.config)
    gan = gans.GAN(args=config, device=device)
    gan.train()
