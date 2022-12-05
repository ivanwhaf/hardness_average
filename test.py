import argparse
import random

import torch
from torch.utils.data import DataLoader
from dataset import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='cifar10_test')
parser.add_argument('-dataset_path', type=str, help='path of dataset', default='F:\Data\dataset')
# parser.add_argument('-batch_size', type=int, help='batch size', default=128)
# parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
# parser.add_argument('-epochs', type=int, help='training epochs', default=100)
# parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
parser.add_argument('-noise_type', type=str, help='noise type', default='symmetric')
parser.add_argument('-noise_rate', type=float, help='noise rate', default=0.8)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(args.seed)

    # noise dataset to txt
    train_set = CIFAR10Noisy(args.dataset_path, train=True, noise_type=args.noise_type, noise_rate=args.noise_rate,
                             need_idx=True)
    train_set.to_txt("data/cifar10_noise_s0.8.txt")


