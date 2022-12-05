import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(0)

class CIFAR10FromTxt(Dataset):
    def __init__(self, txt_path, root, train, transform, download, need_idx):
        self.need_idx =  need_idx
        self.base_dataset = datasets.CIFAR10(root, train, transform, download=download)
        self.raw_idxs = []
        self.targets = []
        self.txt_path = txt_path

        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                line = line.split(' ')
                raw_idx = int(line[0][8:])
                label = int(line[1][6:])
                self.raw_idxs.append(raw_idx)
                self.targets.append(label)

    def __len__(self):
        return len(self.raw_idxs)

    def __getitem__(self, index):
        raw_idx, label = self.raw_idxs[index], self.targets[index]
        img, _ = self.base_dataset.__getitem__(raw_idx)
        if self.need_idx:
            return img, label, raw_idx
        else:
            return img, label


class CIFAR10Noisy(Dataset):
    def __init__(self, root, train, transform=None, download=True, noise_type='symmetric', noise_rate=0.2,
                 need_idx=True):
        self.base_dataset = datasets.CIFAR10(root, train, transform, download=download)
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.noisy_sample_idx = []
        self.clean_sample_idx = []
        self.need_idx = need_idx
        self.base_transform = transform
        # add label noise
        if self.noise_type == 'symmetric':
            self.uniform(noise_rate, 10)
        elif self.noise_type == 'asymmetric':
            self.flip(noise_rate, 10)

    def to_txt(self, txt_path):
        with open(txt_path, 'w') as f:
            for i in range(self.base_dataset.__len__()):
                l = self.base_dataset.targets[i]
                f.write('raw_idx:' + str(i) + ' label:' + str(l) + '\n')

    def uniform(self, noise_rate: float, num_classes: int):
        """Add symmetric noise"""

        # noise transition matrix
        ntm = noise_rate * np.full((num_classes, num_classes), 1 / (num_classes - 1))
        np.fill_diagonal(ntm, 1 - noise_rate)

        figure = plt.figure()
        axes = figure.add_subplot(111)
        # plt.matshow(ntm, cmap='viridis')
        # caxes = axes.matshow(ntm, interpolation='nearest',cmap=plt.cm.Blues)
        caxes = axes.matshow(ntm, interpolation='nearest', cmap="viridis")
        # axes.set_xticklabels([''] + alphabets)
        # axes.set_yticklabels([''] + alphabets)
        cbar = figure.colorbar(caxes, )
        plt.show()

        sample_indices = np.arange(len(self.base_dataset))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.base_dataset.targets[i]])
            if label != self.base_dataset.targets[i]:
                self.noisy_sample_idx.append(i)
            self.base_dataset.targets[i] = label

        self.noisy_sample_idx = np.array(self.noisy_sample_idx)
        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Symmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def flip(self, noise_rate: float, num_classes: int):
        """Add asymmetric noise"""

        # noise transition matrix
        ntm = np.eye(num_classes) * (1 - noise_rate)

        d = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}  # truck->automobile, bird->airplane, cat->dog, dog->cat, deer->horse
        for raw_class, new_class in d.items():
            ntm[raw_class][new_class] = noise_rate

        for i in [0, 1, 6, 7, 8]:
            ntm[i][i] = 1

        figure = plt.figure()
        axes = figure.add_subplot(111)
        # plt.matshow(ntm, cmap='viridis')

        caxes = axes.matshow(ntm, interpolation='nearest', cmap=plt.cm.Blues)
        # axes.set_xticklabels([''] + alphabets)
        # axes.set_yticklabels([''] + alphabets)
        figure.colorbar(caxes)
        plt.show()

        sample_indices = np.arange(len(self.base_dataset))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.base_dataset.targets[i]])
            if label != self.base_dataset.targets[i]:
                self.noisy_sample_idx.append(i)
            self.base_dataset.targets[i] = label

        self.noisy_sample_idx = np.array(self.noisy_sample_idx)
        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Asymmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, index):
        if self.need_idx:
            return self.base_dataset.__getitem__(index), index
        else:
            return self.base_dataset.__getitem__(index)
