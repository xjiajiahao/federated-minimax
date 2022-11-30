import torch
import torchvision
import numpy as np
from .partition import Partition
import torchvision.transforms as transforms
from random import Random
from nltk.corpus import wordnet as wn
import os
import math


class CelebALoader(object):
    """The CelebA dataset loader.
    
    Arguments:
        num_partitions: the number of data partitions
        data_root (string): the data directory (default: './data/')
        train_batch_size: the batch size in training
        test_batch_size: the batch size in testing
        device (str, default: 'cpu'): where to put the dataset (cpu/gpu)
        sort_by (string): sort the training dataset by 'label' or 'norm' (default: None, don't sort)
        similarity (float number in the range [0, 1]): the ratio of random partitioned data (default: 0, no similarity), only applicable to sort_by='label'
    """

    def __init__(self, num_partitions, data_root='./data/', train_batch_size=32, test_batch_size=32, device='cpu', sort_by=None, similarity=0):
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.train_set = torchvision.datasets.CelebA(root=data_root,
                                            split='train',
                                            target_type='identity',
                                            transform=self.transform)

        self.test_set = None

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        self.partitions = []
        data_len = len(self.train_set)
        indices = list(range(data_len))

        if sort_by is None:
            pass
        elif sort_by == 'label':
            targets = torch.squeeze(self.train_set.identity).tolist() 
            if similarity == 0:
                sorted_idx = sorted(range(len(targets)), key=targets.__getitem__)
                indices[:] = [indices[i] for i in sorted_idx]
            else:
                if similarity < 0 or similarity > 1:
                    raise ValueError('similarity must be in the range [0, 1], but got {:f}'.format(similarity))
                rng_np = np.random.default_rng(0)  # for reproducibility
                # Step 1. random permutation
                num_samples = len(self.train_set.identity)
                idx = rng_np.permutation(num_samples)
                # Step 2. sort (1.0 - similarity) ratio of data points
                num_sorted_samples = math.ceil((1.0 - similarity) * num_samples)
                num_shuffled_samples = num_samples - num_sorted_samples
                # remaining_classes = np.unique(targets)
                targets_arr = np.array(targets)
                idx_to_sort = idx[0:num_sorted_samples]
                sorted_idx = np.argsort(targets_arr[idx_to_sort])
                idx[0:num_sorted_samples] = idx_to_sort[sorted_idx]
                # Step 3. randomly insert the indices of shuffled samples into the index array of sorted sampleds
                random_idx = rng_np.permutation(num_samples)[0:num_shuffled_samples]
                new_idx = np.zeros_like(idx)
                new_idx[random_idx] = idx[num_sorted_samples:]
                new_idx[np.setdiff1d(np.arange(num_samples), random_idx)] = idx[0:num_sorted_samples]
                new_idx = new_idx.tolist()
                # Step 4. reorder the data samples
                indices = new_idx
        elif sort_by == 'dirichlet':
            # targets = [self.targets[i] for i in indices]
            targets = torch.squeeze(self.train_set.identity).tolist() 
            remaining_classes = np.unique(targets)
            targets_arr = np.array(targets)
            num_remaining_classes = len(remaining_classes)
            if similarity <= 0:
                raise ValueError(
                    'similarity must be in the range (0, infinity), but got {:f}'.format(similarity))
            if num_partitions <= 0:
                raise ValueError(
                    'num_partitions must be a positive integer, but got {:d}'.format(num_partitions))
            rng_np = np.random.default_rng(0)  # for reproducibility
            # Step 1. sample from the dirichlet distribution
            class_dist = np.zeros([num_partitions, num_remaining_classes])
            for i in range(num_partitions):
                class_dist[i, :] = rng_np.dirichlet(
                    similarity * np.ones(num_remaining_classes))
            # Step 2. sort the data samples by index, and shuffle samples in the same class
            indices_sorted_by_label = []
            for c in remaining_classes:
                indices_label_c = np.where(targets_arr == c)[0]
                np.random.shuffle(indices_label_c)
                indices_sorted_by_label.append(indices_label_c)
            # Step 3. partition the data samples
            # compute the number of clients held by each client
            num_client_samples = int(data_len / num_partitions)
            client_sample_indices = [[] for _ in range(num_partitions)]
            samples_count = np.zeros(num_remaining_classes).astype(int)
            for i in range(num_partitions):
                for j in range(num_client_samples):
                    sampled_label = np.argwhere(
                        np.random.multinomial(1, class_dist[i, :]) == 1)[0][0]
                    client_sample_indices[i].append(
                        indices_sorted_by_label[sampled_label][samples_count[sampled_label]])
                    samples_count[sampled_label] += 1
                    if samples_count[sampled_label] == len(indices_sorted_by_label[sampled_label]):
                        class_dist[:, sampled_label] = 0
                        if samples_count.sum() == data_len:
                            assert(i == num_partitions - 1 and j ==
                                   num_client_samples - 1)
                            break
                        class_dist = (
                            class_dist / class_dist.sum(axis=1)[:, None])
            new_idx = []
            for i in range(num_partitions):
                new_idx += client_sample_indices[i]
            indices = new_idx
        else:
            raise ValueError(
                "the sort_by argument \"{:s}\" is invalid".format(sort_by))

        sizes = [1 / num_partitions for _ in range(num_partitions)]

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indices[0:part_len])
            indices = indices[part_len:]

    def build_train_loader(self, partition_idx):
        if partition_idx < 0 or partition_idx >= len(self.partitions):
            raise ValueError("invalid partion index for train loader")
        local_partition = Partition(
            self.train_set, self.partitions[partition_idx])
        train_loader = torch.utils.data.DataLoader(local_partition,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True, num_workers=0)
        return train_loader

    def build_test_loader(self):
        if self.test_set is None:
            self.test_set = torchvision.datasets.CelebA(root=self.data_root,
                                                split='test',
                                                target_type='identity',
                                                transform=self.transform)

        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=0)
        return test_loader

    def get_feature_shape(self):
        feature_shape = [3, 64, 64]
        return feature_shape
