import torch
import torchvision
import numpy as np
from .partition import Partition
import torchvision.transforms as transforms
from random import Random
from nltk.corpus import wordnet as wn
import os
import math


class ImagenetLoader(object):
    """The binary Imagenet dataset loader.

    The original dataset consists of 1000 classes of images.
    We view the first to spli_index classes as negative, and the rest positive. 

    Arguments:
        num_partitions: the number of data partitions
        data_root (string): the data directory
        train_batch_size: the batch size in training
        test_batch_size: the batch size in testing
        is_multiclass (bool, default: 'False'): binary or multiclass dataset
        neg_keep_ratio: the ratio of negative samples we want to keep,
        test_ratio: the ratio of test samples
        sort_by (string): sort the training dataset by 'label' or 'dirichlet'
            (default: None, don't sort)
        similarity (float):
            if sort_by == 'label', similarity is the ratio (in the range [0, 1)) of random partitioned data (default: 0, no similarity);
            if sort_by == 'dirichlet', similarity is the dirichlet distribution parameter (in the range (0, +infinity))
    """

    def __init__(
            self,
            num_partitions: int,
            data_root: str = './data/',
            train_batch_size: int = 32,
            test_batch_size: int = 32,
            is_multiclass: bool = False,
            neg_keep_ratio: float = 0.3,  # only keep 30% negative samples
            test_ratio: float = 0.1,  # 10% test samples
            sort_by: str = None,
            similarity: float = 0):
        # same seed on all processes, thus the partitions are the same
        self.data_root = os.path.join(data_root, 'imagenet/')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])  # preprocessing data

        self.train_set = torchvision.datasets.ImageNet(self.data_root,
                                                       split='train',
                                                       transform=transform,
                                                       download=None)

        # Divide the data (assume each machine has access to all data)
        if is_multiclass:
            self.neg_keep_ratio = float('nan')
        else:
            # define positive and negative classes
            num_classes = len(self.train_set.classes)
            neg_classes = set()
            binary_classes = np.ones(num_classes)
            for i in range(num_classes):
                wnid = self.train_set.wnids[i]
                ss = [wn.synset_from_pos_and_offset(
                    'n', int(wnid[1:len(wnid)]))]
                while (len(ss) > 0):
                    if (ss[0].offset() == 4258):  # living thing
                        binary_classes[i] = -1
                        neg_classes.add(i)
                        break
                    ss = ss[0].hypernyms()
            self.neg_classes = neg_classes

            self.neg_keep_ratio = neg_keep_ratio
            self.train_set = torchvision.datasets.ImageNet(self.data_root,
                                                           split='train',
                                                           transform=transform,
                                                           download=None,
                                                           target_transform=self.label_transform)

        self.test_set = None
        self.test_ratio = test_ratio
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.is_multiclass = is_multiclass
        self.transform = transform
        self.partitions = []
        data_len = len(self.train_set)

        if is_multiclass:
            indices = list(range(0, data_len))
        else:
            start_indices = np.zeros(num_classes).astype(int)
            end_indices = np.zeros(num_classes).astype(int)
            prev_label = None
            for i in range(data_len):
                curr_label = self.train_set.targets[i]
                if prev_label is None or curr_label != prev_label:
                    start_indices[curr_label] = i
                end_indices[curr_label] = i
                prev_label = curr_label
            neg_indices = []
            pos_indices = []
            for i in range(num_classes):
                if (binary_classes[i] == 1):  # positive
                    pos_indices += list(
                        range(start_indices[i], end_indices[i] + 1))
                else:
                    neg_indices += list(
                        range(start_indices[i], end_indices[i] + 1))

            rng = Random()
            rng.seed(0)  # for reproducibility

            rng.shuffle(neg_indices)

            neg_num = int(len(neg_indices)*neg_keep_ratio)
            neg_indices = neg_indices[0:neg_num]  # drop some negative samples
            neg_indices.sort()
            indices = neg_indices + pos_indices
            data_len = len(indices)
            rng_np = np.random.default_rng(0)  # for reproducibility
            np.random.shuffle(indices)
            test_indices = indices[data_len - int(self.test_ratio * data_len): data_len]
            indices= indices[0 : data_len - int(self.test_ratio * data_len)]
            indices.sort()
            data_len = len(indices)




        if is_multiclass:
            self.pos_ratio = float('nan')
        else:
            self.pos_ratio = len(pos_indices) / \
                (len(neg_indices) + len(pos_indices))

        if sort_by is None:
            pass
        elif sort_by == 'label':
            if similarity != 0:
                if similarity < 0 or similarity > 1:
                    raise ValueError('similarity must be in the range [0, 1], but got {:f}'.format(similarity))
                rng_np = np.random.default_rng(0)  # for reproducibility
                # Step 1. random permutation
                num_samples = len(self.train_set.targets)
                idx = rng_np.permutation(num_samples)
                # Step 2. sort (1.0 - similarity) ratio of data points
                num_sorted_samples = math.ceil((1.0 - similarity) * num_samples)
                num_shuffled_samples = num_samples - num_sorted_samples
                targets = [self.targets[i] for i in indices]
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
            targets = [self.targets[i] for i in indices]
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

        sizes = [self.test_ratio] + [(1 - self.test_ratio) / num_partitions for _ in range(num_partitions)]
        indices = test_indices + indices
        data_len = len(indices)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indices[0:part_len])
            indices = indices[part_len:]

    def label_transform(self, label):
        # redefine the labels
        if label in self.neg_classes:  # @NOTE living things are the negative samples, and the rest are positive
            new_label = -1
        else:
            new_label = 1
        return new_label

    def build_train_loader(self, partition_idx):
        if partition_idx < 0 or partition_idx >= len(self.partitions):
            raise ValueError("invalid partion index for train loader")
        local_partition = Partition(
            self.train_set, self.partitions[partition_idx + 1])
        train_loader = torch.utils.data.DataLoader(local_partition,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True, num_workers=2)
        return train_loader

    def build_test_loader(self):
        local_partition = Partition(
            self.train_set, self.partitions[0])
        test_loader = torch.utils.data.DataLoader(local_partition,
                                                   batch_size=self.test_batch_size,
                                                   shuffle=False, num_workers=2)
        return test_loader

    def get_pos_ratio(self):
        return self.pos_ratio

    def get_feature_shape(self):
        feature_shape = [3, 128, 128]
        return feature_shape
