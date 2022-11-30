import torch
import torchvision
import numpy as np
from torch.utils.data.dataset import TensorDataset
import math
from typing import Callable, Optional
import os
from pathlib import Path
from .partition import Partition
import warnings


class FastCIFAR10(torchvision.datasets.CIFAR10):
    """The (multiclass) Cifar10 dataset (fast version).

    The classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck').

    Arguments:
        data_root (str): the data directory
        train (bool): specify the training or test set
        transform (callable, default: None): the feature preprocessing stuff
        target_transform (callable, default: None): the label preprocessing stuff
        download (bool, default: False): whether to download the dataset
        device (str, default: 'cpu'): where to put the dataset (cpu/gpu)
        sort_by (string): sort the training dataset by 'label' or 'dirichlet'
            (default: None, don't sort)
        similarity (float):
            if sort_by == 'label', similarity is the ratio (in the range [0, 1)) of random partitioned data (default: 0, no similarity);
            if sort_by == 'dirichlet', similarity is the dirichlet distribution parameter (in the range (0, +infinity))
        num_partitions: the number of data partitions, only applicable in the case sort_by == 'dirichlet'
    """

    def __init__(
            self,
            data_root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device: str = 'cpu',
            sort_by: str = None,
            similarity: float = 0,
            num_partitions: int = 0):
        super().__init__(data_root, train, transform, target_transform, download)

        # Scale data to [0,1]
        self.data = torch.from_numpy(self.data.transpose((0, 3, 1, 2))).contiguous()
        if isinstance(self.data, torch.ByteTensor):
            self.data = self.data.float().div_(255)

        # Normalize each pixel to [-1, 1]
        self.data = self.data.sub_(0.5).div_(0.5)

        if sort_by is None:
            pass
        elif sort_by == 'label':
            if similarity == 0:
                sorted_idx = sorted(range(len(self.targets)), key=self.targets.__getitem__)
                self.targets[:] = [self.targets[i] for i in sorted_idx]
                self.data = self.data[sorted_idx, :, :, :]
            else:
                if similarity < 0 or similarity > 1:
                    raise ValueError('similarity must be in the range [0, 1], but got {:f}'.format(similarity))
                rng_state = np.random.get_state()
                np.random.seed(0)  # for reproducibility
                # Step 1. random permutation
                num_samples = len(self.targets)
                idx = np.random.permutation(num_samples)
                # Step 2. sort (1.0 - similarity) ratio of data points
                num_sorted_samples = math.ceil((1.0 - similarity) * num_samples)
                num_shuffled_samples = num_samples - num_sorted_samples
                targets_arr = np.array(self.targets)
                idx_to_sort = idx[0:num_sorted_samples]
                sorted_idx = np.argsort(targets_arr[idx_to_sort])
                idx[0:num_sorted_samples] = idx_to_sort[sorted_idx]
                # Step 3. randomly insert the indices of shuffled samples into the index array of sorted sampleds
                random_idx = np.random.permutation(num_samples)[0:num_shuffled_samples]
                new_idx = np.zeros_like(idx)
                new_idx[random_idx] = idx[num_sorted_samples:]
                new_idx[np.setdiff1d(np.arange(num_samples), random_idx)] = idx[0:num_sorted_samples]
                new_idx = new_idx.tolist()
                # Step 4. reorder the data samples
                self.targets[:] = [self.targets[i] for i in new_idx]
                self.data = self.data[new_idx, :, :, :]
                np.random.set_state(rng_state)
        elif sort_by == 'dirichlet':
            classes = np.unique(self.targets)
            targets_arr = np.array(self.targets)
            num_classes = len(classes)
            if similarity <= 0:
                raise ValueError('similarity must be in the range (0, infinity), but got {:f}'.format(similarity))
            if num_partitions <= 0:
                raise ValueError('num_partitions must be a positive integer, but got {:d}'.format(num_partitions))
            rng_state = np.random.get_state()
            np.random.seed(0)  # for reproducibility
            # Step 1. sample from the dirichlet distribution
            class_dist = np.zeros([num_partitions, num_classes])
            for i in range(num_partitions):
                class_dist[i, :] = np.random.dirichlet(similarity * np.ones(num_classes))
            # Step 2. sort the data samples by index, and shuffle samples in the same class
            indices_sorted_by_label = []
            for c in classes:
                indices_label_c = np.where(targets_arr == c)[0]
                np.random.shuffle(indices_label_c)
                indices_sorted_by_label.append(indices_label_c)
            # Step 3. partition the data samples
            # compute the number of clients held by each client
            data_len = len(self.targets)
            num_client_samples = int(data_len / num_partitions)
            client_sample_indices = [[] for _ in range(num_partitions)]
            samples_count = np.zeros(num_classes).astype(int)
            for i in range(num_partitions):
                for j in range(num_client_samples):
                    sampled_label = np.argwhere(np.random.multinomial(1, class_dist[i, :]) == 1)[0][0]
                    client_sample_indices[i].append(indices_sorted_by_label[sampled_label][samples_count[sampled_label]])
                    samples_count[sampled_label] += 1
                    if samples_count[sampled_label] == len(indices_sorted_by_label[sampled_label]):
                        class_dist[:, sampled_label] = 0
                        if samples_count.sum() == data_len:
                            assert(i == num_partitions - 1 and j == num_client_samples - 1)
                            break
                        class_dist = (class_dist / class_dist.sum(axis=1)[:, None])
            new_idx = []
            for i in range(num_partitions):
                new_idx += client_sample_indices[i]
            new_idx = np.array(new_idx)
            # Step 4. reorder the data samples
            self.targets = [self.targets[i] for i in new_idx]
            self.data = self.data[new_idx, :, :, :]
            np.random.set_state(rng_state)
        else:
            raise ValueError("the sort_by argument \"{:s}\" is invalid".format(sort_by))


        # Put both data and targets on GPU (if GPU is used) in advance
        self.targets = torch.LongTensor(self.targets)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target


class Cifar10Loader(object):
    """The multiclass or binary Cifar10 dataset loader.

    The original classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck').
    If a binary dataset is requried, we select some classes as positive, and the others negative.

    Arguments:
        num_partitions: the number of data partitions
        data_root (string): the data directory (default: './data/')
        train_batch_size: the batch size in training
        test_batch_size: the batch size in testing
        is_multiclass (bool, default: 'False'): binary or multiclass dataset
        pos_classes (list): the classes defined to be positive (only valid when is_multiclass is False)
        device (str, default: 'cpu'): where to put the dataset (cpu/gpu)
        sort_by (string): sort the training dataset by 'label' or 'norm' (default: None, don't sort)
        similarity (float number in the range [0, 1]): the ratio of random partitioned data (default: 0, no similarity), only applicable to sort_by='label'
    """

    def __init__(self, num_partitions, data_root='./data/', train_batch_size=32, test_batch_size=32, is_multiclass=False, pos_classes=[0], device='cpu', sort_by=None, similarity=0):
        if is_multiclass:
            self.pos_ratio = float('nan')
        else:
            self.pos_ratio = len(pos_classes) / 10

        self.data_root = data_root
        self.train_set = FastCIFAR10(data_root=data_root,
                                            train=True,
                                            download=True,
                                            device=device,
                                            sort_by=sort_by,
                                            similarity=similarity,
                                            num_partitions=num_partitions)
        # convert the labels to binary ones
        tmp_target_list = self.train_set.targets.tolist()
        if not is_multiclass:
            for i in range(len(tmp_target_list)):
                if any(tmp_target_list[i] == some_pos_class for some_pos_class in pos_classes):
                    self.train_set.targets[i] = 1
                else:
                    self.train_set.targets[i] = -1

        self.test_set = None

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        self.is_multiclass = is_multiclass
        self.pos_classes = pos_classes
        self.partitions = []
        data_len = len(self.train_set)
        indexes = list(range(data_len))

        sizes = [1 / num_partitions for _ in range(num_partitions)]

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def build_train_loader(self, partition_idx):
        if partition_idx >= len(self.partitions):
            warnings.warn('data partition {:d} not found, the train loader will be None'.format(partition_idx))
            return None
        train_partition = Partition(
            self.train_set, self.partitions[partition_idx])
        train_loader = torch.utils.data.DataLoader(train_partition,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True, num_workers=0)
        return train_loader

    def build_test_loader(self):
        if self.test_set is None:
            self.test_set = FastCIFAR10(data_root=self.data_root,
                                              train=False,
                                            #   train=True,
                                              download=True,
                                              device=self.device)
        # convert the labels to binary ones
        tmp_target_list = self.test_set.targets.tolist()
        if not self.is_multiclass:
            for i in range(len(tmp_target_list)):
                if any(tmp_target_list[i] == some_pos_class for some_pos_class in self.pos_classes):
                    self.test_set.targets[i] = 1
                else:
                    self.test_set.targets[i] = -1

        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=0)
        return test_loader

    def get_pos_ratio(self):
        return self.pos_ratio

    def get_feature_shape(self):
        feature_shape = [3, 32, 32]
        return feature_shape


def BuildFedCifar10Dataset(num_partitions, data_root='./data/', sort_by=None, similarity=0):
    """Constructing the federated Cifar10 dataset.

    The classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck').

    Arguments:
        num_partitions: the number of data partitions
        data_root (string): the data directory (default: './data/')
        sort_by (string): sort the training dataset by 'label' or 'dirichlet' (default: None, don't sort)
        similarity (float number in the range [0, 1]): the ratio of random partitioned data (default: 0, no similarity), only applicable to sort_by='label'
    """
    train_set = FastCIFAR10(data_root=data_root,
                                   train=True,
                                   download=True,
                                   device='cpu',
                                   sort_by=sort_by,
                                   similarity=similarity,
                                   num_partitions=num_partitions)

    test_set = FastCIFAR10(data_root=data_root,
                                          train=False,
                                        #   train=True,
                                          download=True,
                                          device='cpu')

    partitions = []
    data_len = len(train_set)
    indexes = list(range(data_len))
    sizes = [1 / num_partitions for _ in range(num_partitions)]
    for frac in sizes:
        part_len = int(frac * data_len)
        partitions.append(indexes[0:part_len])
        indexes = indexes[part_len:]

    # save data files
    data_folder_name = 'fed_cifar10_n_' + str(num_partitions) + '_sort_by_' + sort_by + '_sim_' + str(similarity)
    data_folder_path = os.path.join(data_root, data_folder_name)
    Path(data_folder_path).mkdir(parents=True, exist_ok=True)
    count = 0
    for curr_partition in partitions:
        curr_file_name = 'train_' + str(count).zfill(int(math.log(num_partitions, 10)) + 1) + '.pt'
        curr_file_path = os.path.join(data_folder_path, curr_file_name)
        torch.save({'features': train_set.data[curr_partition, :], 'labels': train_set.targets[curr_partition]}, curr_file_path)
        count += 1

    curr_file_path = os.path.join(data_folder_path, 'test.pt')
    torch.save({'features': test_set.data, 'labels': test_set.targets}, curr_file_path)

class FedCifar10Loader(object):
    """The federated Cifar10 dataset loader.

    The original classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck').
    If a binary dataset is required, we select some classes as positive, and the others negative.

    Arguments:
        num_partitions: the number of data partitions
        data_root (string): the data directory (default: './data/')
        train_batch_size: the training batch size
        test_batch_size: the test batch size
        is_multiclass (bool, default: 'False'): binary or multiclass dataset
        pos_classes (list): the classes defined to be positive (only valid when is_multiclass is False)
        sort_by (string): sort the training dataset by 'label' or 'dirichlet' (default: None, don't sort)
        similarity (float number in the range [0, 1]): the ratio of random partitioned data (default: 0, no similarity), only applicable to sort_by='label'
        device (str, default: 'cpu'): where to put the dataset (cpu/gpu)
    """

    def __init__(self, num_partitions, data_root='./data/', train_batch_size=32, test_batch_size=32, is_multiclass=False, pos_classes=[0], sort_by=None, similarity=0, device='cpu'):
        self.pos_ratio = len(pos_classes) / 10
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.is_multiclass = is_multiclass
        self.pos_classes = pos_classes
        self.num_partitions = num_partitions
        self.device = device
        data_folder_name = 'fed_cifar10_n_' + str(num_partitions) + '_sort_by_' + sort_by + '_sim_' + str(similarity)
        self.data_folder_path = os.path.join(data_root, data_folder_name)

    def build_train_loader(self, partition_idx):
        partition_file_name = 'train_' + str(partition_idx).zfill(int(math.log(self.num_partitions, 10)) + 1) + '.pt'
        partition_file_path = os.path.join(self.data_folder_path, partition_file_name)
        if not Path(partition_file_path).is_file():
            warnings.warn('data partition {:d} not found, the train loader will be None'.format(partition_idx))
            return None
        data_dict = torch.load(partition_file_path)
        features = data_dict['features']
        labels = data_dict['labels']
        # convert the labels to binary ones
        tmp_label_list = labels.tolist()
        if not self.is_multiclass:
            for i in range(len(tmp_label_list)):
                if any(tmp_label_list[i] == some_pos_class for some_pos_class in self.pos_classes):
                    labels[i] = 1
                else:
                    labels[i] = -1
        # Put both data and targets on GPU in advance
        features, labels = features.to(self.device), labels.to(self.device)
        train_set = TensorDataset(features, labels)
        train_batch_size = min(len(labels), self.train_batch_size)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=train_batch_size,
                                                   shuffle=True, num_workers=0)
        return train_loader

    def build_test_loader(self):
        partition_file_name = 'test.pt'
        partition_file_path = os.path.join(self.data_folder_path, partition_file_name)
        data_dict = torch.load(partition_file_path)
        features = data_dict['features']
        labels = data_dict['labels']
        # convert the labels to binary ones
        tmp_label_list = labels.tolist()
        if not self.is_multiclass:
            for i in range(len(tmp_label_list)):
                if any(tmp_label_list[i] == some_pos_class for some_pos_class in self.pos_classes):
                    labels[i] = 1
                else:
                    labels[i] = -1
        # Put both data and targets on GPU in advance
        features, labels = features.to(self.device), labels.to(self.device)
        test_set = TensorDataset(features, labels)
        test_batch_size = min(len(labels), self.test_batch_size)
        test_loader = torch.utils.data.DataLoader(test_set,
                                           batch_size=test_batch_size,
                                           shuffle=True, num_workers=0)
        return test_loader

    def get_pos_ratio(self):
        return self.pos_ratio

    def get_feature_shape(self):
        feature_shape = [3, 32, 32]
        return feature_shape
