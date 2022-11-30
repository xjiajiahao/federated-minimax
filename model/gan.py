import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
# from torchvision.models.inception import inception_v3
from .inception import InceptionV3
from .fid_score import calculate_fid
import math
import numpy as np
from scipy.stats import entropy

import errno
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('agg')
from datetime import datetime
import time
import warnings


def show_images(images, save_path="sample.png"):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    if (images.shape[1] == 1):
        plt.rcParams['image.cmap'] = 'gray'
        # images reshape to (batch_size, D)
        images = np.reshape(images, [images.shape[0], -1])
        sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    # fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if (len(img.shape) == 1):
            plt.imshow(img.reshape([sqrtimg, sqrtimg]))
        else:
            # plt.imshow(np.rollaxis(img, 0, 3).clip(0.0, 1.0))
            # print normalized figure
            min_pixel_val = np.min(img)
            max_pixel_val = np.max(img)
            plt.imshow(np.rollaxis((img - min_pixel_val) / (max_pixel_val - min_pixel_val), 0, 3))
    try:
        os.environ["DISPLAY"]
    except KeyError:
        # print("no display")
        plt.savefig(save_path, )  # save
    else:
        plt.savefig(save_path)  # save
        # plt.show()


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


class GAN:
    """ The GAN model

    Arguments:
        g_net: the generator model
        d_net: the discriminator model
        feature_shape (int): the shape of the sample feature
        latent_vector_len (int): the length of the latent vector
        device (str): the device to use (cpu or gpu)
        eval_num_samples (int): the number of generated samples used in evaluation
        eval_net_name (str): the network to compute the inception score ('resnet18_mnist' for MNIST, 'resnet18_fashion_mnist' for FashionMNIST, and 'inception_v3' for CIFAR-10, Imagenet, ...)
        eval_pretrained_dir (str): the directory that stores the pretrained eval_net
        eval_batch_size (int): the batch size for feeding into the network in evaluation
        eval_resize (bool): whether to resize the image to 299-by-299 (only applicable if eval_net_name='inception_v3')
        eval_splits (int): number of splits for computing the variance of the Inception score
        eval_device (str): the device for evaluating Inception score (cpu or gpu) 
    """

    class _Logger:
        def __init__(self, output_file_tag):
            self.output_file_tag = output_file_tag
            self.overhead_time = 0.0
            self.t_start = time.time()
            self.t_elapsed = None
            self.running_time = 0.0
            self.count_received_floats = 0

            # each row of the log records [#rounds, elapsed time, #received floats, robust test loss, robust test accuracy]
            with open(self.output_file_tag + '.csv', 'w') as f:
                header = '#round, time (s), #floats received by the server, robust loss, mean of the inception score, std of the inception score\n'
                f.write(header)

        def stop_timing(self):
            self.t_elapsed = time.time() - self.t_start
            self.running_time = max(self.t_elapsed - self.overhead_time, 0.0)

        def continue_timing(self):
            self.overhead_time += time.time() - self.t_start - self.t_elapsed

        def log(self, curr_round, curr_received_floats, score_mean, score_std, images_to_save):
            self.count_received_floats += curr_received_floats
            data_to_log = np.array(
                [[curr_round, self.running_time, self.count_received_floats, score_mean, score_std]])
            print('[{:s} round {:d}] elapsed time: {:.1f}, #received floats: {:.4e}, inception score: {:f}{:s}{:f}'.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), curr_round, self.running_time, self.count_received_floats, score_mean, u"\u00B1", score_std))
            # write to the output file
            with open(self.output_file_tag + '.csv', 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')
            # output fake images
            show_images(
                images_to_save, save_path=self.output_file_tag + '.png')
            show_images(
                images_to_save, save_path=self.output_file_tag + '_' + str(curr_round).zfill(4) + '.png')


    def __init__(self, g_net, d_net, feature_shape, latent_vector_len, device, eval_num_samples=10000, eval_net_name='resnet18_mnist', eval_pretrained_dir='./data/', eval_batch_size=100, eval_resize=False, eval_splits=10, eval_device='cpu'):
        self.g_net = g_net
        self.d_net = d_net
        self.feature_shape = feature_shape
        self.latent_vector_len = latent_vector_len

        self.eval_net_module = None
        self.eval_num_samples = eval_num_samples
        self.eval_net_name = eval_net_name
        self.eval_batch_size = eval_batch_size
        self.eval_pretrained_dir = eval_pretrained_dir
        self.eval_resize = eval_resize
        self.eval_splits = eval_splits
        if device == 'gpu':
            device = 'cpu'
        elif device == 'cpu':
            eval_device = 'cpu'
        self.device = device
        self.eval_device = eval_device

        self.num_parameters = sum(p.numel() for p in self.g_net.parameters(
        ) if p.requires_grad) + sum(p.numel() for p in self.d_net.parameters() if p.requires_grad)
        self.random_noise_buffer = None
        # self.fixed_noise_buffer = torch.zeros(self.eval_num_samples, latent_vector_len, 1, 1, device=device).normal_(0, 1)
        self.fixed_noise_buffer = torch.zeros(
            self.eval_num_samples, latent_vector_len, device=device)
        torch.rand([self.eval_num_samples, latent_vector_len],
                   out=self.fixed_noise_buffer)
        self.fixed_noise_buffer.multiply_(2).add_(-1)
        self.imgs_buffer = torch.zeros(
            [self.eval_num_samples] + feature_shape)

    def logger_init(self, output_file_tag):
        self.logger = self._Logger(output_file_tag)

    def evaluate_and_log(self, curr_round, curr_received_floats, test_loader):
        """Log the progress and return a flag indicating whether to exit.

        Args:
            curr_round: the current round
            curr_received_floats: the number of doubles received by the server in the current round
            test_loader: the test loader for evaluating the performance

        Returns:
            too_bad_flag (bool): returns True if the performance is too bad that we should kill the program to save time for tunning, otherwise return False.
        """

        if self.logger is None:
            warnings.warn(
                'Please call RobustNN.logger_init() before calling RobustNN.log(), otherwise nothing will be logged')
            return False
        self.logger.stop_timing()
        score_mean, score_std = self.evaluate_metric(test_loader)
        imgs_numpy = self.imgs_buffer[0:self.eval_batch_size].numpy()
        self.logger.log(curr_round, curr_received_floats,
                        score_mean, score_std, imgs_numpy)
        if curr_round >= 50 and self.abort_condition(score_mean):
            too_bad_flag = True
        else:
            too_bad_flag = False
        self.logger.continue_timing()
        return too_bad_flag

    def evaluate_metric(self, test_loader):
        """Evaluate the Inception score for generated images.

        Args: 
            test_loader: the data loader of the test dataset (unused)

        Returns:
            score: the Inception score value
        """
        # Step 0. load the inception model if self.eval_net_module is None
        if self.eval_net_module is None:
            if self.eval_net_name == 'inception_v3':
                opt_model_file_name = os.path.join(
                    self.eval_pretrained_dir, 'pt_inception-2015-12-05-6726825d.pth')
                if Path(opt_model_file_name).is_file():
                    self.eval_net_module = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], model_file_name=opt_model_file_name)
                else:
                    self.eval_net_module = InceptionV3(InceptionV3.BLOCK_INDEX_BY_DIM[2048])
            elif self.eval_net_name == 'resnet18_mnist' or self.eval_net_name == 'resnet18_fashion_mnist':
                self.eval_net_module = resnet18(
                    num_classes=10)  # MNIST has 10 classes
                self.eval_net_module.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(
                    2, 2), padding=(3, 3), bias=False)  # MNIST has 1 channel
                if self.eval_net_name == 'resnet18_mnist':
                    opt_model_file_name = os.path.join(
                        self.eval_pretrained_dir, 'resnet18_mnist_opt.pth')
                else:
                    opt_model_file_name = os.path.join(
                        self.eval_pretrained_dir, 'resnet18_fashion_mnist_opt.pth')
                if not Path(opt_model_file_name).is_file():
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), opt_model_file_name)
                self.eval_net_module.load_state_dict(torch.load(
                    opt_model_file_name))   # to restore the model
            else:
                raise ValueError(
                    'network must be inception_v3 or resnet18, but got {:s}'.format(self.eval_net_name))
            self.eval_net_module.to(self.eval_device)

        # Step 1. generate images
        self.eval_net_module.eval()
        num_iters = math.ceil(self.eval_num_samples / self.eval_batch_size)
        with torch.no_grad():
            curr_start_idx = 0
            for i in range(num_iters):
                curr_batch_size = min(
                    self.eval_batch_size, self.eval_num_samples - curr_start_idx)

                curr_noise = self.fixed_noise_buffer.narrow(
                    0, curr_start_idx, curr_batch_size)
                curr_imgs = self.g_net(curr_noise)
                self.imgs_buffer[curr_start_idx:curr_start_idx +
                                 curr_batch_size] = curr_imgs.data.cpu()
                curr_start_idx += curr_batch_size

        if self.eval_net_name == 'inception_v3':  # we use FID score
            score_mean = calculate_fid(self.imgs_buffer, test_loader, self.eval_net_module, batch_size=self.eval_batch_size, device=self.eval_device)
            score_std = 0.0
        elif self.eval_net_name == 'resnet18_mnist' or self.eval_net_name == 'resnet18_fashion_mnist':
            # Step 2. get predictions
            # up = nn.Upsample(size=(299, 299), mode='bilinear',
            #                  align_corners=False).to(self.device)
            up = nn.Upsample(size=(299, 299), mode='bilinear',
                             align_corners=False)

            def get_pred(x):
                if self.eval_resize:
                    x = up(x).to(self.eval_device)
                x = self.eval_net_module(x)
                return F.softmax(x, dim=1).data.cpu().numpy()

            preds = np.zeros((self.eval_num_samples, 10))

            with torch.no_grad():
                curr_start_idx = 0
                for i in range(num_iters):
                    curr_batch_size = min(
                        self.eval_batch_size, self.eval_num_samples - curr_start_idx)

                    curr_imgs = self.imgs_buffer[curr_start_idx:curr_start_idx+curr_batch_size]
                    preds[curr_start_idx:curr_start_idx +
                          curr_batch_size] = get_pred(curr_imgs)
                    curr_start_idx += curr_batch_size

            # Step 3. compute the mean KL divergence
            split_scores = []
            for k in range(self.eval_splits):
                part = preds[k * (self.eval_num_samples // self.eval_num_samples): (k+1) * (self.eval_num_samples // self.eval_splits), :]
                py = np.mean(part, axis=0)
                scores = []
                for i in range(part.shape[0]):
                    pyx = part[i, :]
                    scores.append(entropy(pyx, py))
                split_scores.append(np.exp(np.mean(scores)))
            score_mean = np.mean(split_scores)
            score_std = np.std(split_scores)
        return score_mean, score_std

    def zero_grad(self, set_to_none=False):
        """Clear the gradient of the model."""
        self.g_net.zero_grad(set_to_none)
        self.d_net.zero_grad(set_to_none)

    def forward(self, features, labels, is_snapshot=False):
        """Forward propagation of the GAN.

        Compute the the loss value of the GAN model.

        Args:
            features: the features of data samples 
            labels: the labels of data samples
            is_snapshot: whether to draw new random noise or use the previous one 

        Returns: return the loss
        """
        batch_size = len(labels)
        if self.random_noise_buffer is None or self.random_noise_buffer.shape[0] < batch_size:
            self.random_noise_buffer = torch.zeros(
                batch_size, self.latent_vector_len, device=self.device)
            torch.rand([batch_size, self.latent_vector_len],
                       out=self.random_noise_buffer)
            self.random_noise_buffer.multiply_(2).add_(-1)
            tmp_noise = self.random_noise_buffer
        else:
            tmp_noise = self.random_noise_buffer.narrow(0, 0, batch_size)
            if not is_snapshot:
                torch.rand([batch_size, self.latent_vector_len], out=tmp_noise)
                tmp_noise.multiply_(2).add_(-1)

        real_score = self.d_net(features)
        fake_samples = self.g_net(tmp_noise).detach()
        fake_score = self.d_net(fake_samples)
        d_loss = - bce_loss(real_score, torch.ones_like(real_score, device=self.device)) - bce_loss(fake_score, torch.zeros_like(
            fake_score, device=self.device))

        # resample
        torch.rand([batch_size, self.latent_vector_len], out=tmp_noise)
        tmp_noise.multiply_(2).add_(-1)

        fake_samples = self.g_net(tmp_noise)
        for dp in self.d_net.parameters():  # excluding d
            dp.requires_grad = False
        fake_score = self.d_net(fake_samples)
        g_loss = bce_loss(fake_score, torch.ones_like(
            fake_score, device=self.device))
        for dp in self.d_net.parameters():  # excluding d
            dp.requires_grad = True
        loss = d_loss + g_loss
        return loss

    def exchange(self, communicator, weight=None):
        """Exchange and average the model parameters.

        Arguments: 
            communicator: the communication handler
        """
        if weight is None:
            world_size = float(communicator.get_world_size())
            with torch.no_grad():
                for param in self.g_net.parameters():
                    communicator.all_reduce(param.data)
                    param.data /= world_size

                for param in self.d_net.parameters():
                    communicator.all_reduce(param.data)
                    param.data /= world_size
        else:
            with torch.no_grad():
                for param in self.g_net.parameters():
                    param.data *= weight
                    communicator.all_reduce(param.data)

                for param in self.d_net.parameters():
                    param.data *= weight
                    communicator.all_reduce(param.data)

    def broadcast(self, client_subset, communicator):
        """Broadcast the global model parameters.

        Arguments: 
            client_subset: the list of sampled clients
            communicator: the communication handler
        """
        with torch.no_grad():
            for param in self.g_net.parameters():
                communicator.broadcast(param.data, client_subset)

            for param in self.d_net.parameters():
                communicator.broadcast(param.data, client_subset)

    def reduce(self, client_subset, communicator, average_flag=True):
        """Aggregate the local models.

        Arguments: 
            client_subset: the list of sampled clients
            communicator: the communication handler
            average_flag: take average (True) or sum (False)
        """
        num_sampled_clients = len(client_subset)
        with torch.no_grad():
            for param in self.g_net.parameters():
                communicator.reduce(param.data, client_subset)
                if communicator.is_root() and average_flag:
                    param.data /= num_sampled_clients

            for param in self.d_net.parameters():
                communicator.reduce(param.data, client_subset)
                if communicator.is_root() and average_flag:
                    param.data /= num_sampled_clients

    def zero_param(self):
        with torch.no_grad():
            for param in self.g_net.parameters():
                param.data.zero_()

            for param in self.d_net.parameters():
                param.data.zero_()

    def copy_(self, src):
        with torch.no_grad():
            for param, src_param in zip(self.g_net.parameters(), src.g_net.parameters()):
                param.data.copy_(src_param.data)

            for param, src_param in zip(self.d_net.parameters(), src.d_net.parameters()):
                param.data.copy_(src_param.data)

        self.num_parameters = src.num_parameters
        self.device = src.device

    def add_(self, src, alpha=1):
        with torch.no_grad():
            for param, src_param in zip(self.g_net.parameters(), src.g_net.parameters()):
                param.data.add_(src_param.data, alpha=alpha)

            for param, src_param in zip(self.d_net.parameters(), src.d_net.parameters()):
                param.data.add_(src_param.data, alpha=alpha)

    def multiply_(self, alpha):
        with torch.no_grad():
            for param in self.g_net.parameters():
                param.data.multiply_(alpha)

            for param in self.d_net.parameters():
                param.data.multiply_(alpha)

    def get_variable_lists(self):
        """Get two lists of primal and dual variables, resp.

        Returns: 
            primal_list: the list of primal variables
            dual_list: the list of dual variables
        """
        primal_list = list(self.g_net.parameters())
        dual_list = list(self.d_net.parameters())
        return primal_list, dual_list

    def get_num_parameters(self):
        return self.num_parameters

    def projection(self):
        """Projection operator interface which is not used for the AUC task.

        Returns:
            projection_handle: None
        """
        return lambda: None

    def abort_condition(self, utility):
        """Determine whether the utility value is too bad.

        Returns:
           ret (bool type): True if the utility is too bad, otherwise False 
        """
        ret = False
        if self.eval_net_name == 'inception_v3':  # fid score
            # if math.isnan(utility) or math.isinf(utility):
            # if math.isnan(utility) or math.isinf(utility) or utility > 400:
            if math.isnan(utility) or math.isinf(utility) or utility > 450:
            # if math.isnan(utility) or math.isinf(utility) or utility > 420:
                ret = True
        else:  # Inception score
            # if math.isnan(utility) or math.isinf(utility) or utility < 2.4:
            if math.isnan(utility) or math.isinf(utility) or utility < 3.0:
            # if math.isnan(utility) or math.isinf(utility) or utility < 1.0:
            # if math.isnan(utility) or math.isinf(utility):
                ret = True
        return ret

    def compute_net_norm(self, is_g_net=True):
        """Compute the norm of the network model
        """
        norm_value = 0.0
        with torch.no_grad():
            if is_g_net:
                for param in self.g_net.parameters():
                    norm_value += torch.sum(param.data**2).item()
            else:
                for param in self.d_net.parameters():
                    norm_value += torch.sum(param.data**2).item()
        norm_value = norm_value**0.5
        return norm_value

    def to(self, device):
        self.g_net.to(device)
        self.d_net.to(device)
        if self.random_noise_buffer is not None:
            tmp_buffer = self.random_noise_buffer.to(device)
            del self.random_noise_buffer
            self.random_noise_buffer = tmp_buffer
        self.device = device
