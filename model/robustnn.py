import torch
import torch.nn.functional as F
import math
import numpy as np
from datetime import datetime
import time
import warnings


class RobustNN:
    """ The Robust Neural Network model

    Arguments:
        net: the deep network model
        feature_shape: the shape of the sample feature
        regularizer_coef: the coefficient of the l2 regularizer on the dual
        device: the device to use (cpu or gpu)
        num_epochs_eval (int, default: 20): the number of epochs in evaluating the robust loss
        step_size_eval (float, default: 1.0): the step size of GD (incremental GD) in evaluating the robust loss
        is_deterministic_eval (bool, default: True): whether to use batch GD or incremental GD in evaluating the robust loss
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
                header = '#round, time (s), #floats received by the server, robust loss, robust accuracy\n'
                f.write(header)

        def stop_timing(self):
            self.t_elapsed = time.time() - self.t_start
            self.running_time = max(self.t_elapsed - self.overhead_time, 0.0)

        def continue_timing(self):
            self.overhead_time += time.time() - self.t_start - self.t_elapsed

        def log(self, curr_round, curr_received_floats, robust_loss, robust_accuracy):
            self.count_received_floats += curr_received_floats
            data_to_log = np.array(
                [[curr_round, self.running_time, self.count_received_floats, robust_loss, robust_accuracy]])
            print('[{:s} round {:d}] elapsed time: {:.1f}, #received floats: {:.4e}, loss: {:f}, accuracy: {:f}'.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), curr_round, self.running_time, self.count_received_floats, robust_loss, robust_accuracy))

            # write to the output file
            with open(self.output_file_tag + '.csv', 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')

    def __init__(self, net, feature_shape, regularizer_coef, device, num_epochs_eval=20, step_size_eval=1, is_deterministic_eval=True):
        self.net = net
        self.device = device
        self.regularizer_coef = regularizer_coef
        self.perturbation = torch.zeros(feature_shape, dtype=torch.float32,
                                        device=device, requires_grad=True)
        self.num_parameters = self.perturbation.numel() + \
            sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.num_epochs_eval = num_epochs_eval
        self.step_size_eval = step_size_eval
        self.is_deterministic_eval = is_deterministic_eval
        self.logger = None

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
        robust_loss, robust_accuracy = self.evaluate_metric(test_loader)
        self.logger.log(curr_round, curr_received_floats,
                        robust_loss, robust_accuracy)
        if curr_round >= 50 and self.abort_condition(robust_loss):
            too_bad_flag = True
        else:
            too_bad_flag = False
        self.logger.continue_timing()
        return too_bad_flag

    def evaluate_metric(self, test_loader):
        """Evaluate the robust loss vaule.

        Args: 
            test_loader: the data loader of the test dataset

        Returns:
            (robust_loss_value, robust_accuracy): the robust loss value and accuracy
        """
        def set_requires_grad(module, val):
            for p in module.parameters():
                p.requires_grad = val

        tmp_perturbation = self.perturbation.detach().clone()
        if torch.sum(tmp_perturbation**2) > 10.0:
            tmp_perturbation.data.zero_()
        tmp_perturbation.requires_grad_()
        set_requires_grad(self.net, False)

        # take a few gradient ascent steps to compute the robust loss
        if self.is_deterministic_eval:
            for i in range(self.num_epochs_eval):
                loss = 0
                # clear the gradient of tmp_perturbation
                if tmp_perturbation.grad is not None:
                    if tmp_perturbation.grad.grad_fn is not None:
                        tmp_perturbation.grad.detach_()
                    else:
                        tmp_perturbation.grad.requires_grad_(False)
                    tmp_perturbation.grad.data.zero_()
                # run multiple epochs to optimize tmp_perturbation
                for i, (features, labels) in enumerate(test_loader):
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    mini_batch_size = len(labels)
                    outputs = self.net(features + tmp_perturbation)
                    loss += mini_batch_size * F.cross_entropy(outputs, labels)
                data_size = len(test_loader.dataset)
                loss /= data_size
                loss -= self.regularizer_coef * torch.sum(tmp_perturbation**2)
                loss.backward()
                tmp_perturbation.data.add_(
                    tmp_perturbation.grad.data, alpha=self.step_size_eval)
        else:  # perform incremental GD instead of GD
            for i in range(self.num_epochs_eval):
                # clear the gradient of tmp_perturbation
                if tmp_perturbation.grad is not None:
                    if tmp_perturbation.grad.grad_fn is not None:
                        tmp_perturbation.grad.detach_()
                    else:
                        tmp_perturbation.grad.requires_grad_(False)
                    tmp_perturbation.grad.data.zero_()
                # run multiple epochs to optimize tmp_perturbation
                for i, (features, labels) in enumerate(test_loader):
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.net(features + tmp_perturbation)
                    loss = F.cross_entropy(outputs, labels)
                    loss -= self.regularizer_coef * \
                        torch.sum(tmp_perturbation**2)
                    loss.backward()
                    tmp_perturbation.data.add_(
                        tmp_perturbation.grad.data, alpha=self.step_size_eval)

        # evaluate the robust loss with the optimal perturbation
        robust_loss = 0.0
        count_correct_predictions = 0
        count_samples = 0
        with torch.no_grad():
            for _, data in enumerate(test_loader, 0):
                features, labels = data
                features = features.to(self.device)
                labels = labels.to(self.device)
                mini_batch_size = len(labels)
                outputs = self.net(features + tmp_perturbation)
                robust_loss += mini_batch_size * \
                    F.cross_entropy(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                count_samples += labels.size(0)
                count_correct_predictions += (predicted == labels).sum().item()
            robust_loss /= count_samples
            robust_loss -= self.regularizer_coef * \
                torch.sum(tmp_perturbation**2)
            robust_accuracy = count_correct_predictions / count_samples
        set_requires_grad(self.net, True)
        return robust_loss.item(), robust_accuracy

    def zero_grad(self, set_to_none=False):
        """Clear the gradient of the model."""
        self.net.zero_grad(set_to_none)
        if self.perturbation.grad is not None:
            if self.perturbation.grad.grad_fn is not None:
                self.perturbation.grad.detach_()
            else:
                self.perturbation.grad.requires_grad_(False)
            if set_to_none:
                self.perturbation.grad = None
            else:
                self.perturbation.grad.data.zero_()

    def forward(self, features, labels, is_snapshot=False):
        """Forward propagation of the robust loss.

        Compute the the function value of the robust loss.

        Args:
            features: the features of data samples 
            labels: the labels of data samples
            requires_grad: whether the gradient is required (default: True)

        Returns: return the loss
        """
        # compute the robust loss
        outputs = self.net(features + self.perturbation)
        loss = F.cross_entropy(
            outputs, labels) - self.regularizer_coef * torch.sum(self.perturbation**2)
        return loss

    def exchange(self, communicator, weight=None):
        """Exchange and average the model parameters.

        Arguments: 
            communicator: the communication handler
        """
        if weight is None:
            world_size = float(communicator.get_world_size())
            with torch.no_grad():
                for param in self.net.parameters():
                    communicator.all_reduce(param.data)
                    param.data /= world_size
                communicator.all_reduce(self.perturbation.data)
                self.perturbation.data /= world_size
        else:
            with torch.no_grad():
                for param in self.net.parameters():
                    param.data *= weight
                    communicator.all_reduce(param.data)
                self.perturbation.data *= weight
                communicator.all_reduce(self.perturbation.data)

    def broadcast(self, client_subset, communicator):
        """Broadcast the global model parameters.

        Arguments: 
            client_subset: the list of sampled clients
            communicator: the communication handler
        """
        with torch.no_grad():
            for param in self.net.parameters():
                communicator.broadcast(param.data, client_subset)
            communicator.broadcast(self.perturbation.data, client_subset)

    def reduce(self, client_subset, communicator, average_flag=True):
        """Aggregate the local models.

        Arguments: 
            client_subset: the list of sampled clients
            communicator: the communication handler
        """
        num_sampled_clients = len(client_subset)
        with torch.no_grad():
            for param in self.net.parameters():
                communicator.reduce(param.data, client_subset)
                if communicator.is_root() and average_flag:
                    param.data /= num_sampled_clients
            communicator.reduce(self.perturbation.data, client_subset)

            if communicator.is_root() and average_flag:
                self.perturbation.data /= num_sampled_clients

    def zero_param(self):
        with torch.no_grad():
            for param in self.net.parameters():
                param.data.zero_()
            self.perturbation.data.zero_()

    def copy_(self, src):
        with torch.no_grad():
            for param, src_param in zip(self.net.parameters(), src.net.parameters()):
                param.data.copy_(src_param.data)
            self.perturbation.data.copy_(src.perturbation.data)
        self.num_parameters = src.num_parameters
        self.regularizer_coef = src.regularizer_coef
        self.device = src.device

    def add_(self, src, alpha=1):
        with torch.no_grad():
            for param, src_param in zip(self.net.parameters(), src.net.parameters()):
                param.data.add_(src_param.data, alpha=alpha)
            self.perturbation.data.add_(src.perturbation.data, alpha=alpha)

    def multiply_(self, alpha):
        with torch.no_grad():
            for param in self.net.parameters():
                param.data.multiply_(alpha)
            self.perturbation.data.multiply_(alpha)

    def get_variable_lists(self):
        """Get two lists of primal and dual variables, resp.

        Returns: 
            primal_list: the list of primal variables
            dual_list: the list of dual variables
        """
        primal_list = list(self.net.parameters())
        dual_list = [self.perturbation]
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
        # if math.isnan(utility) or math.isinf(utility) or utility > 3.5:
        if math.isnan(utility) or math.isinf(utility) or utility > 2.5:
        # if math.isnan(utility) or math.isinf(utility):
            ret = True
        return ret

    def compute_net_norm(self):
        """Compute the norm of the network model
        """
        norm_value = 0.0
        with torch.no_grad():
            for param in self.net.parameters():
                norm_value += torch.sum(param.data**2).item()
        norm_value = norm_value**0.5
        return norm_value
