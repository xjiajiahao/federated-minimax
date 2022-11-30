import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import math
from datetime import datetime
import time
import warnings


class DeepAUC:
    """ The AUC model with a deep network

    Arguments:
        net: the deep network model
        pos_ratio (float): the ratio of positive samples
        device: the device to use (cpu or gpu)
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
                header = '#round, time (s), #floats received by the server, auc value\n'
                f.write(header)

        def stop_timing(self):
            self.t_elapsed = time.time() - self.t_start
            self.running_time = max(self.t_elapsed - self.overhead_time, 0.0)

        def continue_timing(self):
            self.overhead_time += time.time() - self.t_start - self.t_elapsed

        def log(self, curr_round, curr_received_floats, auc_value):
            self.count_received_floats += curr_received_floats
            data_to_log = np.array(
                [[curr_round, self.running_time, self.count_received_floats, auc_value]])
            print('[{:s} round {:d}] elapsed time: {:.1f}, #received floats: {:.4e}, auc value: {:f}'.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), curr_round, self.running_time, self.count_received_floats, auc_value))

            # write to the output file
            with open(self.output_file_tag + '.csv', 'a') as f:
                np.savetxt(f, data_to_log, delimiter=', ')

    def __init__(self, net, pos_ratio, device):
        self.net = net
        self.device = device
        self.a = torch.zeros(1, dtype=torch.float32,
                             device=device, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float32,
                             device=device, requires_grad=True)
        self.alpha = torch.zeros(
            1, dtype=torch.float32, device=device, requires_grad=True)
        self.num_parameters = 3 + \
            sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        self.pos_ratio = pos_ratio
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
        auc_value = self.evaluate_metric(test_loader)
        self.logger.log(curr_round, curr_received_floats, auc_value)
        if curr_round >= 50 and self.abort_condition(auc_value):
        # if curr_round >= 90 and self.abort_condition(auc_value):
            too_bad_flag = True
        else:
            too_bad_flag = False
        self.logger.continue_timing()
        return too_bad_flag

    def evaluate_metric(self, test_loader):
        """Evaluate the true AUC vaule.

        Args: 
            test_loader: the data loader of the test dataset

        Returns:
            auc_value: the auc value
        """
        score_list = list()
        label_list = list()

        with torch.no_grad():
            self.net.eval()
            for _, data in enumerate(test_loader, 0):
                tmp_data, tmp_label = data
                tmp_data = tmp_data.to(self.device)
                tmp_score = F.softmax(self.net(tmp_data), dim=1)[:, 1].detach().clone().cpu()
                score_list.append(tmp_score)
                label_list.append(tmp_label.cpu())
            test_label = torch.cat(label_list)
            test_score = torch.cat(score_list)
            # compute AUC
            if torch.sum(torch.isnan(test_score)) > 0 or torch.sum(torch.isinf(test_score)) > 0:
                return 0.0
            fpr, tpr, _ = metrics.roc_curve(
                test_label, test_score, pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            self.net.train()
        return auc_value

    def zero_grad(self, set_to_none=False):
        """Clear the gradient of the model."""
        self.net.zero_grad(set_to_none)
        if self.a.grad is not None:
            if self.a.grad.grad_fn is not None:
                self.a.grad.detach_()
            else:
                self.a.grad.requires_grad_(False)
            if set_to_none:
                self.a.grad = None
            else:
                self.a.grad.data.zero_()
        if self.b.grad is not None:
            if self.b.grad.grad_fn is not None:
                self.b.grad.detach_()
            else:
                self.b.grad.requires_grad_(False)
            if set_to_none:
                self.b.grad = None
            else:
                self.b.grad.data.zero_()
        if self.alpha.grad is not None:
            if self.alpha.grad.grad_fn is not None:
                self.alpha.grad.detach_()
            else:
                self.alpha.grad.requires_grad_(False)
            if set_to_none:
                self.alpha.grad = None
            else:
                self.alpha.grad.data.zero_()

    def forward(self, features, labels, is_snapshot=False):
        """Forward propagation of the minimax reformulation of AUC.

        Compute the the function value of the minimax version of AUC.

        Args:
            features: the features of data samples 
            labels: the labels of data samples
            requires_grad: whether the gradient is required (default: True)

        Returns: return the loss
        """
        # compute the minimax formulation of auc
        score = F.softmax(self.net(features), dim=1)[:, 1]
        loss = (1 - self.pos_ratio) * torch.mean((score - self.a)**2 * (1 == labels).float()) \
            + self.pos_ratio * torch.mean((score - self.b)**2 * (-1 == labels).float()) \
            + 2 * (1 + self.alpha) * torch.mean(
                self.pos_ratio * score * (-1 == labels).float()
                - (1 - self.pos_ratio) * score * (1 == labels).float()) \
            - self.pos_ratio * (1 - self.pos_ratio) * self.alpha**2
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
                communicator.all_reduce(self.a.data)
                communicator.all_reduce(self.b.data)
                communicator.all_reduce(self.alpha.data)

                self.a.data /= world_size
                self.b.data /= world_size
                self.alpha.data /= world_size
        else:
            with torch.no_grad():
                for param in self.net.parameters():
                    param.data *= weight
                    communicator.all_reduce(param.data)
                self.a.data *= weight
                self.b.data *= weight
                self.alpha.data *= weight
                communicator.all_reduce(self.a.data)
                communicator.all_reduce(self.b.data)
                communicator.all_reduce(self.alpha.data)

    def broadcast(self, client_subset, communicator):
        """Broadcast the global model parameters.

        Arguments: 
            client_subset: the list of sampled clients
            communicator: the communication handler
        """
        with torch.no_grad():
            for param in self.net.parameters():
                communicator.broadcast(param.data, client_subset)
            communicator.broadcast(self.a.data, client_subset)
            communicator.broadcast(self.b.data, client_subset)
            communicator.broadcast(self.alpha.data, client_subset)

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
            communicator.reduce(self.a.data, client_subset)
            communicator.reduce(self.b.data, client_subset)
            communicator.reduce(self.alpha.data, client_subset)

            if communicator.is_root() and average_flag:
                self.a.data /= num_sampled_clients
                self.b.data /= num_sampled_clients
                self.alpha.data /= num_sampled_clients

    def zero_param(self):
        with torch.no_grad():
            for param in self.net.parameters():
                param.data.zero_()
            self.a.data.zero_()
            self.b.data.zero_()
            self.alpha.data.zero_()

    def copy_(self, src):
        with torch.no_grad():
            for param, src_param in zip(self.net.parameters(), src.net.parameters()):
                param.data.copy_(src_param.data)
            self.a.data.copy_(src.a.data)
            self.b.data.copy_(src.b.data)
            self.alpha.data.copy_(src.alpha.data)
        self.num_parameters = src.num_parameters
        self.pos_ratio = src.pos_ratio
        self.device = src.device

    def add_(self, src, alpha=1):
        with torch.no_grad():
            for param, src_param in zip(self.net.parameters(), src.net.parameters()):
                param.data.add_(src_param.data, alpha=alpha)
            self.a.data.add_(src.a.data, alpha=alpha)
            self.b.data.add_(src.b.data, alpha=alpha)
            self.alpha.data.add_(src.alpha.data, alpha=alpha)

    def multiply_(self, alpha):
        with torch.no_grad():
            for param in self.net.parameters():
                param.data.multiply_(alpha)
            self.a.data.multiply_(alpha)
            self.b.data.multiply_(alpha)
            self.alpha.data.multiply_(alpha)

    def get_variable_lists(self):
        """Get two lists of primal and dual variables, resp.

        Returns: 
            primal_list: the list of primal variables
            dual_list: the list of dual variables
        """
        primal_list = [self.a, self.b]
        primal_list += list(self.net.parameters())
        dual_list = [self.alpha]
        return primal_list, dual_list

    def get_pos_ratio(self):
        return self.pos_ratio

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
        if math.isnan(utility) or math.isinf(utility) or utility < 0.6:
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
