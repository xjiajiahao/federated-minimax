import os
import copy
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
import torch


def Extra_Step_Local_SGD_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, local_step_size, train_batch_size, max_machine_drop_ratio=0.0, device='cpu', print_freq=10, OUTPUT_DIR='./data/results/'):
    """Run the Extra_Step_Local_SGD algorithm.

    Arguments:
        model: the primal-dual model
        communicator: the communication handler
        data_loader: the data handler
        num_partitions: the number of the data partitions
        num_nodes: the number of available physical nodes
        num_rounds: the number of communication rounds
        num_local_iterations: the number of local iterations per round 
        T_0: the number of iterations in one stage
        local_step_size: the local step size
        train_batch_size: the training batch size
        max_machine_drop_ratio: the maximum drop ratio of selected machines
        device: the device to use (cpu or cuda)
        print_freq: the frequency of printing results
        OUTPUT_DIR: the directory to write output file
    """

    # initialize the optimization algorithm
    rank = communicator.get_rank()
    if rank >= num_nodes:
        raise ValueError('rank MUST be smaller than num_nodes, but rank = {:d}, and num_ndoes = {:d}'.format(
            rank, num_nodes))

    configs = 'Extra_Step_Local_SGD_partitions_{:d}_nodes_{:d}_rounds_{:d}_local_iters_{:d}_local_step_{:f}_train_batch_size_{:d}_drop_ratio_{:f}'.format(
        num_partitions, num_nodes, num_rounds, num_local_iterations, local_step_size, train_batch_size, max_machine_drop_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_name = OUTPUT_DIR + configs

    # set to None to mute warning
    test_loader = None

    if communicator.is_root():
        print('running {:s}, # partitions: {:d}, # nodes: {:d}, # rounds: {:d}, # local iterations: {:d}, local step size: {:f}, train batch size: {:d}, max drop ratio: {:f}'.format(
            'Extra_Step_Local_SGD', num_partitions, num_nodes, num_rounds, num_local_iterations, local_step_size, train_batch_size, max_machine_drop_ratio))
        # initialization

        test_loader = data_loader.build_test_loader()
        model.logger_init(output_file_name)

        # print statistics
        curr_received_doubles = 0
        model.evaluate_and_log(0, curr_received_doubles, test_loader)

    extra_model = copy.deepcopy(model)
    optimizer = Extra_Step_Local_SGD(model.get_variable_lists(), extra_model.get_variable_lists(
    ), local_step_size, model.projection())

    my_own_train_loader = None
    # too_bad_flag = torch.zeros(1, dtype=torch.bool, device=device)
    too_bad_flag = torch.zeros(1, dtype=torch.int32, device=device)
    for curr_round in range(num_rounds):
        # Step 1. Sample a minibatch of clients
        # @NOTE all machines share the same random seed
        actual_num_nodes = num_nodes - math.floor(np.random.rand() * max_machine_drop_ratio * num_nodes)
        curr_weight = None
        if actual_num_nodes < num_partitions:
            candidate_partitions = np.random.choice(
                num_partitions, num_nodes, replace=False)
            curr_partitions = np.sort(np.random.choice(
                candidate_partitions, actual_num_nodes, replace=False)).tolist()
            if rank < len(curr_partitions):
                curr_train_loader = data_loader.build_train_loader(
                    curr_partitions[rank])
                curr_weight = 1.0 / actual_num_nodes
            else:
                curr_weight = 0.0
        else:
            if my_own_train_loader is None:
                my_own_train_loader = data_loader.build_train_loader(rank)
            curr_train_loader = my_own_train_loader

        # Step 2. perform multiple local updates
        if rank < len(curr_partitions):
            for _ in range(num_local_iterations):
                # the first step
                extra_model.copy_(model)
                try:
                    features, labels = train_iter.next()
                except:
                    train_iter = iter(curr_train_loader)
                    features, labels = train_iter.next()
                features = features.to(device)
                labels = labels.to(device)
                # compute the gradients at the current variable
                optimizer.zero_grad()
                curr_loss = extra_model.forward(features, labels)
                curr_loss.backward()
                optimizer.local_step_first()

                # the second step
                try:
                    features, labels = train_iter.next()
                except:
                    train_iter = iter(curr_train_loader)
                    features, labels = train_iter.next()
                features = features.to(device)
                labels = labels.to(device)
                # compute the gradients at the current variable
                optimizer.zero_grad()
                extra_loss = extra_model.forward(features, labels)
                extra_loss.backward()
                optimizer.local_step_second()

        # Step 3. exchange and average primal and dual variables
        model.exchange(communicator, weight=curr_weight)

        # print statistics
        if communicator.is_root() and (curr_round + 1) % print_freq == 0:
            curr_received_doubles = actual_num_nodes * model.get_num_parameters()
            too_bad_flag[0] = model.evaluate_and_log(
                curr_round + 1, curr_received_doubles, test_loader)
            # if too_bad_flag:
            #     raise ValueError(
            #         "The performance of training is so bad that we stop the training.")
        # determines whether we should exit
        if (curr_round + 1) % print_freq == 0:
            communicator.barrier()
            communicator.all_reduce(too_bad_flag)
            if too_bad_flag[0]:
                    raise ValueError(
                        "The performance of training is so bad that we stop the training.") 

    # wait until all processes complete
    print('Finished Training.\a')
    communicator.barrier()


class Extra_Step_Local_SGD(Optimizer):
    """The Extra_Step_Local_SGD algorithm class.

    Arguments:
        model_var_tuple: (primal variable list, dual variable list)
            tuple of the main model
        extra_model_var_tuple: (primal variable list, dual variable
            list) tuple of the old model
        local_step_size (float): the local step size
        projection (function handle): projection onto the constraint (default: None)
    """

    def __init__(self, model_var_tuple, extra_model_var_tuple, local_step_size=required, projection=None):
        if local_step_size is not required and local_step_size < 0.0:
            raise ValueError(
                "Invalid step size: {}".format(local_step_size))
       # add main params to self.param_groups
        # @NOTE descent for the primal variable and ascent for the dual one
        param_groups = [
            {'params': model_var_tuple[0], 'extra_params': extra_model_var_tuple[0],
                'local_step_size': local_step_size, 'primal_flag': True},
            {'params': model_var_tuple[1], 'extra_params': extra_model_var_tuple[1], 'local_step_size': local_step_size, 'primal_flag': False}]
        defaults = {'local_step_size': required}
        super(Extra_Step_Local_SGD, self).__init__(param_groups, defaults)
        self.projection = projection

    def zero_grad(self, set_to_none: bool = False):
        """Clear the gradients of all optimized main params and old params.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.param_groups:
            for p, extra_p in zip(group['params'], group['extra_params']):
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
                if extra_p.grad is not None:
                    if set_to_none:
                        extra_p.grad = None
                    else:
                        if extra_p.grad.grad_fn is not None:
                            extra_p.grad.detach_()
                        else:
                            extra_p.grad.requires_grad_(False)
                        extra_p.grad.zero_()

    def local_step_first(self):
        """Take the first local update step.
        """
        for group in self.param_groups:
            step_size = group['local_step_size']
            for idx, extra_p in enumerate(group['extra_params']):
                if extra_p.grad is None:
                    continue
                d_p = extra_p.grad.data
                # take a descent/ascent step
                if group['primal_flag']:
                    extra_p.data.add_(d_p, alpha=-step_size)
                else:
                    extra_p.data.add_(d_p, alpha=step_size)
        self.projection()

    def local_step_second(self):
        """Take the second local update step.
        """
        for group in self.param_groups:
            step_size = group['local_step_size']
            for idx, p in enumerate(group['params']):
                extra_p = group['extra_params'][idx]
                if extra_p.grad is None:
                    continue
                d_p = extra_p.grad.data
                # take a descent/ascent step
                if group['primal_flag']:
                    p.data.add_(d_p, alpha=-step_size)
                else:
                    p.data.add_(d_p, alpha=step_size)
        self.projection()
