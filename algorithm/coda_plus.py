import os
import copy
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
import torch


def CODA_Plus_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, regularizer_coef, train_batch_size, num_stages=5000, max_machine_drop_ratio=0.0, device='cpu', print_freq=10, OUTPUT_DIR='./data/results/'):
    """Run the CODA_Plus algorithm.

    Arguments:
        model: the primal-dual model
        communicator: the communication handler
        data_loader: the data handler
        num_partitions: the number of the data partitions
        num_nodes: the number of available physical nodes
        num_rounds: the number of communication rounds
        num_local_iterations: the number of local iterations per round 
        T_0: the frequency to decay the step size (by 3 times)
        local_step_size: the local step size
        regularizer_coef: the coefficient of the dual regularizer
        train_batch_size: the training batch size
        num_stages: the number of stages
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

    configs = 'CODA_Plus_partitions_{:d}_nodes_{:d}_rounds_{:d}_local_iters_{:d}_T0_{:d}_local_step_{:f}_reg_coef_{:f}_train_batch_size_{:d}_drop_ratio_{:f}'.format(
        num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, regularizer_coef, train_batch_size, max_machine_drop_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_name = OUTPUT_DIR + configs

    # set to None to mute warning
    test_loader = None

    if communicator.is_root():
        print('running {:s}, # partitions: {:d}, # nodes: {:d}, # rounds: {:d}, # local iterations: {:d}, T0: {:d}, local step size: {:f}, reg coef: {:f}, train batch size: {:d}'.format(
            'CODA_Plus', num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, regularizer_coef, train_batch_size))
        # initialization

        test_loader = data_loader.build_test_loader()
        model.logger_init(output_file_name)

        # print statistics
        curr_received_doubles = 0
        model.evaluate_and_log(0, curr_received_doubles, test_loader)

    snapshot_model = copy.deepcopy(model)
    optimizer = CODA_Plus(model.get_variable_lists(), snapshot_model.get_variable_lists(
    ), local_step_size, regularizer_coef, model.projection())
    count_total_rounds = 0
    num_rounds_per_stage = int(T0 / num_local_iterations)
    update_snapshot_flag = False

    my_own_train_loader = None
    # too_bad_flag = torch.zeros(1, dtype=torch.bool, device=device)
    too_bad_flag = torch.zeros(1, dtype=torch.int32, device=device)
    # the main loop
    for curr_stage in range(num_stages):
        # update the proximal point
        snapshot_model.copy_(model)
        # for each communication round
        for _ in range(num_rounds_per_stage):
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

            # Step 2. perform multiple local updates (actually, epochs)
            if rank < len(curr_partitions):
                train_iter = iter(curr_train_loader)
                for _ in range(num_local_iterations):
                    # Step 3. sample a minibatch of local data
                    try:
                        features, labels = train_iter.next()
                    except:
                        train_iter = iter(curr_train_loader)
                        features, labels = train_iter.next()
                    features = features.to(device)
                    labels = labels.to(device)
                    # Step 4. compute the gradients at the current variable
                    optimizer.zero_grad()
                    curr_loss = model.forward(features, labels)
                    curr_loss.backward()

                    optimizer.local_step(curr_stage)

            # Step 3. exchange and average primal and dual variables
            model.exchange(communicator, weight=curr_weight)

            # exit if the #rounds of communication attains some value
            count_total_rounds += 1
            # print statistics
            if communicator.is_root() and count_total_rounds % print_freq == 0:
                curr_received_doubles = actual_num_nodes * model.get_num_parameters()
                too_bad_flag[0] = model.evaluate_and_log(
                    count_total_rounds, curr_received_doubles, test_loader)
                # if too_bad_flag:
                #     raise ValueError(
                #         "The performance of training is so bad that we stop the training.")
            # determines whether we should exit
            if count_total_rounds % print_freq == 0:
                communicator.barrier()
                communicator.all_reduce(too_bad_flag)
                if too_bad_flag[0]:
                        raise ValueError(
                            "The performance of training is so bad that we stop the training.") 

            if count_total_rounds >= num_rounds:
                break
        if count_total_rounds >= num_rounds:
            break

    # wait until all processes complete
    print('Finished Training.\a')
    communicator.barrier()


class CODA_Plus(Optimizer):
    """The CODA_Plus algorithm class.

    Arguments:
        model_var_tuple: (primal variable list, dual variable list)
            tuple of the main model
        snapshot_model_var_tuple: (primal variable list, dual variable
            list) tuple of the snapshot model (the proximal point)
        step_size (float): the local step size
        projection (function handle): projection onto the constraint (default: None)
    """

    def __init__(self, model_var_tuple, snapshot_model_var_tuple, step_size=required, regularizer_coef=required, projection=None):
        if step_size is not required and step_size < 0.0:
            raise ValueError(
                "Invalid step size: {}".format(step_size))
        if regularizer_coef is not required and regularizer_coef < 0.0:
            raise ValueError(
                "Invalid regularizer coefficient: {}".format(step_size))
        # add main params to self.param_groups
        # @NOTE descent for the primal variable and ascent for the dual one
        param_groups = [
            {'params': model_var_tuple[0], 'snapshot_params': snapshot_model_var_tuple[0],
                'step_size': step_size, 'regularizer_coef': regularizer_coef, 'primal_flag': True},
            {'params': model_var_tuple[1], 'snapshot_params': snapshot_model_var_tuple[1], 'step_size': step_size, 'primal_flag': False}]
        defaults = {'step_size': required}
        super(CODA_Plus, self).__init__(param_groups, defaults)
        self.projection = projection

    def local_step(self, curr_stage):
        """Take a local update step on both the primal and dual variables.

        Arguments:
            curr_stage: the step size decaying level
        """
        for group in self.param_groups:
            step_size = group['step_size'] / 3**(curr_stage)
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                sp = group['snapshot_params'][idx]
                d_p = p.grad.data
                if group['primal_flag']:
                    d_p.add_(p.data, alpha=group['regularizer_coef'])
                    d_p.add_(sp.data, alpha=-group['regularizer_coef'])
                    p.data.add_(d_p, alpha=-step_size)
                else:
                    p.data.add_(d_p, alpha=step_size)
        self.projection()

    def zero_grad(self, set_to_none: bool = False):
        """Clear the gradients of all optimized main params.

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
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
