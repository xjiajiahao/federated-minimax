import os
import copy
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
import torch


def Catalyst_Scaffold_S_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, global_step_size, regularizer_coef, train_batch_size, num_stages=5000, resample_flag=False, max_machine_drop_ratio=0.0, device='cpu', print_freq=10, OUTPUT_DIR='./data/results/'):
    """Run the SCAFFOLD_Catalyst_S algorithm.

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
        global_step_size: the global step size
        regularizer_coef: the coefficient of the dual regularizer
        train_batch_size: the training batch size
        num_stages: the number of stages
        resample_flag (bool, default: False): whether to sample twice in one round
        max_machine_drop_ratio: the maximum drop ratio of selected machines
        device: the device to use (cpu or cuda)
        print_freq: the frequency of printing results
        OUTPUT_DIR: the directory to write output file
    """

    def closure(features, labels):
        """Evaluates the model, backward propagates, and returns the loss.
        """
        features = features.to(device)
        labels = labels.to(device)
        # @NOTE sum instead of average loss
        loss_value = model.forward(
            features, labels) * len(labels)
        model.zero_grad()
        loss_value.backward()
        return loss_value.item()

    # initialize the optimization algorithm
    rank = communicator.get_rank()
    if rank >= num_nodes:
        raise ValueError('rank MUST be smaller than num_nodes, but rank = {:d}, and num_ndoes = {:d}'.format(
            rank, num_nodes))

    configs = 'Catalyst_Scaffold_S_partitions_{:d}_nodes_{:d}_rounds_{:d}_local_iters_{:d}_T0_{:d}_local_step_{:f}_global_step_{:f}_reg_coef_{:f}_train_batch_size_{:d}_drop_ratio_{:f}'.format(
        num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, global_step_size, regularizer_coef, train_batch_size, max_machine_drop_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_name = OUTPUT_DIR + configs

    # set to None to mute warning
    test_loader = None

    if communicator.is_root():
        print('running {:s}, # partitions: {:d}, # nodes: {:d}, # rounds: {:d}, # local iterations: {:d}, T0: {:d}, local step size: {:f}, global step size: {:f}, reg coef: {:f}, train batch size: {:d}, max drop ratio: {:f}'.format(
            'Catalyst_Scaffold_S', num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, global_step_size, regularizer_coef, train_batch_size, max_machine_drop_ratio))
        # initialization

        test_loader = data_loader.build_test_loader()
        model.logger_init(output_file_name)

        # print statistics
        curr_received_doubles = 0
        model.evaluate_and_log(0, curr_received_doubles, test_loader)

    snapshot_model = copy.deepcopy(model)
    old_model = copy.deepcopy(model)
    agg_grad = copy.deepcopy(model)
    optimizer = Catalyst_Scaffold_S(model.get_variable_lists(), old_model.get_variable_lists(), snapshot_model.get_variable_lists(
    ), agg_grad.get_variable_lists(), local_step_size, global_step_size, regularizer_coef, model.projection())
    count_total_rounds = 0
    update_snapshot_flag = False

    my_own_train_loader = None
    # too_bad_flag = torch.zeros(1, dtype=torch.bool, device=device)
    too_bad_flag = torch.zeros(1, dtype=torch.int32, device=device)
    # the main loop
    for curr_stage in range(num_stages):
        # update the proximal point
        snapshot_model.copy_(model)

        # draw T0 samples from the Bernoulli's distribution B(1, 1.0/ num_local_iterations)
        comm_indices = []
        while True:
            comm_flags = np.random.binomial(1, 1.0/num_local_iterations, T0)
            comm_indices = np.where(comm_flags == 1)[0]
            if len(comm_indices) > 0:
                break
        comm_indices = comm_indices.tolist()
        # Run multiple rounds
        for i in range(len(comm_indices)):
            if comm_indices[i] == 0:
                continue
            if i > 0:
                curr_num_local_iterations = comm_indices[i] - comm_indices[i-1]
            else:
                curr_num_local_iterations = comm_indices[i]
            # Step 1. Sample a minibatch of clients
            # @NOTE all machines share the same random seed
            actual_num_nodes = num_nodes - math.floor(np.random.rand() * max_machine_drop_ratio * num_nodes)
            curr_weight = None
            candidate_partitions = None
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

            # Step 3. compute and exchange full gradients
            if rank < len(curr_partitions):
                agg_grad.zero_param()
                optimizer.compute_full_gradient(curr_train_loader, closure)
            agg_grad.exchange(communicator, curr_weight)

            if resample_flag:
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
            else:
                if actual_num_nodes < num_partitions:
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

            # Step 4. perform multiple local updates
            if rank < len(curr_partitions):
                old_model.copy_(model)
                for _ in range(curr_num_local_iterations):
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

                    old_loss = old_model.forward(features, labels, is_snapshot=True)
                    old_loss.backward()

                    optimizer.local_step()
                optimizer.global_step()

            # Step 3. exchange and average primal and dual variables
            model.exchange(communicator, weight=curr_weight)

            count_total_rounds += 1
            # print statistics
            if communicator.is_root() and count_total_rounds % print_freq == 0:
                curr_received_doubles = actual_num_nodes * model.get_num_parameters() * 2
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
            # exit if the #rounds of communication attains some value
            if count_total_rounds >= num_rounds:
                break
        if count_total_rounds >= num_rounds:
            break

    # wait until all processes complete
    print('Finished Training.\a')
    communicator.barrier()


class Catalyst_Scaffold_S(Optimizer):
    """The Catalyst_Scaffold_S algorithm class.

    Arguments:
        model_var_tuple: (primal variable list, dual variable list)
            tuple of the main model
        old_model_var_tuple: (primal variable list, dual variable
            list) tuple of the old model
        snapshot_model_var_tuple: (primal variable list, dual variable
        agg_grad_var_tuple: (primal variable list, dual variable
            list) tuple of the aggregated gradient
        local_step_size (float): the local step size
        global_step_size (float): the global step size
        regularizer_coef (float): the coefficient of the l2 regularizer
        projection (function handle): projection onto the constraint (default: None)
    """

    def __init__(self, model_var_tuple, old_model_var_tuple, snapshot_model_var_tuple, agg_grad_var_tuple, local_step_size=required, global_step_size=required, regularizer_coef=required, projection=None):
        if local_step_size is not required and local_step_size < 0.0:
            raise ValueError(
                "Invalid step size: {}".format(local_step_size))
        if global_step_size is not required and global_step_size < 0.0:
            raise ValueError(
                "Invalid step size: {}".format(global_step_size))
        if regularizer_coef is not required and regularizer_coef < 0.0:
            raise ValueError(
                "Invalid regularizer coefficient: {}".format(regularizer_coef))
        # add main params to self.param_groups
        # @NOTE descent for the primal variable and ascent for the dual one
        param_groups = [
            {'params': model_var_tuple[0], 'old_params': old_model_var_tuple[0], 'agg_grad': agg_grad_var_tuple[0], 'snapshot_params': snapshot_model_var_tuple[0],
                'local_step_size': local_step_size, 'global_step_size': global_step_size, 'regularizer_coef': regularizer_coef, 'primal_flag': True},
            {'params': model_var_tuple[1], 'old_params': old_model_var_tuple[1], 'agg_grad': agg_grad_var_tuple[1], 'snapshot_params': snapshot_model_var_tuple[1], 'local_step_size': local_step_size, 'global_step_size': global_step_size, 'regularizer_coef': regularizer_coef, 'primal_flag': False}]
        defaults = {'local_step_size': required, 'global_step_size': required}
        super(Catalyst_Scaffold_S, self).__init__(param_groups, defaults)
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
            for p, old_p in zip(group['params'], group['old_params']):
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
                if old_p.grad is not None:
                    if set_to_none:
                        old_p.grad = None
                    else:
                        if old_p.grad.grad_fn is not None:
                            old_p.grad.detach_()
                        else:
                            old_p.grad.requires_grad_(False)
                        old_p.grad.zero_()

    def compute_full_gradient(self, train_loader, closure):
        # Step 1. Zero the gradient of the parameters
        self.zero_grad()

        # Step 2. Iterate over all the data samples to compute the average gradient
        for i, (features, labels) in enumerate(train_loader):
            closure(features, labels)
            for group in self.param_groups:
                for idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    agg_g = group['agg_grad'][idx]
                    agg_g.data.add_(
                        p.grad.data, alpha=1/len(train_loader.dataset))

    def local_step(self):
        """Take a local update step on both the primal and dual variables.
        """
        for group in self.param_groups:
            step_size = group['local_step_size']
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                old_p = group['old_params'][idx]
                sp = group['snapshot_params'][idx]
                agg_g = group['agg_grad'][idx]
                d_p = p.grad.data
                # add the global gradient estiamate
                d_p.add_(agg_g.data)
                # subtract the snapshot gradient
                if old_p.grad is not None:
                    d_p.add_(old_p.grad.data, alpha=-1)
                # take a descent/ascent step
                if group['primal_flag']:
                    d_p.add_(p.data, alpha=group['regularizer_coef'])
                    d_p.add_(sp.data, alpha=-group['regularizer_coef'])
                    p.data.add_(d_p, alpha=-step_size)
                else:
                    d_p.add_(p.data, alpha=-group['regularizer_coef'])
                    d_p.add_(sp.data, alpha=group['regularizer_coef'])
                    p.data.add_(d_p, alpha=step_size)
        self.projection()

    def global_step(self):
        for group in self.param_groups:
            step_size = group['global_step_size'] / group['local_step_size']
            for idx, p in enumerate(group['params']):
                old_p = group['old_params'][idx]
                p.data.multiply_(step_size)
                p.data.add_(old_p.data, alpha=1-step_size)
