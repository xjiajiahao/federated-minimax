import os
import copy
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
import torch


def Local_SGDA_Plus_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, primal_step_size, dual_step_size, train_batch_size, max_machine_drop_ratio=0.0, device='cpu', print_freq=10, OUTPUT_DIR='./data/results/'):
    """Run the Local_SGDA_Plus algorithm.

    Arguments:
        model: the primal-dual model
        communicator: the communication handler
        data_loader: the data handler
        num_partitions: the number of the data partitions
        num_nodes: the number of available physical nodes
        num_rounds: the number of communication rounds
        num_local_iterations: the number of local iterations per round 
        primal_step_size: the primal step size
        dual_step_size: the dual step size
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

    configs = 'Local_SGDA_Plus_partitions_{:d}_nodes_{:d}_rounds_{:d}_local_epochs_{:d}_primal_step_{:f}_dual_step_{:f}_train_batch_size_{:d}_drop_ratio_{:f}'.format(
        num_partitions, num_nodes, num_rounds, num_local_iterations, primal_step_size, dual_step_size, train_batch_size, max_machine_drop_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_name = OUTPUT_DIR + configs

    # set to None to mute warning
    test_loader = None

    if communicator.is_root():
        print('running {:s}, # partitions: {:d}, # nodes: {:d}, # rounds: {:d}, # local iterations: {:d}, primal step size: {:f}, dual step size: {:f}, train batch size: {:d}, max drop ratio: {:f}'.format(
            'Local_SGDA_Plus', num_partitions, num_nodes, num_rounds, num_local_iterations, primal_step_size, dual_step_size, train_batch_size, max_machine_drop_ratio))

        test_loader = data_loader.build_test_loader()
        model.logger_init(output_file_name)

        # print statistics
        curr_received_doubles = 0
        model.evaluate_and_log(0, curr_received_doubles, test_loader)

    snapshot_model = copy.deepcopy(model)
    optimizer = Local_SGDA_Plus(model.get_variable_lists(), snapshot_model.get_variable_lists(
    ), primal_step_size, dual_step_size, model.projection())

    my_own_train_loader = None
    # too_bad_flag = torch.zeros(1, dtype=torch.bool, device=device)
    too_bad_flag = torch.zeros(1, dtype=torch.int32, device=device)
    # for each communication round
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

		# Step 2. update snapshot model
        optimizer.update_snapshot_model()

        # Step 3. perform multiple local updates
        if rank < len(curr_partitions):
            train_iter = iter(curr_train_loader)
            for _ in range(num_local_iterations):
                # Step 3.1 sample a minibatch of local data
                try:
                    features, labels = train_iter.next()
                except:
                    train_iter = iter(curr_train_loader)
                    features, labels = train_iter.next()
                features = features.to(device)
                labels = labels.to(device)
                # Step 3.2 compute the gradients at the current variable
                optimizer.zero_grad()
                curr_loss = model.forward(features, labels)
                curr_loss.backward()

                snapshot_loss = snapshot_model.forward(features, labels)
                snapshot_loss.backward()

                optimizer.step()

        # Step 4. exchange and average primal and dual variables
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


class Local_SGDA_Plus(Optimizer):
    """The Local_SGDA_Plus algorithm class.

    Arguments:
        model_variables (tuple): (primal variable list, dual variable list)
            tuple of the main model
        primal_step_size (float): the primal step size
        dual_step_size (float): the dual step size
        projection (function handle): projection onto the constraint (default: None)
    """

    def __init__(self, model_var_tuple, snapshot_model_var_tuple, primal_step_size=required, dual_step_size=required, projection=None):
        if primal_step_size is not required and primal_step_size < 0.0:
            raise ValueError(
                "Invalid primal step size: {}".format(primal_step_size))
        if dual_step_size is not required and dual_step_size < 0.0:
            raise ValueError(
                "Invalid dual step size: {}".format(dual_step_size))
        # add main params to self.param_groups
        # @NOTE descent for the primal variable and ascent for the dual one
        param_groups = [
            {'params': model_var_tuple[0], 'snapshot_params': snapshot_model_var_tuple[0],
                'step_size': primal_step_size, 'primal_flag': True},
            {'params': model_var_tuple[1], 'snapshot_params': snapshot_model_var_tuple[1], 'step_size': -1.0 * dual_step_size, 'primal_flag': False}]
        defaults = {'step_size': required}
        super(Local_SGDA_Plus, self).__init__(param_groups, defaults)

        self.projection = projection

    def step(self, closure=None):
        """Take an update step on both the primal and dual variables.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            if isinstance(loss, tuple):
                loss, _ = loss

        for group in self.param_groups:
            if group['primal_flag']:
                for p, sp in zip(group['params'], group['snapshot_params']):
                    if p.grad is None:
                        continue
                    # gradient data
                    d_p = p.grad.data
                    p.data.add_(d_p, alpha=-group['step_size'])
                    sp.data.copy_(p.data)
            else:
                for p, sp in zip(group['params'], group['snapshot_params']):
                    if sp.grad is None:
                        continue
                    # gradient data
                    d_sp = sp.grad.data
                    sp.data.add_(d_sp, alpha=-group['step_size'])
                    p.data.copy_(sp.data)
        self.projection()
        return loss

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
            for p, sp in zip(group['params'], group['snapshot_params']):
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
                if sp.grad is not None:
                    if set_to_none:
                        sp.grad = None
                    else:
                        if sp.grad.grad_fn is not None:
                            sp.grad.detach_()
                        else:
                            sp.grad.requires_grad_(False)
                        sp.grad.zero_()

    def update_snapshot_model(self):
        for group in self.param_groups:
            for p, sp in zip(group['params'], group['snapshot_params']):
                sp.data.copy_(p.data)
