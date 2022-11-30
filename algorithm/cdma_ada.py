import os
import copy
import torch
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required


def CDMA_ADA_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, primal_step_size, dual_step_size, step_size_exp, primal_alpha, dual_alpha, alpha_exp, train_batch_size, resample_flag=False, max_machine_drop_ratio=0.0, device='cpu', print_freq=10, OUTPUT_DIR='./data/results/'):
    """Run the CDMA_ADA algorithm (with the minibatch gradient estimator).

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
        step_size_exp (float): the exponential of the step size
        primal_alpha (float): the coefficient of primal momentum parameter
        dual_alpha (float): the coefficient of the dual momentum parameter
        alpha_exp (float): the exponential of momentum parameter
        resample_flag (bool, default: False): whether to sample twice in one round
        max_machine_drop_ratio: the maximum drop ratio of selected machines
        device: the device to use (cpu or cuda)
        print_freq: the frequency of printing results
        OUTPUT_DIR: the directory to write output file
    """

    snapshot_model = copy.deepcopy(model)

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

        snapshot_loss_value = snapshot_model.forward(
            features, labels) * len(labels)  # @NOTE sum instead of average loss
        snapshot_model.zero_grad()
        snapshot_loss_value.backward()

    # initialize the optimization algorithm
    rank = communicator.get_rank()
    if rank >= num_nodes:
        raise ValueError('rank MUST be smaller than num_nodes, but rank = {:d}, and num_ndoes = {:d}'.format(
            rank, num_nodes))

    configs = 'CDMA_ADA_partitions_{:d}_nodes_{:d}_rounds_{:d}_local_epochs_{:d}_primal_step_{:f}_dual_step_{:f}_step_exp_{:f}_primal_alpha_{:f}_dual_alpha_{:f}_alpha_exp_{:f}_train_batch_size_{:d}_drop_ratio_{:f}'.format(
        num_partitions, num_nodes, num_rounds, num_local_iterations, primal_step_size, dual_step_size, step_size_exp, primal_alpha, dual_alpha, alpha_exp, train_batch_size, max_machine_drop_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_name = OUTPUT_DIR + configs

    # set to None to mute warning
    test_loader = None

    if communicator.is_root():
        print('running {:s}, # partitions: {:d}, # nodes: {:d}, # rounds: {:d}, # local iterations: {:d}, primal step size: {:f}, dual step size: {:f}, step exp: {:f}, primal alpha: {:f}, dual alpha: {:f}, alpha exp: {:f}, train batch size: {:d}, max drop ratio: {:f}'.format(
            'CD-MAGE-VR', num_partitions, num_nodes, num_rounds, num_local_iterations, primal_step_size, dual_step_size, step_size_exp, primal_alpha, dual_alpha, alpha_exp, train_batch_size, max_machine_drop_ratio))

        test_loader = data_loader.build_test_loader()
        model.logger_init(output_file_name)

        # print statistics
        curr_received_doubles = 0
        model.evaluate_and_log(0, curr_received_doubles, test_loader)

    optimizer = CDMA_ADA(model.get_variable_lists(), snapshot_model.get_variable_lists(
    ), primal_step_size, dual_step_size, step_size_exp, primal_alpha, dual_alpha, alpha_exp, model.projection())

    my_own_train_loader = None
    # too_bad_flag = torch.zeros(1, dtype=torch.bool, device=device)
    too_bad_flag = torch.zeros(1, dtype=torch.int32, device=device)
    # for each communication round
    for curr_round in range(num_rounds):
        # gradient collection phase
        # Step 1. Sample a minibatch of clients
        # @NOTE all machines share the same random seed
        actual_num_nodes_gc = num_nodes - math.floor(np.random.rand() * max_machine_drop_ratio * num_nodes)
        curr_weight = None
        candidate_partitions = None
        if actual_num_nodes_gc < num_partitions:
            candidate_partitions = np.random.choice(
                num_partitions, num_nodes, replace=False)
            curr_partitions = np.sort(np.random.choice(
                candidate_partitions, actual_num_nodes_gc, replace=False)).tolist()
            if rank < len(curr_partitions):
                curr_train_loader = data_loader.build_train_loader(
                    curr_partitions[rank])
                curr_weight = 1.0 / actual_num_nodes_gc
            else:
                curr_train_loader = None
                curr_weight = 0.0
        else:
            if my_own_train_loader is None:
                my_own_train_loader = data_loader.build_train_loader(rank)
            curr_train_loader = my_own_train_loader

        # Step 2. compute and exchange the local gradient (u_t, v_t)
        optimizer.take_snapshot(
            curr_train_loader, closure, communicator, curr_round, curr_weight)

		# parameter update phase
        # Step 3. resample another subset of clients
        actual_num_nodes_pu = num_nodes - math.floor(np.random.rand() * max_machine_drop_ratio * num_nodes)
        if resample_flag:
            curr_weight = None
            if actual_num_nodes_pu < num_partitions:
                candidate_partitions = np.random.choice(
                    num_partitions, num_nodes, replace=False)
                curr_partitions = np.sort(np.random.choice(
                    candidate_partitions, actual_num_nodes_pu, replace=False)).tolist()
                if rank < len(curr_partitions):
                    curr_train_loader = data_loader.build_train_loader(
                        curr_partitions[rank])
                    curr_weight = 1.0 / actual_num_nodes_pu
                else:
                    curr_weight = 0.0
            else:
                if my_own_train_loader is None:
                    my_own_train_loader = data_loader.build_train_loader(rank)
                curr_train_loader = my_own_train_loader
        else:
            if actual_num_nodes_pu < num_partitions:
                curr_partitions = np.sort(np.random.choice(
                    candidate_partitions, actual_num_nodes_pu, replace=False)).tolist()
                if rank < len(curr_partitions):
                    curr_train_loader = data_loader.build_train_loader(
                        curr_partitions[rank])
                    curr_weight = 1.0 / actual_num_nodes_pu
                else:
                    curr_weight = 0.0
            else:
                if my_own_train_loader is None:
                    my_own_train_loader = data_loader.build_train_loader(rank)
                curr_train_loader = my_own_train_loader

        # Step 4. perform multiple local updates (actually, epochs)
        if rank < len(curr_partitions):
            for curr_local_iter in range(num_local_iterations):
                # Step 4.1 sample a minibatch of local data
                try:
                    features, labels = train_iter.next()
                except:
                    train_iter = iter(curr_train_loader)
                    features, labels = train_iter.next()
                features = features.to(device)
                labels = labels.to(device)
                # Step 4.2 compute the gradients at the current variable and the snapshot variable
                optimizer.zero_grad()
                curr_loss = model.forward(features, labels)
                curr_loss.backward()

                snapshot_loss = snapshot_model.forward(features, labels, is_snapshot=True)
                snapshot_loss.backward()
                optimizer.step()

        # Step 5. exchange and average primal and dual variables
        model.exchange(communicator, weight=curr_weight)
        # print statistics
        if communicator.is_root() and (curr_round + 1) % print_freq == 0:
            curr_num_participating_nodes = 0
            if num_local_iterations > 1:
                curr_num_participating_nodes = actual_num_nodes_gc + actual_num_nodes_pu
            else:
                curr_num_participating_nodes = actual_num_nodes_gc
            curr_received_doubles = curr_num_participating_nodes * model.get_num_parameters()
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


class CDMA_ADA(Optimizer):
    """The CDMA_ADA algorithm class.

    Arguments:
        model_variables (tuple): (primal variable list, dual variable list)
            tuple of the main model
        snapshot_model_variables (tuple): (primal variable list, dual variable
            list) tuple of the snapshot model
        primal_step_size_coef (float): the coefficient of primal step size
        dual_step_size_coef (float): the coefficient of dual step size
        step_size_exp (float): the exponential of the step size
        primal_alpha_coef (float): the coefficient of primal momentum parameter
        dual_alpha_coef (float): the coefficient of the dual momentum parameter
        alpha_exp (float): the exponential of momentum parameter
        projection (function handle): projection onto the constraint (default: None)
    """

    def __init__(self, model_var_tuple, snapshot_model_var_tuple, primal_step_size_coef=required, dual_step_size_coef=required, step_size_exp=required, primal_alpha_coef=required, dual_alpha_coef=required, alpha_exp=required, projection=None):
        if primal_step_size_coef is not required and primal_step_size_coef < 0.0:
            raise ValueError(
                "Invalid primal step size: {}".format(primal_step_size_coef))
        if dual_step_size_coef is not required and dual_step_size_coef < 0.0:
            raise ValueError(
                "Invalid dual step size: {}".format(dual_step_size_coef))
        # add main params to self.param_groups
        # @NOTE descent for the primal variable and ascent for the dual one
        param_groups = [
            {'params': model_var_tuple[0], 'snapshot_params': snapshot_model_var_tuple[0], 'step_size': primal_step_size_coef, 'step_size_coef': primal_step_size_coef,
                'step_size_exp': step_size_exp, 'alpha': primal_alpha_coef, 'alpha_coef': primal_alpha_coef, 'alpha_exp': alpha_exp},
            {'params': model_var_tuple[1], 'snapshot_params': snapshot_model_var_tuple[1], 'step_size': -1.0 * dual_step_size_coef, 'step_size_coef': -1.0 * dual_step_size_coef, 'step_size_exp': step_size_exp, 'alpha': dual_alpha_coef, 'alpha_coef': dual_alpha_coef, 'alpha_exp': alpha_exp}]
        defaults = {'step_size': required}
        super(CDMA_ADA, self).__init__(param_groups, defaults)
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
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                # State initialization, add the average gradient to the state
                if 'estimate_gradient' not in state:
                    state['estimate_gradient'] = torch.zeros_like(p.grad.data)
                average_gradient = state['estimate_gradient']
                snapshot_params = group['snapshot_params'][idx]
                # gradient data
                d_p = p.grad.data
                # add the average gradient
                d_p.add_(average_gradient)
                # subtract the snapshot gradient
                if snapshot_params.grad is not None:
                    d_p.add_(snapshot_params.grad.data, alpha=-1)
                p.data.add_(d_p, alpha=-group['step_size'])
        self.projection()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clear the gradients of all optimized main params and snapshot params.

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

    def take_snapshot(self, train_loader, closure, communicator, curr_round, weight):
        """Copies the model parameters to snapshot_model, and evaluates and
            exchanges the full (local) gradient.

        Arguments:
            train_loader: the training data loader
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        # Zero the gradient of the parameters
        self.zero_grad()

        if curr_round == 0:
            # state initialization
            for group in self.param_groups:
                for _, p in enumerate(group['params']):
                    state = self.state[p]
                    state['estimate_gradient'] = torch.zeros_like(p.data)
                    state['old_estimate_gradient'] = torch.zeros_like(p.data)

        for group in self.param_groups:
            group['step_size'] = group['step_size_coef'] / \
                (curr_round + 1) ** group['step_size_exp']
            group['alpha'] = min(group['alpha_coef'] /
                                 (curr_round + 1) ** group['alpha_exp'], 1.0)

        if weight != 0:
            for i, (features, labels) in enumerate(train_loader):
                closure(features, labels)
                for group in self.param_groups:
                    for p, sp in zip(group['params'], group['snapshot_params']):
                        state = self.state[p]
                        if p.grad is None:
                            continue
                        if i == 0:
                            state['estimate_gradient'].zero_()
                        state['estimate_gradient'].add_(
                            p.grad.data, alpha=1/len(train_loader.dataset))  # @NOTE exact average over data samples
                        if curr_round != 0 and sp.grad is not None:
                            state['estimate_gradient'].add_(
                                sp.grad.data, alpha=-(1-group['alpha'])/len(train_loader.dataset))   # @NOTE exact average over data samples

        if weight is None:
            world_size = float(communicator.get_world_size())
            for group in self.param_groups:
                for _, p in enumerate(group['params']):
                    state = self.state[p]
                    communicator.all_reduce(state['estimate_gradient'])
                    state['estimate_gradient'] /= world_size
                    if curr_round != 0:
                        state['estimate_gradient'].add_(
                            state['old_estimate_gradient'], alpha=1-group['alpha'])
        else:
            for group in self.param_groups:
                for _, p in enumerate(group['params']):
                    state = self.state[p]
                    state['estimate_gradient'] *= weight
                    communicator.all_reduce(state['estimate_gradient'])
                    if curr_round != 0:
                        state['estimate_gradient'].add_(
                            state['old_estimate_gradient'], alpha=1-group['alpha'])

        for group in self.param_groups:
            for p, sp in zip(group['params'], group['snapshot_params']):
                state = self.state[p]
                sp.data.copy_(p.data)
                state['old_estimate_gradient'].copy_(
                    state['estimate_gradient'])
