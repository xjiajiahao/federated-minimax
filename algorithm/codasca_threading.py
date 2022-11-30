import os
import copy
from random import random
import torch
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
if __debug__:
    from datetime import datetime


def CODASCA_Threading_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, global_step_size, regularizer_coef, train_batch_size, num_stages=5000, max_machine_drop_ratio=0.0, device='cpu', print_freq=10, OUTPUT_DIR='./data/results/', random_seed_id=1234):
    """Run the CODASCA algorithm.

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
        global_step_size: the global step size
        regularizer_coef: the coefficient of the dual regularizer
        train_batch_size: the training batch size
        num_stages: the number of stages
        max_machine_drop_ratio: the maximum drop ratio of selected machines
        device: the device to use (cpu or cuda)
        print_freq: the frequency of printing results
        OUTPUT_DIR: the directory to write output file
    """

    def wakeup_clients(signal_tensor):
        # wake up the selected clients
        curr_clients = signal_tensor[2:num_partitions+2]
        curr_clients = (curr_clients[curr_clients > 0]).tolist()
        communicator.set_activity(1, curr_clients)

    # initialize the optimization algorithm
    rank = communicator.get_rank()
    clients_list = list(range(1, num_partitions + 1))

    configs = 'CODASCA_partitions_{:d}_nodes_{:d}_rounds_{:d}_local_iters_{:d}_T0_{:d}_local_step_{:f}_global_step_{:f}_reg_coef_{:f}_train_batch_size_{:d}_drop_ratio_{:f}'.format(
        num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, global_step_size, regularizer_coef, train_batch_size, max_machine_drop_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_name = OUTPUT_DIR + configs

    # set to None to mute warning
    test_loader = None

    # there is a root node and num_partitions worker nodes
    if communicator.is_root():
        print('running {:s}, # partitions: {:d}, # nodes: {:d}, # rounds: {:d}, # local iterations: {:d}, T0: {:d}, local step size: {:f}, global step size: {:f}, reg coef: {:f}, train batch size: {:d}, max drop ratio: {:f}'.format(
            'CODASCA', num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size, global_step_size, regularizer_coef, train_batch_size, max_machine_drop_ratio))
        # initialization
        old_model = copy.deepcopy(model)
        snapshot_model = copy.deepcopy(model)
        agg_grad = copy.deepcopy(model)
        agg_grad.zero_param()

        test_loader = data_loader.build_test_loader()
        model.logger_init(output_file_name)

        # print statistics
        curr_received_doubles = 0
        model.evaluate_and_log(0, curr_received_doubles, test_loader)

        optimizer = CODASCA(model.get_variable_lists(), old_model.get_variable_lists(), snapshot_model.get_variable_lists(
        ), agg_grad.get_variable_lists(), global_step_size, regularizer_coef, model.projection())

        count_total_rounds = 0
        early_exit_flag = False
        num_rounds_per_stage = int(T0 / num_local_iterations)
        signal_tensor = torch.zeros(
            num_partitions + 2, dtype=int, device='cpu')
        update_snapshot_flag = False
        # the main loop
        np_rng = np.random.default_rng(random_seed_id)  # for thread safety
        for curr_stage in range(num_stages):
            # update the proximal point
            snapshot_model.copy_(model)
            # Run T0 iterations
            for tmp_round in range(num_rounds_per_stage):
                # 1. sample clients and send a signal to active clients
                actual_num_nodes = num_nodes - math.floor(np_rng.random() * max_machine_drop_ratio * num_nodes)
                if actual_num_nodes < num_partitions:
                    candidate_clients = (np_rng.choice(
                        num_partitions, num_nodes, replace=False) + 1)
                    curr_clients = np.sort(np_rng.choice(
                        candidate_clients, actual_num_nodes, replace=False)).tolist()
                else:
                    curr_clients = clients_list
                # if __debug__:
                #     print('[{:s}] round {:d}, curr_clients: {:}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tmp_round, curr_clients), flush=True)

                signal_tensor.zero_()
                signal_tensor[0] = curr_stage
                signal_tensor[1] = update_snapshot_flag
                signal_tensor[2:actual_num_nodes+2] = torch.as_tensor(curr_clients)
                # if __debug__:
                #     print('[{:s}] root sent signal'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank), flush=True)

                communicator.broadcast(
                    signal_tensor, curr_clients, to_execute=wakeup_clients)

                # 2. send messages to the sampled clients
                old_model.copy_(model)
                model.broadcast(curr_clients, communicator)
                if (num_nodes < num_partitions or max_machine_drop_ratio != 0) and update_snapshot_flag == False:
                    snapshot_model.broadcast(curr_clients, communicator)
                agg_grad.broadcast(curr_clients, communicator)
                # 3. receive messages from the sampled clients
                model.zero_param()
                model.reduce(curr_clients, communicator)
                if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                    # agg_grad <- agg_grad + 1/n \sum_{i \in S} (agg_grad_new_{i} - agg_grad_old_{i})
                    agg_grad.reduce(curr_clients, communicator,
                                    average_flag=False)
                else:
                    agg_grad.zero_param()
                    agg_grad.reduce(curr_clients, communicator)

                communicator.set_activity(0, curr_clients)

                # 4. take a global step
                optimizer.global_step()
                update_snapshot_flag = False
                # exit if the #rounds of communication attains some value
                count_total_rounds += 1
                # print statistics
                if count_total_rounds % print_freq == 0:
                    curr_received_doubles = actual_num_nodes * model.get_num_parameters() * 2
                    too_bad_flag = model.evaluate_and_log(
                        count_total_rounds, curr_received_doubles, test_loader)
                    if too_bad_flag:
                        early_exit_flag = True
                        print(
                            "The performance of training is so bad that we stop the training.")
                        break
                if count_total_rounds >= num_rounds:
                    break
            if count_total_rounds >= num_rounds or early_exit_flag:
                break

        # send exit signal to all clients
        signal_tensor.zero_()
        signal_tensor[0] = -1
        signal_tensor[2:num_partitions+2] = torch.as_tensor(clients_list)
        communicator.broadcast(signal_tensor, clients_list,
                               to_execute=wakeup_clients)
        communicator.set_activity(0, clients_list)
        # # finally, print statistics again
        # if not early_exit_flag:  # normal exit
        #     curr_received_doubles = num_nodes * model.get_num_parameters() * 2
        #     model.evaluate_and_log(
        #         num_rounds, curr_received_doubles, test_loader)

        # wait until all processes complete
        communicator.barrier()
        print('Rank 0 Finished Training.\a')

    elif communicator.is_process_root():  # for coordinators
        signal_tensor = torch.zeros(
            num_partitions + 2, dtype=int, device='cpu')
        model.zero_param()  # temporarily store the local old gradient
        curr_clients = [rank]
        while True:
            # Step 1. wait for signal
            communicator.broadcast(
                signal_tensor, curr_clients, to_execute=wakeup_clients)
            curr_stage = signal_tensor[0].item()
            update_snapshot_flag = bool(signal_tensor[1].item())
            # exit if signal < 0
            if curr_stage < 0:
                communicator.set_activity(0, clients_to_deactivate)
                break
            # Step 2. initialize model and optimizer
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                model.broadcast(curr_clients, communicator)
                if num_nodes < num_partitions and update_snapshot_flag == False:
                    model.broadcast(curr_clients, communicator)
            else:
                # receive the global model
                model.broadcast(curr_clients, communicator)
            model.broadcast(curr_clients, communicator)
            model.zero_param()
            model.reduce(curr_clients, communicator)
            model.zero_param()
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                model.reduce(curr_clients, communicator,
                             average_flag=False)
            else:
                model.reduce(curr_clients, communicator)
            clients_to_deactivate = signal_tensor[2:num_partitions+2]
            clients_to_deactivate = (
                clients_to_deactivate[clients_to_deactivate > 0]).tolist()
            communicator.set_activity(0, clients_to_deactivate)

        # wait until all processes complete
        print('Rank {:d} Finished Training.\a'.format(rank))
        communicator.barrier()

    else:  # for clients
        signal_tensor = torch.zeros(
            num_partitions + 2, dtype=int, device='cpu')
        if num_nodes < num_partitions or max_machine_drop_ratio != 0:
            model.zero_param()  # temporarily store the local old gradient
            curr_clients = [rank]
        else:
            # agg_grad stores the local old gradient
            agg_grad = copy.deepcopy(model)
            agg_grad.zero_param()
            old_agg_grad = copy.deepcopy(agg_grad)
            # make a copy of the local gradient
            old_model = copy.deepcopy(model)
            snapshot_model = copy.deepcopy(model)
            curr_train_loader = data_loader.build_train_loader(rank-1)
            optimizer = CODASCA(model.get_variable_lists(), old_model.get_variable_lists(), snapshot_model.get_variable_lists(
            ), agg_grad.get_variable_lists(), local_step_size, regularizer_coef, model.projection())  # initialize the optimizer
            curr_clients = clients_list
        # if __debug__:
        #     print('[{:s}] rank: {:d} initialized'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank), flush=True)
        while True:
            # Step 1. wait for signal
            communicator.broadcast(signal_tensor, curr_clients)
            curr_stage = signal_tensor[0].item()
            update_snapshot_flag = bool(signal_tensor[1].item())
            # if __debug__:
            #     print('[{:s}] rank: {:d} got signal'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank), flush=True)
            # exit if signal < 0
            if curr_stage < 0:
                # if __debug__:
                #     print('[{:s}] rank: {:d} exiting'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank), flush=True)
                break
            # Step 2. initialize model and optimizer
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                # agg_grad stores the local old gradient
                agg_grad = copy.deepcopy(model)
                # make a copy of the local gradient
                old_agg_grad = copy.deepcopy(model)
                # make a copy of the local gradient
                old_model = copy.deepcopy(model)
                # receive the global model
                model.broadcast(curr_clients, communicator)
                old_model.copy_(model)
                snapshot_model = copy.deepcopy(model)
                if (num_nodes < num_partitions or max_machine_drop_ratio != 0) and update_snapshot_flag == False:
                    snapshot_model.broadcast(curr_clients, communicator)
                curr_train_loader = data_loader.build_train_loader(rank-1)
                optimizer = CODASCA(model.get_variable_lists(), old_model.get_variable_lists(), snapshot_model.get_variable_lists(
                ), agg_grad.get_variable_lists(), local_step_size, regularizer_coef, model.projection())  # initialize the optimizer
            else:
                # receive the global model
                model.broadcast(curr_clients, communicator)
                old_model.copy_(model)  # backup local old gradient
                old_agg_grad.copy_(agg_grad)
                if update_snapshot_flag:
                    snapshot_model.copy_(model)
            # if __debug__:
            #     print('[{:s}] rank: {:d} received model'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank), flush=True)
            # receive the global old gradient
            agg_grad.broadcast(curr_clients, communicator)
            # add the global old gradient to the local gradient
            agg_grad.add_(old_agg_grad, alpha=-1.0)
            # if __debug__:
            #     print('[{:s}] rank: {:d} received agg_grad'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank), flush=True)
            # move to gpu
            device = communicator.get_device()  # just wait for a device
            # if __debug__:
            #     print('[{:s}] rank: {:d} got device: {:}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank, device), flush=True)
            model.to(device)
            snapshot_model.to(device)
            old_model.to(device)
            agg_grad.to(device)
            old_agg_grad.to(device)
            for curr_local_iter in range(num_local_iterations):
                # sample a minibatch of local data
                try:
                    features, labels = train_iter.next()
                except:
                    train_iter = iter(curr_train_loader)
                    features, labels = train_iter.next()
                features = features.to(device)  # redefine device
                labels = labels.to(device)
                # if __debug__:
                #     print('rank: {:d}, local_iter: {:d}'.format(rank, curr_local_iter), flush=True)
                # # compute the gradients at the current variable
                optimizer.zero_grad()
                curr_loss = model.forward(features, labels)
                curr_loss.backward()
                optimizer.local_step(curr_stage)
            # update the local old gradient
            optimizer.update_local_old_grad(curr_stage, num_local_iterations)
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                agg_grad.add_(old_agg_grad, alpha=-1.0)
                agg_grad.multiply_(1.0/num_partitions)

            # return back the GPU
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                del optimizer
                del snapshot_model
                del curr_train_loader
            model.to('cpu')
            old_model.to('cpu')
            agg_grad.to('cpu')
            old_agg_grad.to('cpu')
            communicator.release_device(device)  # release the device
            # if __debug__:
            #     print('[{:s}] rank: {:d} released device: {:}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank, device), flush=True)

            # send the local model to the server
            model.reduce(curr_clients, communicator)
            # send the local old gradient to the server
            # @NOTE backup agg_grad before reduction because the "reduce" operation changes the value of agg_grad unintentionally
            old_model.copy_(agg_grad)
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                # agg_grad <- agg_grad + 1/n \sum_{i \in S} (agg_grad_new_{i} - agg_grad_old_{i})
                agg_grad.reduce(curr_clients, communicator,
                                average_flag=False)
            else:
                agg_grad.reduce(curr_clients, communicator)
            agg_grad.copy_(old_model)
            # store the local old gradient in model
            if num_nodes < num_partitions or max_machine_drop_ratio != 0:
                # delete other objects to save space
                model.zero_grad(set_to_none=True)
                agg_grad.multiply_(num_partitions)
                agg_grad.add_(old_agg_grad)
                model.copy_(agg_grad)  # model <- agg_grad to save space
                del agg_grad
                del old_agg_grad
                del old_model
            # finish one round of training

class CODASCA(Optimizer):
    """The CODASCA algorithm class.

    Arguments:
        model_var_tuple: (primal variable list, dual variable list)
            tuple of the main model
        old_model_var_tuple: (primal variable list, dual variable
            list) tuple of the old model
        snapshot_model_var_tuple: (primal variable list, dual variable
            list) tuple of the snapshot model (the proximal point)
        agg_grad_var_tuple: (primal variable list, dual variable
            list) tuple of the aggregated gradient
        step_size (float): the local/global step size
        regularizer_coef: the coefficient of the l2 regularizer
        projection (function handle): projection onto the constraint (default: None)
    """

    def __init__(self, model_var_tuple, old_model_var_tuple, snapshot_model_var_tuple, agg_grad_var_tuple, step_size=required, regularizer_coef=required, projection=None):
        if step_size is not required and step_size < 0.0:
            raise ValueError(
                "Invalid step size: {}".format(step_size))
        if regularizer_coef is not required and regularizer_coef < 0.0:
            raise ValueError(
                "Invalid regularizer coefficient: {}".format(regularizer_coef))
        # add main params to self.param_groups
        param_groups = [
            {'params': model_var_tuple[0], 'old_params': old_model_var_tuple[0], 'agg_grad': agg_grad_var_tuple[0],
                'snapshot_params': snapshot_model_var_tuple[0], 'step_size': step_size, 'regularizer_coef': regularizer_coef, 'primal_flag': True},
            {'params': model_var_tuple[1], 'old_params': old_model_var_tuple[1], 'snapshot_params': snapshot_model_var_tuple[1], 'agg_grad': agg_grad_var_tuple[1], 'step_size': step_size, 'primal_flag': False}]
        defaults = {'step_size': required}
        super(CODASCA, self).__init__(param_groups, defaults)
        self.projection = projection

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

    def local_step(self, curr_stage):
        """Take a local update step on both the primal and dual variables.

        Arguments:
            curr_stage: the step size decaying level
        """
        for group in self.param_groups:
            step_size = group['step_size'] / 3**curr_stage
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                sp = group['snapshot_params'][idx]
                agg_g = group['agg_grad'][idx]
                d_p = p.grad.data
                # d_p.add_(agg_g.data)   # @TODO uncomment this line
                if group['primal_flag']:
                    d_p.add_(agg_g.data)   # @TODO comment this line
                    d_p.add_(p.data, alpha=group['regularizer_coef'])
                    d_p.add_(sp.data, alpha=-group['regularizer_coef'])
                    p.data.add_(d_p, alpha=-step_size)
                else:
                    p.data.add_(d_p, alpha=step_size)
        self.projection()

    def global_step(self):
        """Take a global update step on both the primal and dual variables.
        """
        for group in self.param_groups:
            step_size = group['step_size']
            for idx, p in enumerate(group['params']):
                old_p = group['old_params'][idx]
                p.data.multiply_(step_size)
                p.data.add_(old_p.data, alpha=1-step_size)

    def update_local_old_grad(self, curr_stage, num_local_iterations):
        for group in self.param_groups:
            step_size = 1.0 / (num_local_iterations *
                               group['step_size'] / 3**curr_stage)
            for idx, p in enumerate(group['params']):
                old_p = group['old_params'][idx]
                agg_g = group['agg_grad'][idx]
                agg_g.data.multiply_(-1.0)
                if group['primal_flag']:
                    agg_g.data.add_(old_p.data, alpha=step_size)
                    agg_g.data.add_(p.data, alpha=-step_size)
                else:
                    agg_g.data.add_(old_p.data, alpha=-step_size)
                    agg_g.data.add_(p.data, alpha=step_size)
