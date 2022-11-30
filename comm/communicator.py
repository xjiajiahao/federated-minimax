import os
import torch
import torch.distributed as dist
import numpy as np
import threading
import datetime

class SyncLock:

    def __init__(self, thread_num, backend):
        self.condition = threading.Condition()
        self.count = 0
        self.thread_num = thread_num
        self.backend = backend

    def acquire(self):
        self.condition.acquire()

    def release(self):
        self.condition.release()

    def __seed_reset_execute(self, to_execute):
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if self.backend == 'nccl':
            torch_cuda_state = torch.cuda.get_rng_state()
        to_execute()
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if self.backend == 'nccl':
            torch.cuda.set_rng_state(torch_cuda_state)

    def synchronize(self, start_execute=None, main_execute=None, end_execute=None):
        self.count += 1
        if self.count == 1 and start_execute is not None:
            start_execute()
        if main_execute is not None:
            main_execute()
        if self.count < self.thread_num:
            self.__seed_reset_execute(self.condition.wait)
        else:
            self.count = 0
            if end_execute is not None:
                end_execute()
            self.__seed_reset_execute(self.condition.notify_all)


class Communicator:
    """ The communication handler.

    Arguments:
        main_addr: the IP address of the central node
        requires_thread_manager (bool, default: False): whether to make one of the threads the coordinator, which acts like a server (only applicable for CODASCA in the multithreading case), or to treat all threads equally (for other algorithms)
    """

    def __init__(self, device, main_addr, target, backend='nccl', thread_num=1, requires_thread_coordinator=False):
        os.environ['MASTER_ADDR'] = main_addr
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        # nccl is a distributed communication backend
        dist.init_process_group(backend, timeout=datetime.timedelta(0, 60))
        self.backend = backend
        self.use_gpu = False
        if device == 'gpu':
            device = 'cpu'
            self.use_gpu = True
        self.world_size = dist.get_world_size()
        self.group = dist.new_group(range(self.world_size))
        self.rank = dist.get_rank()
        self.root_rank = 0  # set the id of the root process to zero
        self.thread_num = thread_num
        self.requires_thread_coordinator = requires_thread_coordinator
        if self.thread_num == 1 and self.requires_thread_coordinator:
            raise ValueError('thread_num must be larger than 1 when requires_thread_coordinator == True')

        if self.requires_thread_coordinator:
            self.client_lock = {i : threading.Condition() for i in range(1, self.thread_num)} 
            self.sync_tensor_lock = threading.Condition()
            self.sync_thread_flag = [0 for _ in range(thread_num)]
            self.sync_thread_lock = threading.Condition()
            if self.use_gpu:
                # let us do sth
                num_cuda_devices = torch.cuda.device_count()
                self.gpu_arr = np.zeros(num_cuda_devices, dtype=int)
                self.gpu_lock = threading.Condition()
                self.gpu_semaphore = threading.Semaphore(num_cuda_devices)
        else:
            client_list = list(set(range(self.world_size)) - set([self.root_rank]))
            self.p2p_group_dict = {i:dist.new_group([self.root_rank, i]) for i in client_list}
            self.sync_lock = SyncLock(thread_num, self.backend)
        # threads communication
        self.target = target
        self.thread_pool = []
        if self.requires_thread_coordinator:
            self.activate_thread = [0 for _ in range(thread_num)]
        else:
            self.activate_thread = [1 for _ in range(thread_num)]
        self.activate_thread_num = torch.zeros(1, device=device)
        self.sync_tensor = None
        self.break_signal = False
        print('process intilized, rank: {:d}/{:d}.'.format(self.rank, self.world_size))
        # @DEBUG adding the following code makes things work
        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        # print('Rank ', self.rank, ' has data ', tensor[0])
        dist.barrier()

    def is_root(self):
        """ Determine whether the current node is the root
        """
        ret = True if self.root_rank == self.rank and self.is_process_root() else False
        return ret

    def is_process_root(self):
        return int(threading.current_thread().name) == 0

    def get_thread_id(self):
        return int(threading.current_thread().name)

    def get_rank(self):
        """ Get the rank of the current node
        """
        rank = 0
        if self.requires_thread_coordinator:
            if self.is_root():
                rank = self.root_rank
            elif self.is_process_root():
                rank = self.rank + self.world_size * (self.thread_num - 1)
            else:
                rank = (int(threading.current_thread().name) - 1) * self.world_size + self.rank + 1
        else:
            rank = int(threading.current_thread().name) * self.world_size + self.rank
        return rank

    def get_world_size(self):
        """ Get the number of available nodes
        """
        def compute_activate_thread_num():
            self.activate_thread_num.copy_(torch.tensor([sum(self.activate_thread)]))
            dist.all_reduce(self.activate_thread_num, op=dist.ReduceOp.SUM, group=self.group)

        if self.requires_thread_coordinator:
            world_size = self.world_size
        else:
            self.sync_lock.synchronize(end_execute=compute_activate_thread_num)
            world_size = int(self.activate_thread_num.item())

        return world_size

    def is_activate(self):
        return bool(self.activate_thread[int(threading.current_thread().name)])

    def set_activity(self, activity, clients_to_activate=None):
        if clients_to_activate is None:
            self.activate_thread[int(threading.current_thread().name)] = activity
        else:
            for i in range(1, self.thread_num):
                if (i - 1) * self.world_size + self.rank + 1 in clients_to_activate:
                    self.activate_thread[i] = activity

    def all_reduce(self, tensor_var):
        """ All_reduce with SUM operation

        Arguments:
            tensor_var: the tensor to be reduced
        """
        def initialize_sync_tensor():
            self.sync_tensor = torch.zeros_like(tensor_var)
        
        def threads_reduce():
            self.sync_tensor.add_(tensor_var.data)

        def processes_reduce():
            # print(tensor_var)  # DEBUG
            dist.all_reduce(self.sync_tensor, op=dist.ReduceOp.SUM, group=self.group)
            dist.barrier()  # DEBUG

        self.sync_lock.synchronize(start_execute=initialize_sync_tensor, main_execute=threads_reduce if self.is_activate() else None, end_execute=processes_reduce)

        def distribute_tensors():
            tensor_var.data.copy_(self.sync_tensor)

        self.sync_lock.synchronize(main_execute=distribute_tensors)

    def set_break(self, is_break):
        self.break_signal = is_break

    def check_break(self):
        self.sync_lock.synchronize()
        return self.break_signal

    def acquire(self):
        if self.requires_thread_coordinator:
            if self.is_process_root():
                for i in range(1, self.thread_num):
                    self.client_lock[i].acquire()
            else:
                thread_id = self.get_thread_id()
                self.client_lock[thread_id].acquire()
                while self.sync_thread_flag[thread_id] == 0:
                    self.client_lock[thread_id].notify_all()
                    self.client_lock[thread_id].wait()

        else:
            self.sync_lock.acquire()

    def release(self):
        if self.requires_thread_coordinator:
            if self.is_process_root():
                for i in range(1, self.thread_num):
                    self.client_lock[i].release()
            else:
                thread_id = self.get_thread_id()
                try:
                    self.client_lock[thread_id].release()
                except:
                    pass
        else:
            self.sync_lock.release()

    def threads_start(self):
        for i in range(self.thread_num):
            thread = threading.Thread(name=str(i), target=self.target)
            self.thread_pool.append(thread)
            # thread.daemon = True
            thread.start()

    def threads_join(self):
        if __debug__:
            count = 0
        for thread in self.thread_pool:
            thread.join()
            if __debug__:
                print('[{:s}] joined thread {:d}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count), flush=True)
                count += 1
        if __debug__:
            print('[{:s}] joined all threads'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), flush=True)

    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this function, if async_op is False, or if async work handle is called on wait().
        """
        dist.barrier(self.group)
    
    def broadcast(self, tensor_var, client_subset, p2p_flag=False, to_execute=None):
        """ Broadcast tensor_var to sampled clients

        Arguments:
            tensor_var: the tensor to be reduced
            client_subset: the list of sampled clients (only applicable when self.requires_thread_coordinator is False)
        """
        if self.requires_thread_coordinator:
            if self.is_process_root():
                # broadcast first
                self.sync_tensor = tensor_var
                dist.broadcast(tensor_var, self.root_rank, group=self.group)
                if to_execute is not None:
                    to_execute(tensor_var)
                # notify the subset of clients to copy tensor
                for i in range(1, self.thread_num):
                    if self.activate_thread[i] > 0:
                        self.sync_thread_lock.acquire()
                        # if __debug__:
                        #     print('[{:s}] root acquired sync_thread_lock'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        self.sync_thread_flag[i] = 1
                        self.sync_thread_lock.notify_all()
                        self.sync_thread_lock.release()

                        self.client_lock[i].notify_all()
                        self.client_lock[i].release()
                for i in range(1, self.thread_num):
                    if self.activate_thread[i] > 0:
                        # if __debug__:
                        #     if torch.numel(tensor_var) == 502:
                        #         print('[{:s}] root acquiring client_lock[{:d}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i), flush=True)
                        self.client_lock[i].acquire()
                        # if __debug__:
                        #     if torch.numel(tensor_var) == 502:
                        #         print('[{:s}] root acquired client_lock[{:d}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i), flush=True)
                        while self.sync_thread_flag[i] != 0:
                            # if __debug__:
                            #     if torch.numel(tensor_var) == 502:
                            #         print('[{:s}] root reacquiring client_lock[{:d}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i), flush=True)
                            self.client_lock[i].notify_all()
                            self.client_lock[i].wait()
            else:
                # copy data and wait
                thread_id = self.get_thread_id()
                # if __debug__:
                #     if torch.numel(tensor_var) == 502:
                #         print('[{:s}] thread {:d} attempting to acquire client_lock'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), thread_id), flush=True)
                self.client_lock[thread_id].acquire()
                # if __debug__:
                #     if torch.numel(tensor_var) == 502:
                #         print('[{:s}] thread {:d} acquired client_lock'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), thread_id), flush=True)
                while self.sync_thread_flag[thread_id] == 0: # not ready to work
                    # if __debug__:
                    #     if torch.numel(tensor_var) == 502:
                    #         print('[{:s}] thread {:d} reacquiring client_lock'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), thread_id), flush=True)
                    self.client_lock[thread_id].notify_all()
                    self.client_lock[thread_id].wait()
                tensor_var.data.copy_(self.sync_tensor)
                self.sync_thread_lock.acquire()
                # if __debug__:
                #     if torch.numel(tensor_var) == 502:
                #         print('[{:s}] thread {:d} acquired thread_lock'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), thread_id), flush=True)
                self.sync_thread_flag[thread_id] = 0
                self.sync_thread_lock.notify_all()
                self.sync_thread_lock.release()

                self.client_lock[thread_id].notify_all()
                self.client_lock[thread_id].release()
        else:
            if self.world_size == len(client_subset) + 1 and p2p_flag == False:
                dist.broadcast(tensor_var, self.root_rank, group=self.group)
            else:
                reqs = []
                for curr_client in client_subset:
                    curr_group = self.p2p_group_dict[curr_client]
                    curr_req = dist.broadcast(tensor_var, self.root_rank, group=curr_group, async_op=True)
                    reqs.append(curr_req)
        
                for req in reqs:
                    req.wait()

    def reduce(self, tensor_var, client_subset):
        """ Aggregate tensor_var from sampled clients

        Arguments:
            tensor_var: the tensor to be reduced
            client_subset: the list of sampled clients
        """
        if self.requires_thread_coordinator:
            if self.is_process_root():
                self.sync_tensor = tensor_var
                # wait for clients to compute the sum of tensors
                for i in range(1, self.thread_num):
                    if self.activate_thread[i] > 0:
                        self.sync_thread_lock.acquire()
                        self.sync_thread_flag[i] = 1
                        self.sync_thread_lock.notify_all()
                        self.sync_thread_lock.release()

                        self.client_lock[i].notify_all()
                        self.client_lock[i].release()
                for i in range(1, self.thread_num):
                    if self.activate_thread[i] > 0:
                        self.client_lock[i].acquire()
                        while self.sync_thread_flag[i] != 0:
                            self.client_lock[i].notify_all()
                            self.client_lock[i].wait()
                dist.reduce(tensor_var, self.root_rank, group=self.group)
                # notify the subset of clients to copy tensor
            else:
                # copy data and wait
                thread_id = self.get_thread_id()
                self.client_lock[thread_id].acquire()
                while self.sync_thread_flag[thread_id] == 0:
                    self.client_lock[thread_id].notify_all()
                    self.client_lock[thread_id].wait()
                self.sync_tensor_lock.acquire()
                self.sync_tensor.add_(tensor_var.data)
                self.sync_tensor_lock.notify_all()
                self.sync_tensor_lock.release()

                self.sync_thread_lock.acquire()
                self.sync_thread_flag[thread_id] = 0
                self.sync_thread_lock.notify_all()
                self.sync_thread_lock.release()

                self.client_lock[thread_id].notify_all()
                self.client_lock[thread_id].release()
        else:
            # @NOTE we actually use point-to-point communication because the receivers are only aware of the server
            # synchronous implementation
            if self.world_size == len(client_subset) + 1:
                dist.reduce(tensor_var, self.root_rank, group=self.group)
            else:
                for curr_client in client_subset:
                    curr_group = self.p2p_group_dict[curr_client]
                    dist.reduce(tensor_var, self.root_rank, group=curr_group)


    def get_device(self):
        if self.use_gpu:
            self.gpu_semaphore.acquire()
            self.gpu_lock.acquire()
            cuda_id = -1
            for i in range(len(self.gpu_arr)):
                if self.gpu_arr[i] == 0:
                    self.gpu_arr[i] = 1
                    cuda_id = i
                    break
            self.gpu_lock.notify_all()
            self.gpu_lock.release()
            device = torch.device(torch.device("cuda:{}".format(cuda_id)))
        else:
            device = 'cpu'
        return device

    def release_device(self, device):
        if self.use_gpu:
            self.gpu_lock.acquire()
            self.gpu_lock.notify_all()
            self.gpu_arr[device.index] = 0
            self.gpu_lock.release()
            self.gpu_semaphore.release()
