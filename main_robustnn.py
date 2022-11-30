import argparse
from numpy.lib.type_check import _nan_to_num_dispatcher
import torch

import numpy as np

from model import MLP_784_200_200_10, RobustNN
from dataloader import FedMNISTLoader, MNISTLoader, FedFashionMNISTLoader, FashionMNISTLoader
from algorithm import CDMA_ONE_wrapper, CDMA_ADA_wrapper, CDMA_NC_wrapper, Local_SGDA_Plus_wrapper, CODASCA_wrapper, CODA_Plus_wrapper, Catalyst_Scaffold_S_wrapper, Extra_Step_Local_SGD_wrapper, CODASCA_Threading_wrapper
from comm import Communicator


torch.backends.cudnn.deterministic = True  # fix the random seed of cudnn

# define input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--master_addr', type=str, default='127.0.0.1')
# provided by the launch utility in torch.distributed.launch
parser.add_argument('--local_rank', type=int, default=0)
# provided by the launch utility in torch.distributed.launch
parser.add_argument('--data_dir', type=str, default='./data/')
# provided by the launch utility in torch.distributed.launch
parser.add_argument('--output_dir', type=str, default='./data/results/')
parser.add_argument('--use_gpu', type=lambda x: (str(x).lower()
                                                 in ['true', '1', 'yes']), default=True)
parser.add_argument('--algorithm_to_run', type=str, default='CDMA_ONE')
parser.add_argument('--dataset_name', type=str, default='MNIST')
parser.add_argument('--model_reg_coef', type=float, default=0.001)
parser.add_argument('--num_partitions', type=int)
parser.add_argument('--num_nodes', type=int)
parser.add_argument('--num_rounds', type=int)
parser.add_argument('--num_local_iterations', type=int)
parser.add_argument('--primal_step_size', type=float,
                    default=0.1)  # primal step size
parser.add_argument('--dual_step_size', type=float,
                    default=0.1)  # dual step size
parser.add_argument('--step_size_exp', type=float,
                    default=0.3333)  # step size exponent
parser.add_argument('--primal_alpha', type=float, default=0.5)
parser.add_argument('--dual_alpha', type=float, default=0.5)
parser.add_argument('--alpha_exp', type=float,
                    default=0.6667)  # step size exponent
parser.add_argument('--resample_flag', type=lambda x: (str(x).lower()
                                                 in ['true', '1', 'yes']), default=False)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--pretrained', type=lambda x: (str(x).lower()
                                                    in ['true', '1', 'yes']), default=True)
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--random_seed_id', type=int, default=1234)
parser.add_argument('--sort_by', type=str, default=None)
parser.add_argument('--similarity', type=float, default=0.1)
parser.add_argument('--max_machine_drop_ratio', type=float, default=0.0)
# the following arguments are required by CODASCA/Catalyst_Scaffold_S
parser.add_argument('--local_step_size', type=float, default=1.0)
parser.add_argument('--global_step_size', type=float, default=1.0)
parser.add_argument('--algorithm_reg_coef', type=float, default=1.0)
parser.add_argument('--T0', type=int, default=2000)
parser.add_argument('--model_num_epochs_eval', type=int, default=20)
parser.add_argument('--model_step_size_eval', type=float, default=1.0)
parser.add_argument('--model_is_deterministic_eval', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)

args = parser.parse_args()

MAIN_ADDR = args.master_addr
LOCAL_PROCESS_RANK = args.local_rank
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
USE_GPU = args.use_gpu
algorithm_to_run = args.algorithm_to_run
num_partitions = args.num_partitions
num_nodes = args.num_nodes
num_rounds = args.num_rounds
num_local_iterations = args.num_local_iterations
print_freq = args.print_freq
primal_step_size = args.primal_step_size
dual_step_size = args.dual_step_size
step_size_exp = args.step_size_exp
primal_alpha = args.primal_alpha
dual_alpha = args.dual_alpha
alpha_exp = args.alpha_exp
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
num_threads = args.num_threads  # oversubscribe
random_seed_id = args.random_seed_id
dataset_name = args.dataset_name
sort_by = args.sort_by
similarity = args.similarity
local_step_size = args.local_step_size
global_step_size = args.global_step_size
algorithm_reg_coef = args.algorithm_reg_coef
model_reg_coef = args.model_reg_coef
T0 = args.T0
resample_flag = args.resample_flag
model_num_epochs_eval = args.model_num_epochs_eval
model_step_size_eval = args.model_step_size_eval
model_is_deterministic_eval = args.model_is_deterministic_eval
max_machine_drop_ratio = args.max_machine_drop_ratio

def thread_main():
    communicator.acquire()

    torch.manual_seed(random_seed_id)
    torch.cuda.manual_seed(random_seed_id)
    np.random.seed(random_seed_id)

    # Step 2. prepare the dataset, the clients share the same random number seed
    # in each round, each client samples a subset of data
    net = None
    data_loader = None
    if dataset_name == 'MNIST':
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if USE_GPU:
            torch_cuda_state = torch.cuda.get_rng_state()
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        net = MLP_784_200_200_10()
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if USE_GPU:
            torch.cuda.set_rng_state(torch_cuda_state)

        if algorithm_to_run == 'CODASCA' or num_nodes == num_partitions:
            data_loader = FedMNISTLoader(num_partitions,
                                           DATA_DIR, train_batch_size,
                                           test_batch_size, is_multiclass=True,
                                           device=device, sort_by=sort_by,
                                           similarity=similarity)
        else:
            data_loader = MNISTLoader(num_partitions,
                                      DATA_DIR, train_batch_size,
                                      test_batch_size, is_multiclass=True,
                                      device=device, sort_by=sort_by,
                                      similarity=similarity)
    elif dataset_name == 'FashionMNIST':
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if USE_GPU:
            torch_cuda_state = torch.cuda.get_rng_state()
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        net = MLP_784_200_200_10()
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if USE_GPU:
            torch.cuda.set_rng_state(torch_cuda_state)

        if algorithm_to_run == 'CODASCA' or num_nodes == num_partitions:
            data_loader = FedFashionMNISTLoader(num_partitions,
                                           DATA_DIR, train_batch_size,
                                           test_batch_size, is_multiclass=True,
                                           device=device, sort_by=sort_by,
                                           similarity=similarity)
        else:
            data_loader = FashionMNISTLoader(num_partitions,
                                      DATA_DIR, train_batch_size,
                                      test_batch_size, is_multiclass=True,
                                      device=device, sort_by=sort_by,
                                      similarity=similarity)
    else:
        raise Exception(
            'The "{:s}" dataset is not supported yet.'.format(dataset_name))

    # Step 3. initialize the model
    net = net.to(device)
    model = RobustNN(net, data_loader.get_feature_shape(),
                     model_reg_coef, device, num_epochs_eval=model_num_epochs_eval,
                     step_size_eval=model_step_size_eval, is_deterministic_eval=model_is_deterministic_eval)

    # Step 4. run the CDMA_ONE algorithm
    if algorithm_to_run == 'CDMA_ONE':
        CDMA_ONE_wrapper(model, communicator, data_loader,
                           num_partitions, num_nodes, num_rounds, num_local_iterations,
                           primal_step_size, dual_step_size, train_batch_size,
                           resample_flag=resample_flag,
                           max_machine_drop_ratio=max_machine_drop_ratio,
                           device=device,
                           print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'CDMA_ADA':
        CDMA_ADA_wrapper(model, communicator, data_loader,
                              num_partitions, num_nodes, num_rounds, num_local_iterations,
                              primal_step_size, dual_step_size, step_size_exp,
                              primal_alpha, dual_alpha, alpha_exp,
                              train_batch_size,
                              resample_flag=resample_flag,
                              max_machine_drop_ratio=max_machine_drop_ratio,
                              device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'CDMA_NC':
        CDMA_NC_wrapper(model, communicator, data_loader,
                       num_partitions, num_nodes, num_rounds, num_local_iterations,
                       primal_step_size, dual_step_size, train_batch_size,
                       max_machine_drop_ratio=max_machine_drop_ratio,
                       device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'Local_SGDA_Plus':
        Local_SGDA_Plus_wrapper(model, communicator, data_loader,
                            num_partitions, num_nodes, num_rounds, num_local_iterations,
                            primal_step_size, dual_step_size, train_batch_size,
                            max_machine_drop_ratio=max_machine_drop_ratio,
                            device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'CODASCA':
        if num_threads != 1:
            CODASCA_Threading_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size,
                            global_step_size, algorithm_reg_coef, train_batch_size,
                            max_machine_drop_ratio=max_machine_drop_ratio,
                            device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
        else:
            CODASCA_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size,
                            global_step_size, algorithm_reg_coef, train_batch_size,
                            max_machine_drop_ratio=max_machine_drop_ratio,
                            device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'CODA_Plus':
        CODA_Plus_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0,
                          local_step_size, algorithm_reg_coef, train_batch_size,
                          max_machine_drop_ratio=max_machine_drop_ratio,
                          device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'Catalyst_Scaffold_S':
        Catalyst_Scaffold_S_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations, T0, local_step_size,
                            global_step_size, algorithm_reg_coef, train_batch_size,
                            max_machine_drop_ratio=max_machine_drop_ratio,
                            device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    elif algorithm_to_run == 'Extra_Step_Local_SGD':
        Extra_Step_Local_SGD_wrapper(model, communicator, data_loader, num_partitions, num_nodes, num_rounds, num_local_iterations,
                                  local_step_size, train_batch_size,
                                  max_machine_drop_ratio=max_machine_drop_ratio,
                                  device=device, print_freq=print_freq, OUTPUT_DIR=OUTPUT_DIR)
    else:
        raise ValueError('the value of algorithm_to_run, "' +
                         algorithm_to_run + '", is invalid.')

    communicator.release()


device = "cpu"
backend = 'gloo'
if USE_GPU is True:
    if torch.cuda.is_available():
        backend = 'nccl'
        device = torch.device("cuda:{}".format(LOCAL_PROCESS_RANK))
    else:
        print("[Warning] no cuda device available, use cpu instead")

# Step 1. initialize the distributed environment and the communicator
requires_thread_coordinator = False
if algorithm_to_run == 'CODASCA' and num_threads != 1:
    requires_thread_coordinator = True

communicator = Communicator(
    device, MAIN_ADDR, target=thread_main, backend=backend, thread_num=num_threads, requires_thread_coordinator=requires_thread_coordinator)

communicator.threads_start()
communicator.threads_join()
