#!/bin/bash
# input argument:
#   nic: the network interface card name (e.g., eno2)
#   machine_rank: the rank of the node (only applicable when using multiple nodes)

if [[ $# -ne 0 ]] ; then
    export GLOO_SOCKET_IFNAME=$1
    export NCCL_SOCKET_IFNAME=$1
fi
machine_rank=$2
random_seed_id=1234
export PYTHONOPTIMIZE=1  # remove the __debug__ code
export OMP_NUM_THREADS=1

master_addr='XXX.XXX.XXX.XXX'  # @NOTE replace this line with your central node's IP address
num_machines_base=2
num_machines_double=4
num_proc_per_node=4
latent_vector_len=100
image_size=64
num_partitions=500
# num_proc_per_node_codasca=5
# num_threads_codasca=101
# num_proc_per_node_codasca=1
# num_threads_codasca=501
num_proc_per_node_codasca=1
num_machines_codasca=1
num_threads_codasca=501
num_nodes_base=8
num_nodes_double=16
max_machine_drop_ratio=0.5
# num_rounds=800
# num_rounds=1000
num_rounds=900
# train_batch_size=30
# train_batch_size=82
# train_batch_size=164
# train_batch_size=110
train_batch_size=60
# test_batch_size=256
test_batch_size=100
dataset_name='CelebA'
print_freq=20
data_dir='./data/'
output_dir_prefix="${data_dir}results_gan_celeba_best_${random_seed_id}/"
output_dir_CDMA_ONE="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_MB_T_${num_rounds}/"
output_dir_CDMA_ADA="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_STORM_T_${num_rounds}/"
output_dir_CDMA_NC="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_Avg_T_${num_rounds}/"
output_dir_Local_SGDA_Plus="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_Avg_Plus_T_${num_rounds}/"
output_dir_Catalyst_Scaffold_S="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_Catalyst_Scaffold_S_T_${num_rounds}/"
output_dir_Extra_Step_Local_SGD="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_Avg_Extra_Step_T_${num_rounds}/"
output_dir_CODA_Plus="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_CODA_Plus_T_${num_rounds}/"
output_dir_CODASCA="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_CODASCA_T_${num_rounds}/"
output_dir_Momentum_Local_SGDA="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_Momentum_Local_SGDA_T_${num_rounds}/"
sort_by='label'
similarity=0.0


for num_local_iterations in 20
do
    for primal_step_size in 0.001
    do
        for dual_step_size in 0.01
        do
            python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
                --nnodes=${num_machines_double} \
                --node_rank=${machine_rank} \
                --master_addr=${master_addr} \
                --master_port=2500 \
                ./main_gan.py --master_addr=${master_addr} \
                --data_dir=${data_dir} \
                --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
                --use_gpu=True \
                --latent_vector_len=${latent_vector_len} \
                --image_size=${image_size} \
                --algorithm_to_run=CDMA_NC \
                --dataset_name=${dataset_name} \
                --num_partitions=${num_partitions} \
                --num_nodes=${num_nodes_double} \
                --num_rounds=${num_rounds} \
                --num_local_iterations=${num_local_iterations} \
                --primal_step_size=${primal_step_size} \
                --dual_step_size=${dual_step_size} \
                --print_freq=${print_freq} \
                --sort_by=${sort_by} \
                --similarity=${similarity} \
                --random_seed_id=${random_seed_id} \
                --train_batch_size=${train_batch_size} \
                --test_batch_size=${test_batch_size} \
                --max_machine_drop_ratio=${max_machine_drop_ratio}
        done
    done

    for primal_step_size in 0.001
    do
        for dual_step_size in 0.01
        do
            python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
                --nnodes=${num_machines_double} \
                --node_rank=${machine_rank} \
                --master_addr=${master_addr} \
                --master_port=2500 \
                ./main_gan.py --master_addr=${master_addr} \
                --data_dir=${data_dir} \
                --output_dir="${output_dir_Local_SGDA_Plus}K_${num_local_iterations}/" \
                --use_gpu=True \
                --latent_vector_len=${latent_vector_len} \
                --image_size=${image_size} \
                --algorithm_to_run=Local_SGDA_Plus \
                --dataset_name=${dataset_name} \
                --num_partitions=${num_partitions} \
                --num_nodes=${num_nodes_double} \
                --num_rounds=${num_rounds} \
                --num_local_iterations=${num_local_iterations} \
                --primal_step_size=${primal_step_size} \
                --dual_step_size=${dual_step_size} \
                --print_freq=${print_freq} \
                --sort_by=${sort_by} \
                --similarity=${similarity} \
                --random_seed_id=${random_seed_id} \
                --train_batch_size=${train_batch_size} \
                --test_batch_size=${test_batch_size} \
                --max_machine_drop_ratio=${max_machine_drop_ratio}
        done
    done

    for local_step_size in 0.01
    do
        global_step_size=${local_step_size}
        for algorithm_reg_coef in 0
        do
            if [ "${local_step_size}" == 0.001 ] && [ "${algorithm_reg_coef}" == 0 ]; then
                continue
            fi
            for T0 in 2000
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
                    --nnodes=${num_machines_base} \
                    --node_rank=${machine_rank} \
                    --master_addr=${master_addr} \
                    --master_port=2500 \
                    ./main_gan.py --master_addr=${master_addr} \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_Catalyst_Scaffold_S}K_${num_local_iterations}/" \
                    --use_gpu=True \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=Catalyst_Scaffold_S \
                    --dataset_name=${dataset_name} \
                    --num_partitions=${num_partitions} \
                    --num_nodes=${num_nodes_base} \
                    --num_rounds=${num_rounds} \
                    --num_local_iterations=${num_local_iterations} \
                    --local_step_size=${local_step_size} \
                    --global_step_size=${global_step_size} \
                    --algorithm_reg_coef=${algorithm_reg_coef} \
                    --T0=${T0} \
                    --print_freq=${print_freq} \
                    --sort_by=${sort_by} \
                    --similarity=${similarity} \
                    --random_seed_id=${random_seed_id} \
                    --train_batch_size=${train_batch_size} \
                    --test_batch_size=${test_batch_size} \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done    
    done

    for local_step_size in 0.01
    do
        for algorithm_reg_coef in 0
        do
            for T0 in 4000
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
                    --nnodes=${num_machines_double} \
                    --node_rank=${machine_rank} \
                    --master_addr=${master_addr} \
                    --master_port=2500 \
                    ./main_gan.py --master_addr=${master_addr} \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CODA_Plus}K_${num_local_iterations}/" \
                    --use_gpu=True \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CODA_Plus \
                    --dataset_name=${dataset_name} \
                    --num_partitions=${num_partitions} \
                    --num_nodes=${num_nodes_double} \
                    --num_rounds=${num_rounds} \
                    --num_local_iterations=${num_local_iterations} \
                    --local_step_size=${local_step_size} \
                    --algorithm_reg_coef=${algorithm_reg_coef} \
                    --T0=${T0} \
                    --print_freq=${print_freq} \
                    --sort_by=${sort_by} \
                    --similarity=${similarity} \
                    --random_seed_id=${random_seed_id} \
                    --train_batch_size=${train_batch_size} \
                    --test_batch_size=${test_batch_size} \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done

done

for num_local_iterations in 10
do
    for local_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
            --nnodes=${num_machines_double} \
            --node_rank=${machine_rank} \
            --master_addr=${master_addr} \
            --master_port=2500 \
            ./main_gan.py --master_addr=${master_addr} \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_Extra_Step_Local_SGD}K_${num_local_iterations}/" \
            --use_gpu=True \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=Extra_Step_Local_SGD \
            --dataset_name=${dataset_name} \
            --num_partitions=${num_partitions} \
            --num_nodes=${num_nodes_double} \
            --num_rounds=${num_rounds} \
            --num_local_iterations=${num_local_iterations} \
            --local_step_size=${local_step_size} \
            --print_freq=${print_freq} \
            --sort_by=${sort_by} \
            --similarity=${similarity} \
            --random_seed_id=${random_seed_id} \
            --train_batch_size=${train_batch_size} \
            --test_batch_size=${test_batch_size} \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done




for num_local_iterations in 25
do
    for primal_step_size in 0.01
    do
        for dual_step_size in 0.01
        do
            for step_size_exp in 0
            do
                alpha_exp=$(echo $step_size_exp*2 | bc)
                for primal_alpha in 0.99
                do
                    python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
                        --nnodes=${num_machines_base} \
                        --node_rank=${machine_rank} \
                        --master_addr=${master_addr} \
                        --master_port=2500 \
                        ./main_gan.py --master_addr=${master_addr} \
                        --data_dir=${data_dir} \
                        --output_dir="${output_dir_CDMA_ADA}K_${num_local_iterations}/" \
                        --use_gpu=True \
                        --latent_vector_len=${latent_vector_len} \
                        --image_size=${image_size} \
                        --algorithm_to_run=CDMA_ADA \
                        --dataset_name=${dataset_name} \
                        --num_partitions=${num_partitions} \
                        --num_nodes=${num_nodes_base} \
                        --num_rounds=${num_rounds} \
                        --num_local_iterations=${num_local_iterations} \
                        --primal_step_size=${primal_step_size} \
                        --dual_step_size=${dual_step_size} \
                        --step_size_exp=${step_size_exp} \
                        --primal_alpha=${primal_alpha} \
                        --dual_alpha=${primal_alpha} \
                        --alpha_exp=${alpha_exp} \
                        --print_freq=${print_freq} \
                        --sort_by=${sort_by} \
                        --similarity=${similarity} \
                        --random_seed_id=${random_seed_id} \
                        --train_batch_size=${train_batch_size} \
                        --test_batch_size=${test_batch_size} \
                        --max_machine_drop_ratio=${max_machine_drop_ratio}
                done
            done
        done
    done
done
    
for num_local_iterations in 20
do
    for primal_step_size in 0.01
    do
        for dual_step_size in 0.01
        do
            python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
                --nnodes=${num_machines_base} \
                --node_rank=${machine_rank} \
                --master_addr=${master_addr} \
                --master_port=2500 \
                ./main_gan.py --master_addr=${master_addr} \
                --data_dir=${data_dir} \
                --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
                --use_gpu=True \
                --latent_vector_len=${latent_vector_len} \
                --image_size=${image_size} \
                --algorithm_to_run=CDMA_ONE \
                --dataset_name=${dataset_name} \
                --num_partitions=${num_partitions} \
                --num_nodes=${num_nodes_base} \
                --num_rounds=${num_rounds} \
                --num_local_iterations=${num_local_iterations} \
                --primal_step_size=${primal_step_size} \
                --dual_step_size=${dual_step_size} \
                --print_freq=${print_freq} \
                --sort_by=${sort_by} \
                --similarity=${similarity} \
                --random_seed_id=${random_seed_id} \
                --train_batch_size=${train_batch_size} \
                --test_batch_size=${test_batch_size} \
                --max_machine_drop_ratio=${max_machine_drop_ratio}
        done
    done
done

num_local_iterations=1
output_dir_CDMA_ONE="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_MB_T_${num_rounds}/"

for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node} \
            --nnodes=${num_machines_double} \
            --node_rank=${machine_rank} \
            --master_addr=${master_addr} \
            --master_port=2500 \
            ./main_gan.py --master_addr=${master_addr} \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=True \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
            --dataset_name=${dataset_name} \
            --num_partitions=${num_partitions} \
            --num_nodes=${num_nodes_double} \
            --num_rounds=${num_rounds} \
            --num_local_iterations=${num_local_iterations} \
            --primal_step_size=${primal_step_size} \
            --dual_step_size=${dual_step_size} \
            --print_freq=${print_freq} \
            --sort_by=${sort_by} \
            --similarity=${similarity} \
            --random_seed_id=${random_seed_id} \
            --train_batch_size=${train_batch_size} \
            --test_batch_size=${test_batch_size} \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done 

for num_local_iterations in 25
do
    for local_step_size in 0.01
    do
        for global_step_size in 1.1
        do
            for algorithm_reg_coef in 0
            do
                count=0
                for T0 in 4000
                do
            	((count+=1))
            	if [ "${algorithm_reg_coef}" == 0 ] && [ "${count}" != 1 ]; then
            	    continue
            	fi
                    # python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node_codasca} \
                    #     --nnodes=${num_machines_codasca} \
                    #     --node_rank=${machine_rank} \
                    #     --master_addr=${master_addr} \
                    #     --master_port=2500 \
                    #     ./main_gan.py --master_addr=${master_addr} \
                    python3 -m torch.distributed.launch --nproc_per_node=${num_proc_per_node_codasca} \
                        ./main_gan.py --master_addr='127.0.0.1' \
                        --data_dir=${data_dir} \
                        --output_dir="${output_dir_CODASCA}K_${num_local_iterations}/" \
                        --use_gpu=True \
                        --latent_vector_len=${latent_vector_len} \
                        --image_size=${image_size} \
                        --algorithm_to_run=CODASCA \
                        --dataset_name=${dataset_name} \
                        --num_partitions=${num_partitions} \
                        --num_nodes=${num_nodes_base} \
                        --num_threads=${num_threads_codasca} \
                        --num_rounds=${num_rounds} \
                        --num_local_iterations=${num_local_iterations} \
                        --local_step_size=${local_step_size} \
                        --global_step_size=${global_step_size} \
                        --algorithm_reg_coef=${algorithm_reg_coef} \
                        --T0=${T0} \
                        --print_freq=${print_freq} \
                        --sort_by=${sort_by} \
                        --similarity=${similarity} \
                        --random_seed_id=${random_seed_id} \
                        --train_batch_size=${train_batch_size} \
                        --test_batch_size=${test_batch_size} \
                        --max_machine_drop_ratio=${max_machine_drop_ratio}
                done
            done
        done
    done
done
