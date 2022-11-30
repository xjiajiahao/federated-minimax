#!/bin/bash
# input argument:
#   nic: the network interface card name (e.g., eno2)

if [[ $# -ne 0 ]] ; then
    export GLOO_SOCKET_IFNAME=$1
fi
random_seed_id=1234
export PYTHONOPTIMIZE=1  # remove the __debug__ code

latent_vector_len=100
image_size=28
num_partitions=500
num_processes_codasca=5
num_threads_codasca=101
num_nodes_base=8
num_nodes_double=16
max_machine_drop_ratio=0.5
num_rounds=400
train_batch_size=30
print_freq=20
data_dir='./data/'
output_dir_prefix="${data_dir}results_gan_mnist_best_${random_seed_id}/"
output_dir_CDMA_ONE="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_CDMA_ONE_T_${num_rounds}/"
output_dir_CDMA_ONE_VR="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_CDMA_ADA_T_${num_rounds}/"
output_dir_CDMA_NC="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_CDMA_NC_T_${num_rounds}/"
output_dir_Local_SGDA_Plus="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_Local_SGDA_Plus_T_${num_rounds}/"
output_dir_Catalyst_Scaffold_S="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_Catalyst_Scaffold_S_T_${num_rounds}/"
output_dir_Extra_Step_Local_SGD="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_Extra_Step_Local_SGD_T_${num_rounds}/"
output_dir_CODA_Plus="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_CODA_Plus_T_${num_rounds}/"
output_dir_CODASCA="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_base}_B_${train_batch_size}_results_CODASCA_T_${num_rounds}/"
sort_by='label'
similarity=0.0




num_local_iterations=5
for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        for step_size_exp in 0
        do
            alpha_exp=$(echo $step_size_exp*2 | bc)
            for primal_alpha in 0.99
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
                    ./main_gan.py --master_addr='127.0.0.1' \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CDMA_ONE_VR}K_${num_local_iterations}/" \
                    --use_gpu=False \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CDMA_ADA \
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
                    --test_batch_size=100 \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_NC \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done














num_local_iterations=10
for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        for step_size_exp in 0
        do
            alpha_exp=$(echo $step_size_exp*2 | bc)
            for primal_alpha in 0.99
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
                    ./main_gan.py --master_addr='127.0.0.1' \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CDMA_ONE_VR}K_${num_local_iterations}/" \
                    --use_gpu=False \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CDMA_ADA \
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
                    --test_batch_size=100 \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_NC \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done










num_local_iterations=15
for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        for step_size_exp in 0
        do
            alpha_exp=$(echo $step_size_exp*2 | bc)
            for primal_alpha in 0.99
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
                    ./main_gan.py --master_addr='127.0.0.1' \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CDMA_ONE_VR}K_${num_local_iterations}/" \
                    --use_gpu=False \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CDMA_ADA \
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
                    --test_batch_size=100 \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_NC \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done








num_local_iterations=20
for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        for step_size_exp in 0
        do
            alpha_exp=$(echo $step_size_exp*2 | bc)
            for primal_alpha in 0.99
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
                    ./main_gan.py --master_addr='127.0.0.1' \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CDMA_ONE_VR}K_${num_local_iterations}/" \
                    --use_gpu=False \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CDMA_ADA \
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
                    --test_batch_size=100 \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_NC \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done















num_local_iterations=25
for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        for step_size_exp in 0
        do
            alpha_exp=$(echo $step_size_exp*2 | bc)
            for primal_alpha in 0.99
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_base} \
                    ./main_gan.py --master_addr='127.0.0.1' \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CDMA_ONE_VR}K_${num_local_iterations}/" \
                    --use_gpu=False \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CDMA_ADA \
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
                    --test_batch_size=100 \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.001
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_NC \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done










num_local_iterations=1
output_dir_CDMA_ONE="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_CDMA_ONE_T_${num_rounds}/"
output_dir_CDMA_ONE_VR="${output_dir_prefix}/n_${num_partitions}_S_${num_nodes_double}_B_${train_batch_size}_results_CDMA_ADA_T_${num_rounds}/"

for primal_step_size in 0.001
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_ONE}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_ONE \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done

for primal_step_size in 0.001
do
    for dual_step_size in 0.01
    do
        for step_size_exp in 0
        do
            alpha_exp=$(echo $step_size_exp*2 | bc)
            for primal_alpha in 0.99
            do
                python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
                    ./main_gan.py --master_addr='127.0.0.1' \
                    --data_dir=${data_dir} \
                    --output_dir="${output_dir_CDMA_ONE_VR}K_${num_local_iterations}/" \
                    --use_gpu=False \
                    --latent_vector_len=${latent_vector_len} \
                    --image_size=${image_size} \
                    --algorithm_to_run=CDMA_ADA \
                    --num_partitions=${num_partitions} \
                    --num_nodes=${num_nodes_double} \
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
                    --test_batch_size=100 \
                    --max_machine_drop_ratio=${max_machine_drop_ratio}
            done
        done
    done
done

for primal_step_size in 0.01
do
    for dual_step_size in 0.01
    do
        python3 -m torch.distributed.launch --nproc_per_node=${num_nodes_double} \
            ./main_gan.py --master_addr='127.0.0.1' \
            --data_dir=${data_dir} \
            --output_dir="${output_dir_CDMA_NC}K_${num_local_iterations}/" \
            --use_gpu=False \
            --latent_vector_len=${latent_vector_len} \
            --image_size=${image_size} \
            --algorithm_to_run=CDMA_NC \
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
            --test_batch_size=100 \
            --max_machine_drop_ratio=${max_machine_drop_ratio}
    done
done
