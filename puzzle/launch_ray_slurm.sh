#!/bin/bash

#SBATCH -p Nvidia_A800
#SBATCH -J AE
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch
#SBATCH --exclusive
#SBATCH --output=out.log

set -x

export RAY_DEDUP_LOGS=0

SLURM_CPUS_PER_TASK=64
SLURM_GPUS_PER_TASK=8

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1  --cpus-per-task=8 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" --mem=500GB --gres=gpu:8 --cpus-per-task=8 --exclusive \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-gpus "${SLURM_GPUS_PER_TASK}" --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --mem=500GB --gres=gpu:8 --cpus-per-task=8 --exclusive \
        ray start --address "$ip_head" \
        --num-gpus "${SLURM_GPUS_PER_TASK}" --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 5
done

# echo "> STARTING TRAINING"
srun --nodes=1 --ntasks=1 -w "$head_node" --gres=gpu:0 --mem-per-cpu=4G --exclusive $1

sleep 3600

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."