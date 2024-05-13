#!/bin/bash

start_time=$(date +%s)

# echo " - Figure 10 starts now: $(date -Iseconds)"

LOG_FILES=$1

# create log file
# Figure 10
if [ "$LOG_FILES" == "" ]; then
    LOG_FILES=`pwd`/fig10_$(date -Iseconds)
    mkdir -p $LOG_FILES
fi

echo "Figure 10..."
# > $NODELIST pass from `./RUNME-b.sh`
MASTER_NODE=$(echo $NODELIST | cut -d',' -f1)

echo "Start 7b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_16_1_1_w_bulk.sh > "$LOG_FILES/llama_7b_7b_128_16_1_1_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_16_1_1_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_128_16_1_1_wo_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_4_1_4_w_bulk.sh > "$LOG_FILES/llama_7b_7b_128_4_1_4_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_4_1_4_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_128_4_1_4_wo_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_8_1_2_w_bulk.sh > "$LOG_FILES/llama_7b_7b_128_8_1_2_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_8_1_2_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_128_8_1_2_wo_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_8_2_1_w_bulk.sh > "$LOG_FILES/llama_7b_7b_128_8_2_1_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_7b_7b_128_8_2_1_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_128_8_2_1_wo_bulk_$(date -Iseconds).log" 2>&1

# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_16_1_1_w_bulk.sh > "$LOG_FILES/llama_7b_7b_256_16_1_1_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_16_1_1_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_256_16_1_1_wo_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_4_1_4_w_bulk.sh > "$LOG_FILES/llama_7b_7b_256_4_1_4_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_4_1_4_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_256_4_1_4_wo_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_8_1_2_w_bulk.sh > "$LOG_FILES/llama_7b_7b_256_8_1_2_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_8_1_2_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_256_8_1_2_wo_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_8_2_1_w_bulk.sh > "$LOG_FILES/llama_7b_7b_256_8_2_1_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_7b_7b_256_8_2_1_wo_bulk.sh > "$LOG_FILES/llama_7b_7b_256_8_2_1_wo_bulk_$(date -Iseconds).log" 2>&1


echo "Start 13b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_13b_7b_64_4_1_4_w_bulk.sh > "$LOG_FILES/llama_13b_7b_64_4_1_4_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_13b_7b_64_4_1_4_wo_bulk.sh > "$LOG_FILES/llama_13b_7b_64_4_1_4_wo_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_13b_7b_64_8_1_2_w_bulk.sh > "$LOG_FILES/llama_13b_7b_64_8_1_2_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_13b_7b_64_8_1_2_wo_bulk.sh > "$LOG_FILES/llama_13b_7b_64_8_1_2_wo_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_13b_7b_64_8_2_1_w_bulk.sh > "$LOG_FILES/llama_13b_7b_64_8_2_1_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure10/run_rlhf_13b_7b_64_8_2_1_wo_bulk.sh > "$LOG_FILES/llama_13b_7b_64_8_2_1_wo_bulk_$(date -Iseconds).log" 2>&1

# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_13b_7b_128_4_1_4_w_bulk.sh > "$LOG_FILES/llama_13b_7b_128_4_1_4_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_13b_7b_128_4_1_4_wo_bulk.sh > "$LOG_FILES/llama_13b_7b_128_4_1_4_wo_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_13b_7b_128_8_1_2_w_bulk.sh > "$LOG_FILES/llama_13b_7b_128_8_1_2_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_13b_7b_128_8_1_2_wo_bulk.sh > "$LOG_FILES/llama_13b_7b_128_8_1_2_wo_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_13b_7b_128_8_2_1_w_bulk.sh > "$LOG_FILES/llama_13b_7b_128_8_2_1_w_bulk_$(date -Iseconds).log" 2>&1
# srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
# bash ae/figure10/run_rlhf_13b_7b_128_8_2_1_wo_bulk.sh > "$LOG_FILES/llama_13b_7b_128_8_2_1_wo_bulk_$(date -Iseconds).log" 2>&1


end_time=$(date +%s)

