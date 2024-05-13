#!/bin/bash

# set -x

# create log file
LOG_FILES=$1
# Figure 8
if [ "$LOG_FILES" == "" ]; then
    LOG_FILES=`pwd`/fig8_$(date -Iseconds)
    mkdir -p $LOG_FILES
fi

# !!! THE SELECTED PARTITION is Nvidia_A800 and NODES FOR AE ARE NODELIST=gpu20,gpu21,gpu16,gpu17, WHICH ARE STABLE NODES IN OUR TEST ENVIRONMENT
# !!! PLEASE MODIFY HERE IF YOU NEED
PARTITION=Nvidia_A800

# # NODELIST=gpu21
srun -u -N 1 --gres=gpu:8 -p ${PARTITION} --mem=0 --cpus-per-task=64 -w "${NODELIST}" \
bash ./training_scripts/ae/run_llama2_7b_350m.sh > "$LOG_FILES/llama_7b_350m_dschat_$(date -Iseconds).log" 2>&1

# # NODELIST=gpu20,gpu21
srun -u -N 2 --gres=gpu:8 -p ${PARTITION} --mem=0 --cpus-per-task=64 -w "${NODELIST}" \
bash ./training_scripts/ae/run_llama2_7b_7b.sh > "$LOG_FILES/llama_7b_7b_dschat_$(date -Iseconds).log" 2>&1

# # NODELIST=gpu20,gpu21
srun -u -N 2 --gres=gpu:8 -p ${PARTITION} --mem=0 --cpus-per-task=64 -w "${NODELIST}" \
bash ./training_scripts/ae/run_llama2_13b_350m.sh > "$LOG_FILES/llama_13b_350m_dschat_$(date -Iseconds).log" 2>&1

# # NODELIST=gpu20,gpu21
srun -u -N 2 --gres=gpu:8 -p ${PARTITION} --mem=0 --cpus-per-task=64 -w "${NODELIST}" \
bash ./training_scripts/ae/run_llama2_13b_7b.sh > "$LOG_FILES/llama_13b_7b_dschat_$(date -Iseconds).log" 2>&1

# # NODELIST=gpu20,gpu21,gpu16,gpu17
srun -u -N 4 --gres=gpu:8 -p ${PARTITION} --mem=0 --cpus-per-task=64 -w "${NODELIST}" \
bash ./training_scripts/ae/run_llama2_33b_7b.sh > "$LOG_FILES/llama_33b_7b_dschat_$(date -Iseconds).log" 2>&1
