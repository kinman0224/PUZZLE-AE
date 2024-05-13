#!/bin/bash

start_time=$(date +%s)

# echo " - Figure 9 starts now: $(date -Iseconds)"

# create log file
LOG_FILES=$1
# Figure 9
if [ "$LOG_FILES" == "" ]; then
    LOG_FILES=`pwd`/fig9_$(date -Iseconds)
    mkdir -p $LOG_FILES
fi

echo "Figure 9..."
# > $NODELIST pass from `./RUNME-b.sh`
MASTER_NODE=$(echo $NODELIST | cut -d',' -f1)

echo "Start 7b/350m..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure9/run_rlhf_7b_350m_w_bulk.sh > "$LOG_FILES/llama_7b_350m_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure9/run_rlhf_7b_350m_wo_bulk.sh > "$LOG_FILES/llama_7b_350m_wo_bulk_$(date -Iseconds).log" 2>&1

echo "Start 7b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure9/run_rlhf_7b_7b_w_bulk.sh > "$LOG_FILES/llama_7b_7b_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure9/run_rlhf_7b_7b_wo_bulk.sh  > "$LOG_FILES/llama_7b_7b_wo_bulk_$(date -Iseconds).log" 2>&1

echo "Start 13b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure9/run_rlhf_13b_7b_w_bulk.sh > "$LOG_FILES/llama_13b_7b_w_bulk_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure9/run_rlhf_13b_7b_wo_bulk.sh  > "$LOG_FILES/llama_13b_7b_wo_bulk_$(date -Iseconds).log" 2>&1

end_time=$(date +%s)

# echo " - Figure 8 ends now: $(date -Iseconds)"
