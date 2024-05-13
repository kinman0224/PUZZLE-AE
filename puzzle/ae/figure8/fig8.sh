#!/bin/bash

start_time=$(date +%s)

# echo " - Figure 8 starts now: $(date -Iseconds)"

# create log file
LOG_FILES=$1
# Figure 8
if [ "$LOG_FILES" == "" ]; then
    LOG_FILES=`pwd`/fig8_$(date -Iseconds)
    mkdir -p $LOG_FILES
fi

echo "Figure 8..."
# > $NODELIST pass from `./RUNME-b.sh`
MASTER_NODE=$(echo $NODELIST | cut -d',' -f1)

echo "Start 7b/350m..., master node on $MASTER_NODE"
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_7b_350m_base.sh > "$LOG_FILES/llama_7b_350m_base_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_7b_350m_opt.sh > "$LOG_FILES/llama_7b_350m_opt_$(date -Iseconds).log" 2>&1

echo "Start 7b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_7b_7b_base.sh > "$LOG_FILES/llama_7b_7b_base_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_7b_7b_opt.sh  > "$LOG_FILES/llama_7b_7b_opt_$(date -Iseconds).log" 2>&1

echo "Start 13b/350m..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_13b_350m_base.sh > "$LOG_FILES/llama_13b_350m_base_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_13b_350m_opt.sh  > "$LOG_FILES/llama_13b_350m_opt_$(date -Iseconds).log" 2>&1

echo "Start 13b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_13b_7b_base.sh > "$LOG_FILES/llama_13b_7b_base_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_13b_7b_opt.sh  > "$LOG_FILES/llama_13b_7b_opt_$(date -Iseconds).log" 2>&1

echo "Start 33b/7b..."
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_33b_7b_base.sh > "$LOG_FILES/llama_33b_7b_base_$(date -Iseconds).log" 2>&1
srun -N1 -p Nvidia_A800 -w ${MASTER_NODE} --mem=100G --cpus-per-task=1 \
bash ae/figure8/run_rlhf_33b_7b_opt.sh  > "$LOG_FILES/llama_33b_7b_opt_$(date -Iseconds).log" 2>&1

end_time=$(date +%s)

# echo " - Figure 8 ends now: $(date -Iseconds)"
