#!/bin/bash

source env.sh

if [ -z $MASTER_ADDR ]
then
    if [ -z $SLURM_JOB_ID ]
    then
        export MASTER_ADDR=localhost
    else
        export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
    fi
fi

GPUS_PER_NODE=8
MASTER_IP_ADDR=$(ping $MASTER_ADDR -c 1 | grep "PING" | awk '{print $3}' | tr -d '()')
MASTER_PORT=6379
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID

if [ "$NODE_RANK" == "0" ]; then
    ray start --head --num-gpus=$GPUS_PER_NODE --block
else
    ray start --address=$MASTER_IP_ADDR:$MASTER_PORT --num-gpus=$GPUS_PER_NODE --block
fi
