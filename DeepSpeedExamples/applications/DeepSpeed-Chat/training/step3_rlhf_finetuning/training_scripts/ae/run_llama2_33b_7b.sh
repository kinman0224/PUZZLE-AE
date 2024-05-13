#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=/public/thu_ljw_workspace/dataset/Llama-2-33b-hf/
CRITIC_MODEL_PATH=/public/thu_ljw_workspace/dataset/Llama-2-7b-hf/
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step3_llama
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

GPUS_PER_NODE=8
if [ -z $MASTER_ADDR ]
then
    if [ -z $SLURM_JOB_ID ]
    then
        export MASTER_ADDR=localhost
    else
        export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
    fi
fi

MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS main.py \
   --data_path /public/thu_ljw_workspace/dataset/Dahoas/rm-static/ \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --inference_tp_size 8 \
   --tp_gather_partition_size 4 \
   --actor_dropout 0.0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_hybrid_engine \
   --offload_reference_model \
   --data_output_path /public/home/qinghuatest/dstmp \
   --early_finish 3
