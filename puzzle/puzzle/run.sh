#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

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

MAX_PROMPT_SEQ_LEN=256
MAX_ANSWER_SEQ_LEN=256
MAX_SEQ_LEN=$((MAX_PROMPT_SEQ_LEN+MAX_ANSWER_SEQ_LEN))

ACTOR_PATH=/home/zhaijidong/kinman/hf-models/Llama-2-7b-sft-model-ocra-500k/
CRITIC_PATH=/home/zhaijidong/kinman/ae/puzzle/puzzle/example/config/Llama2-350m-hf/
# CRITIC_PATH=/home/zhaijidong/kinman/hf-models/Llama-2-7b-sft-model-ocra-500k/

TOKENIZER_MODEL=/home/zhaijidong/kinman/hf-models/Llama-2-7b-sft-model-ocra-500k/
DATA_PATH=/home/zhaijidong/kinman/hf-datasets/Dahoas/rm-static

RLHF_ARGS="
    --actor_model_name_or_path $ACTOR_PATH \
    --critic_model_name_or_path $CRITIC_PATH \
    --max_prompt_seq_len $MAX_PROMPT_SEQ_LEN \
    --max_answer_seq_len $MAX_ANSWER_SEQ_LEN
"

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4 \
    --load-model-from-hf-config \
    --model-name-or-path $ACTOR_PATH \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_SEQ_LEN \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --inference-batch-times-seqlen-threshold 1024 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --disable-bias-linear \
    --no-position-embedding \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --use_dis_bubble_generation \
    --pf_stage_mbs 4 \
    --ar_stage_mbs 32 \
    --fp16
"
    # --bulk_switch_on \

DATA_ARGS="
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS puzzle/main.py \
    $RLHF_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl

