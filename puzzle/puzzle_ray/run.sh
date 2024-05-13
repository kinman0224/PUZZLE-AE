#!/bin/bash

set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1

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
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --load-model-from-hf-config \
    --model-name-or-path $ACTOR_PATH \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_SEQ_LEN \
    --micro-batch-size 4 \
    --global-batch-size 128 \
    --inference-batch-times-seqlen-threshold 1024 \
    --train-iters 500000 \
    --lr 0.00015 \
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
    --ar_stage_mbs 16 \
    --fp16
"

DATA_ARGS="
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH
"

python3 puzzle_ray/main_ray.py \
    $RLHF_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    --placement_type 4