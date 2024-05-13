#!/bin/bash

# Example log output:
#
# epoch: 0 | step: 0 | ppo_ep: 1 | average_reward: -0.176 | act_loss: 5.719 | cri_loss: 5.067 | training_time (ms): 8261.691 | generation_time (ms): 15369.438 | e2e_time (ms): 23631.130
# epoch: 0 | step: 1 | ppo_ep: 1 | average_reward: -0.182 | act_loss: 5.829 | cri_loss: 5.072 | training_time (ms): 4557.357 | generation_time (ms): 11043.075 | e2e_time (ms): 15600.432
# epoch: 0 | step: 2 | ppo_ep: 1 | average_reward: -0.174 | act_loss: 5.656 | cri_loss: 5.236 | training_time (ms): 4547.725 | generation_time (ms): 11088.776 | e2e_time (ms): 15636.501
# epoch: 0 | step: 3 | ppo_ep: 1 | average_reward: -0.223 | act_loss: 5.629 | cri_loss: 5.180 | training_time (ms): 4531.187 | generation_time (ms): 11043.801 | e2e_time (ms): 15574.988
# epoch: 0 | step: 4 | ppo_ep: 1 | average_reward: -0.216 | act_loss: 5.647 | cri_loss: 5.167 | training_time (ms): 4542.104 | generation_time (ms): 11034.389 | e2e_time (ms): 15576.493
# epoch: 0 | step: 5 | ppo_ep: 1 | average_reward: -0.164 | act_loss: 5.901 | cri_loss: 5.139 | training_time (ms): 4581.679 | generation_time (ms): 11121.413 | e2e_time (ms): 15703.092

export CUDA_DEVICE_MAX_CONNECTIONS=1

OTHER_ARGS="
    --num-gpus-per-node 8 \
    --num-nodes 1 \
    --early_finish 5
"

MAX_PROMPT_SEQ_LEN=256
MAX_ANSWER_SEQ_LEN=256
MAX_SEQ_LEN=$((MAX_PROMPT_SEQ_LEN+MAX_ANSWER_SEQ_LEN))

ACTOR_PATH=/public/thu_ljw_workspace/dataset/Llama-2-7b-hf/
CRITIC_PATH=./puzzle/example/config/Llama2-350m-hf/

TOKENIZER_MODEL=/public/thu_ljw_workspace/dataset/Llama-2-7b-hf/
DATA_PATH=/public/thu_ljw_workspace/dataset/Dahoas/rm-static/

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
    --data-path $DATA_PATH \
    --data_output_path /public/home/qinghuatest/puzzletmp
"

python3 puzzle_ray/main_ray.py \
    $OTHER_ARGS \
    $RLHF_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    --placement_type 4