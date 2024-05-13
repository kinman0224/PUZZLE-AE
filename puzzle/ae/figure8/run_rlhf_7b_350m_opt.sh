#!/bin/bash

# Example log output:
#
# epoch: 0 | step: 0 | ppo_ep: 1 | average_reward: 0.354 | act_loss: 5.626 | cri_loss: 5.109 | training_time (ms): 4914.274 | generation_time (ms): 13176.463 | e2e_time (ms): 18090.737
# epoch: 0 | step: 1 | ppo_ep: 1 | average_reward: 0.346 | act_loss: 5.535 | cri_loss: 5.047 | training_time (ms): 4351.295 | generation_time (ms): 7408.764 | e2e_time (ms): 11760.058
# epoch: 0 | step: 2 | ppo_ep: 1 | average_reward: 0.329 | act_loss: 5.648 | cri_loss: 5.093 | training_time (ms): 4360.937 | generation_time (ms): 7291.672 | e2e_time (ms): 11652.609
# epoch: 0 | step: 3 | ppo_ep: 1 | average_reward: 0.358 | act_loss: 5.769 | cri_loss: 5.169 | training_time (ms): 4342.493 | generation_time (ms): 7291.751 | e2e_time (ms): 11634.244
# epoch: 0 | step: 4 | ppo_ep: 1 | average_reward: 0.327 | act_loss: 5.459 | cri_loss: 5.102 | training_time (ms): 4324.281 | generation_time (ms): 7255.829 | e2e_time (ms): 11580.110
# epoch: 0 | step: 5 | ppo_ep: 1 | average_reward: 0.347 | act_loss: 5.677 | cri_loss: 5.171 | training_time (ms): 4320.825 | generation_time (ms): 7269.019 | e2e_time (ms): 11589.843

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
    --ar_stage_mbs 64 \
    --bulk_switch_on \
    --use-shadow \
    --shadow-tensor-model-parallel-size 2 \
    --shadow-pipeline-model-parallel-size 4 \
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