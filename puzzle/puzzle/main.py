import argparse
from datetime import datetime
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch

from transformers import AutoConfig

import sys
sys.path.extend("..")

from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.model import LlamaModel, LlamaForCausalLM

from puzzle.rl_engine import RLHFEngine
from puzzle.trainer.ppo_trainer import PPOTrainer
from puzzle.pipeline.utils.data import broadcast_data as pipeline_broadcast_data
from puzzle.utils.data.data_utils import create_datasets, MiniDataset
from puzzle.utils.config import core_transformer_config_from_hf_config

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

def add_rlhf_args(parser):
    group = parser.add_argument_group(title='rlhf')
    group.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    group.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.'
    )
    group.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    group.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    group.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library).")
    group.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    group.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    group.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    group.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    group.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    group.add_argument(
        "--actor_model_name_or_path",
        type=str,
        default=None,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    group.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    group.add_argument(
        "--use_dis_bubble_generation",
        action='store_true',
        help="use hybrid inference for generation.")
    group.add_argument(
        "--pf_stage_mbs",
        default=4,
        type=int,
        help="The micro-batch size used in the prefill stage of generation.")
    group.add_argument(
        "--ar_stage_mbs",
        default=8,
        type=int,
        help="The micro-batch size used in the autoregressive stage of generation.")

    # For pipeline optimization
    group.add_argument(
        "--bulk_switch_on",
        action='store_true',
        help=
        "Merge all experience repeat processes into one continuous pipeline")
    group.add_argument(
        "--exp_repeat",
        type=int,
        default=1,
        help=
        "How many times of repeat the experience dataset is used for training (Only for debug)")
    return parser

def model_provider(pre_process=True, post_process=True, reward_base_model=False):
    """Build the model."""

    print_rank_0('building Llama model ...')

    args = get_args()

    if not reward_base_model:
        model_config = AutoConfig.from_pretrained(args.actor_model_name_or_path)
        core_config = core_transformer_config_from_hf_config(model_config)
        model_class = LlamaForCausalLM
    else:
        model_config = AutoConfig.from_pretrained(args.critic_model_name_or_path)
        core_config = core_transformer_config_from_hf_config(model_config)
        model_class = LlamaModel

    model = model_class(
            config=core_config,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=False
            )

    return model

def main(extra_args_provider, args_defaults):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    args.global_rank = torch.distributed.get_rank()
    torch.cuda.set_device(args.local_rank)

    # TODO(lkm): need to update
    args.per_device_generation_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()
    args.per_device_training_batch_size = args.global_batch_size // parallel_state.get_data_parallel_world_size()

    tokenizer = get_tokenizer()

    timers('create-datasets', log_level=0).start(barrier=True)
    prompt_train_dataloader, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=3)
    if prompt_train_dataloader is not None:
        prompt_train_data_iterator = iter(prompt_train_dataloader)
    timers('create-datasets').stop()

    timers('create-RLHFEngine', log_level=0).start(barrier=True)
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = RLHFEngine(
        model_provider_func=model_provider,
        model_type=ModelType.encoder_or_decoder,
        tokenizer=tokenizer
        )
    timers('create-RLHFEngine').stop()

    timers('create-trainer', log_level=0).start(barrier=True)
    trainer = PPOTrainer(rlhf_engine, args)
    timers('create-trainer', log_level=0).stop()

    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    exp_repeat = args.exp_repeat
    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batches * exp_repeat,
                                    args.per_device_training_batch_size)

    print_rank_0("***** Running training *****")

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {num_total_iters}")

        for step in range(num_total_iters):
            batch_prompt = None
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                batch_prompt = next(prompt_train_data_iterator)
            keys = ['prompt', 'prompt_att_mask']
            batch_prompt = tensor_parallel.broadcast_data(keys, batch_prompt, torch.int64)

            # Generate experience
            torch.cuda.synchronize()
            generation_start = time.time()
            generate_kwargs = {
                'prompts': batch_prompt['prompt'],
                'mask': batch_prompt['prompt_att_mask'],
                'step': step,
            }
            out = trainer.generate_experience(**generate_kwargs)
            torch.cuda.synchronize()
            generation_end = time.time()
            generation_time = generation_end - generation_start
            # print_rank_0(f"> Generation time: {generation_time:.3f} s")

            # ========
            # TODO(lkm):
            batch_exp = None
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                key_int64 = ['prompts', 'input_ids', 'attention_mask']
                batch_exp = pipeline_broadcast_data(key_int64, out, torch.int64)
                key_float32 = ['logprobs', 'ref_logprobs', 'value', 'rewards']
                batch_exp.update(pipeline_broadcast_data(key_float32, out, torch.float32))

            exp_dataset = None
            if batch_exp is not None:
                for _ in range(exp_repeat):
                    exp_dataset = exp_mini_dataset.add(batch_exp)
            num_total_exp_iters = torch.cuda.LongTensor([len(exp_dataset) if exp_dataset is not None else 0])
            torch.distributed.broadcast(num_total_exp_iters, parallel_state.get_tensor_model_parallel_src_rank(), group=parallel_state.get_tensor_model_parallel_group())
            num_total_exp_iters = num_total_exp_iters[0].item()
            # ========

            # Update model
            inner_iter = 0
            actor_loss_sum, critic_loss_sum = 0, 0
            torch.cuda.synchronize()
            training_start = time.time()
            for ppo_ep in range(args.ppo_epochs):
                for i in range(num_total_exp_iters):
                    inner_iter += 1
                    exp_data = None
                    if parallel_state.get_tensor_model_parallel_rank() == 0:
                        exp_data = exp_dataset[i]
                    if not args.bulk_switch_on:
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                    else:
                        actor_loss, critic_loss = trainer.train_rlhf_bulk(exp_data)
                    if parallel_state.is_pipeline_last_stage():
                        actor_loss_sum += actor_loss["actor_loss"].item()
                        critic_loss_sum += critic_loss["critic_loss"].item()
            torch.cuda.synchronize()
            training_end = time.time()
            training_time = training_end - training_start

            # print_rank_0(f"> Training time: {training_time:.3f} s")

            e2e_time = training_time + generation_time
            if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
                print(f'epoch: {epoch} | step: {step} | ppo_ep: {ppo_ep+1} | act_loss: {actor_loss_sum/inner_iter:.3f} | cri_loss: {critic_loss_sum/inner_iter:.3f} | training_time (ms): {training_time*1000:.3f} | generation_time (ms): {generation_time*1000:.3f} | e2e_time (ms): {e2e_time*1000:.3f}')
            # timers.log(['generate-seq', 'train-actor', 'train-critic'])


if __name__ == "__main__":
    main(extra_args_provider=add_rlhf_args,
         args_defaults={'tokenizer_type': 'PretrainedFromHF'})
