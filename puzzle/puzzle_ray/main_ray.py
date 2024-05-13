import argparse
import time
from typing import List, Dict

import sys
sys.path.extend("..")

import ray
from ray.util.placement_group import placement_group

from puzzle_ray.ray.group import PPORayGroup, ReferenceModelRayRole, RewardModelRayRole
from puzzle_ray.ray.actor import ActorModelRayRole
from puzzle_ray.ray.critic import CriticModelRayRole
from puzzle_ray.ray.time_shared import TimeSharedModelRayRole
from puzzle_ray.trainer import PPOTrainer, PPOTrainerTimeShared


def parser_args(parser):
    group = parser.add_argument_group(title='rlhf')

    # DATA args
    group.add_argument(
        '--tokenizer-type',
        type=str,
        default='PretrainedFromHF',
        help='What type of tokenizer to use.'
    )
    group.add_argument(
        '--tokenizer-model',
        type=str,
        default=None,
        help='Sentencepiece tokenizer model.'
    )
    group.add_argument(
        '--data-path',
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
        '--seq-length',
        type=int,
        default=None,
        help='Maximum sequence length to process.'
    )

    # RLHF args
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
    # group.add_argument(
    #     "--unsupervised_dataset_name",
    #     type=str,
    #     default=None,
    #     help="The name of the dataset to use (via the datasets library).")
    # group.add_argument(
    #     "--unsupervised_dataset_config_name",
    #     type=str,
    #     default=None,
    #     help=
    #     "The configuration name of the dataset to use (via the datasets library).")

    # For Megatron GPT training configure
    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--load-model-from-hf-config',
                       action='store_true',
                       help='Load model config from HuggingFace config file.')
    group.add_argument('--load-model-hf-checkpoint',
                       action='store_true',
                       help='Load model checkpoint from Huggingface')
    group.add_argument('--model-name-or-path',
                       type=str, default=None,
                          help='HuggingFace model name or path to config file.'),
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--inference-batch-times-seqlen-threshold',
                       type=int, default=512,
                       help='During inference, if batch-size times '
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--no-position-embedding',
                       action='store_false',
                       help='Disable position embedding.',
                       dest='add_position_embedding')

    # Megatron-LM learning
    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')

    # Megatron regularization args
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')

    # Megatron mixed prcision
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')

    # For pipeline generation
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

    # Shadow optimization
    group.add_argument(
        "--use-shadow",
        action='store_true',
        help="use shadow model for generation.")
    group.add_argument(
        "--shadow-tensor-model-parallel-size",
        default=1,
        type=int,
        help="The tensor model parallel size for shadow model.")
    group.add_argument(
        "--shadow-pipeline-model-parallel-size",
        default=1,
        type=int,
        help="The pipeline model parallel size for shadow model.")

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

    # component placement type
    group.add_argument(
        '--placement_type',
        default=1,
        type=int,
        help='(1) allocate all \n(2) distributed each \n(3) colocate actor and critic and Ref and Reward are half-half')
    group.add_argument(
        '--num-nodes',
        default=1,
        type=int,
        help='The number of nodes.'
    )
    group.add_argument(
        '--num-gpus-per-node',
        default=8,
        type=int,
        help='The number of gpus within a node.'
    )

    ## Debug
    parser.add_argument(
        "--early_finish",
        type=int,
        default=-1,
        help=
        "Early finish for debug."
    )
    return parser


def main(args):

    ray.init()

    # init each compontent
    # now support placement_type
    # For instance, consider a scenario with 4 GPUs.
    # (1) allocate all               (2) distributed each           (3) colocate actor and critic
    #                                                                   and ref and reward are half-half
    # |-------|---|---|---|---|      |-------|---|---|---|---|      |-------|---|---|---|---|
    # |       | A | R | R | C |      |       | A | R | R | C |      |       | A | R | R | C |
    # |-------|---|---|---|---|      |-------|---|---|---|---|      |-------|---|---|---|---|
    # | GPU0  | 1 | 1 | 1 | 1 |      | GPU0  | 1 | 0 | 0 | 0 |      | GPU0  | 1 | 1 | 0 | 1 |
    # | GPU1  | 1 | 1 | 1 | 1 |  or  | GPU1  | 0 | 1 | 0 | 0 |  or  | GPU1  | 1 | 1 | 0 | 1 |
    # | GPU2  | 1 | 1 | 1 | 1 |      | GPU2  | 0 | 0 | 1 | 0 |      | GPU2  | 1 | 0 | 1 | 1 |
    # | GPU3  | 1 | 1 | 1 | 1 |      | GPU3  | 0 | 0 | 0 | 1 |      | GPU3  | 1 | 0 | 1 | 1 |
    # |-------|---|---|---|---|      |-------|---|---|---|---|      |-------|---|---|---|---|
    #
    # NOTE:
    # it can easys to support more complex placement_type

    pg = None
    if args.placement_type == 1:
        # NOTE: This placement results in each model creating a process on the GPU,
        # leading to increased memory pressure. Each process initiates the CUDA runtime
        # and some CUDA kernel, typically accounting for around 2GB of memory.

        num_gpus_per_node = 8
        bundles = [
            {"GPU": num_gpus_per_node, "CPU": num_gpus_per_node*4}
            for _ in range(1)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

        actor_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=ActorModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )

        critic_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=8,
            ray_actor_type=CriticModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.22 if pg else 1,
        )

        reward_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=8,
            ray_actor_type=RewardModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.02 if pg else 1,
        )

        ref_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=8,
            ray_actor_type=ReferenceModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.01 if pg else 1,
        )

        num_total_iters = ray.get(actor_models.async_init_model_from_pretrained())[0]
        actor_models.get_parallel_size()

        refs = []
        refs.extend(critic_models.async_init_model_from_pretrained())
        refs.extend(reward_models.async_init_model_from_pretrained())
        refs.extend(ref_models.async_init_model_from_pretrained())

        ray.get(refs)

        trainer = PPOTrainer(actor_models, critic_models, reward_models, ref_models)

    elif args.placement_type == 2:
        raise NotImplementedError

    elif args.placement_type == 3:
        # NOTE: For experiment, same problem with case (1), but less memory pressure
        # as reward and reference model is distributed to different GPU.

        num_gpus_per_node = 8
        bundles = [
            {"GPU": num_gpus_per_node, "CPU": num_gpus_per_node*4}
            for _ in range(1)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

        actor_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=ActorModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )

        critic_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=8,
            ray_actor_type=CriticModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.22 if pg else 1,
        )

        reward_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=4,
            ray_actor_type=RewardModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.03 if pg else 1,
        )

        ref_models = PPORayGroup(
            num_nodes=1,
            num_gpus_per_node=4,
            ray_actor_type=ReferenceModelRayRole,
            pg=pg,
            num_gpus_per_actor=0.03 if pg else 1,
        )

        num_total_iters = ray.get(actor_models.async_init_model_from_pretrained())[0]
        actor_models.get_parallel_size()

        refs = []
        refs.extend(critic_models.async_init_model_from_pretrained())
        refs.extend(reward_models.async_init_model_from_pretrained())
        refs.extend(ref_models.async_init_model_from_pretrained())

        ray.get(refs)

        trainer = PPOTrainer(actor_models, critic_models, reward_models, ref_models)

    elif args.placement_type == 4:
        # In this case, the memory usage would much better, as only one process is created per GPU.

        # TODO(lkm):
        num_nodes = args.num_nodes
        num_gpus_per_node = args.num_gpus_per_node
        bundles = [
            {"GPU": num_gpus_per_node, "CPU": num_gpus_per_node*4}
            for _ in range(num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

        time_shared_models = PPORayGroup(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=TimeSharedModelRayRole,
            pg=pg,
            num_gpus_per_actor=1,
        )

        num_total_iters = ray.get(time_shared_models.async_init_model_from_pretrained(**vars(args)))[0]

        trainer = PPOTrainerTimeShared(time_shared_models)
        print("num_total_iters:", num_total_iters)

        time_shared_models.get_parallel_size()

    else:
        raise NotImplementedError(f"Unknown placement type: {args.placement_type}")

    for epoch in range(args.num_train_epochs):
        print(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {num_total_iters}")

        for step in range(num_total_iters):
            generation_start = time.time()
            batch_exp: List[Dict] = trainer.generate_experience(step)
            generation_end = time.time()
            generation_time = generation_end - generation_start

            num_total_exp_iters = trainer.add_exp_dataset(batch_exp)

            # update model
            inner_iter = 0
            actor_loss_sum, critic_loss_sum, average_reward = 0, 0, trainer.average_reward
            training_start = time.time()
            for ppo_ep in range(args.ppo_epochs):
                for i in range(num_total_exp_iters):
                    inner_iter += 1
                    actor_loss, critic_loss = trainer.train_rlhf(i)
                    actor_loss_sum += actor_loss["actor_loss"].item()
                    critic_loss_sum += critic_loss["critic_loss"].item()

            training_end = time.time()
            training_time = training_end - training_start

            generate_seq_time = trainer.generate_seq_time
            e2e_time = training_time + generation_time
            print(f'epoch: {epoch} | step: {step}/{num_total_iters} | ppo_ep: {ppo_ep+1} | global_batch_size: {args.global_batch_size} '
                  f'| average_reward: {average_reward:.3f} | act_loss: {actor_loss_sum/inner_iter:.3f} | cri_loss: {critic_loss_sum/inner_iter:.3f} '
                  f'| stage 1(gen.) (ms): {generate_seq_time*1000:.3f} | stage 2(infer.) (ms): {(generation_time-generate_seq_time)*1000:.3f} | stage 3(train.) (ms): {training_time*1000:.3f} | e2e_time (ms): {e2e_time*1000:.3f}')

            if args.early_finish > 0 and step == args.early_finish:
                print("exit with early finished, for debug")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()
    main(args)