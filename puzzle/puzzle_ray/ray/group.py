import logging
import os
import socket

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from transformers import AutoConfig

from megatron import print_rank_0
from megatron.core import parallel_state, tensor_parallel
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.core.enums import ModelType
from megatron.model import Float16Module
from megatron.training import get_megatron_optimizer, get_optimizer_param_scheduler
from megatron import get_args, get_timers
from megatron import get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.model import LlamaModel, LlamaForCausalLM

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from puzzle.utils.model.reward_model import RewardWrapper
from puzzle.utils.config import core_transformer_config_from_hf_config
from puzzle.pipeline.inference import forward, forward_value

def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True, reward_base_model=False):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                reward_base_model=reward_base_model
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = parallel_state.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = parallel_state.is_pipeline_stage_before_split()
                add_decoder = parallel_state.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
                reward_base_model=reward_base_model)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                reward_base_model=reward_base_model
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Wrap model for reward model.
    if reward_base_model:
        model = [RewardWrapper(model_module) for model_module in model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if parallel_state.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=parallel_state.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model

def get_reward_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    model = get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=wrap_with_ddp, reward_base_model=True)
    return model

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


class DistributedTorchRayActor:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BaseRole(DistributedTorchRayActor):

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class RewardModelRayRole(BaseRole):

    # def init_model_from_pretrained(self, model_name, model_config, model_weights, model_type):
    def init_model_from_pretrained(self, *args, **kwargs):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        config = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 8,
            "load_model_from_hf_config": True,
            "model_name_or_path": "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            "critic_model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
            "seq_length": 512,
            "max_position_embeddings": 512,
            "micro_batch_size": 2,
            "global_batch_size": 16,
            "inference_batch_times_seqlen_threshold": 1024,
            "lr": 0.00015,
            "train_iters": 500000,
            "lr_decay_iters": 320000,
            "lr_decay_style": "cosine",
            "min_lr": 1.0e-5,
            "weight_decay": 1e-2,
            "add_bias_linear": False,
            "add_position_embedding": False,
            "lr_warmup_fraction": 0.01,
            "clip_grad": 1.0,
            "use_contiguous_buffers_in_local_ddp": False,
            "fp16": True,
            # Tokenizer args
            'tokenizer_type': 'PretrainedFromHF',
            'tokenizer_model': "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            'data_path': "/home/dataset/rlhf-data/Dahoas/rm-static",
            'split': "949,50,1",
        }

        self._init_megatron(extra_args_provider=None, args_defaults=config)

        self.model = self._init_reward(model_provider, model_type=ModelType.encoder_or_decoder)

    def _init_megatron(self, extra_args_provider, args_defaults):
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, ignore_unknown_args=True)

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

    def _init_reward(self, model_provider_func, model_type):
        model = get_reward_model(model_provider_func, model_type, wrap_with_ddp=False)
        return model

    def forward_value(self, seq, attention_mask, batch_size, prompt_length, seq_length, PAD_ID):
        seq = seq.cuda()
        attention_mask = attention_mask.cuda()
        with torch.no_grad():
            reward_score = forward_value(self.model,
                                         seq, attention_mask,
                                         batch_size=batch_size,
                                         prompt_length=prompt_length,
                                         seq_length=seq_length,
                                         PAD_ID=PAD_ID)['chosen_end_scores']
        if parallel_state.is_pipeline_first_stage():
            reward_score = reward_score.cpu()
        return reward_score

    def print_mem(self):
        if torch.distributed.get_rank() == 0:
            print(f"current memory allocated: {torch.cuda.memory_summary()}")

    def empty_cache(self):
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class ReferenceModelRayRole(BaseRole):

    # def init_model_from_pretrained(self, model_name, model_config, model_weights, model_type):
    def init_model_from_pretrained(self, *args, **kwargs):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        config = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 8,
            "load_model_from_hf_config": True,
            # "model_name_or_path": "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            # "actor_model_name_or_path": "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            "model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
            "actor_model_name_or_path": "/home/kinman/code/RLHF/puzzle/puzzle/example/config/Llama2-350m-hf",
            "seq_length": 512,
            "max_position_embeddings": 512,
            "micro_batch_size": 2,
            "global_batch_size": 16,
            "inference_batch_times_seqlen_threshold": 1024,
            "lr": 0.00015,
            "train_iters": 500000,
            "lr_decay_iters": 320000,
            "lr_decay_style": "cosine",
            "min_lr": 1.0e-5,
            "weight_decay": 1e-2,
            "add_bias_linear": False,
            "add_position_embedding": False,
            "lr_warmup_fraction": 0.01,
            "clip_grad": 1.0,
            "use_contiguous_buffers_in_local_ddp": False,
            "fp16": True,
            # Tokenizer args
            'tokenizer_type': 'PretrainedFromHF',
            'tokenizer_model': "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
            'data_path': "/home/dataset/rlhf-data/Dahoas/rm-static",
            'split': "949,50,1",
        }

        self._init_megatron(extra_args_provider=None, args_defaults=config)

        self.model = self._init_ref(model_provider, model_type=ModelType.encoder_or_decoder)

        # print(f"current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")

    def _init_megatron(self, extra_args_provider, args_defaults):
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, ignore_unknown_args=True)

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

    def _init_ref(self, model_provider_func, model_type):
        model = get_model(model_provider_func, model_type, wrap_with_ddp=False)
        return model

    def forward(self, seq, attention_mask, batch_size, seq_length, return_output_log_probs):
        # print(f"current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        seq = seq.cuda()
        attention_mask = attention_mask.cuda()
        with torch.no_grad():
            ref_logprobs = forward(self.model, seq, attention_mask, batch_size, seq_length, return_output_log_probs=True)
        if parallel_state.is_pipeline_first_stage():
            ref_logprobs = ref_logprobs.cpu()
        return ref_logprobs

    def print_mem(self):
        if torch.distributed.get_rank() == 0:
            print(f"current memory allocated: {torch.cuda.memory_summary()}")

    def empty_cache(self):
        torch.cuda.empty_cache()


class PPORayGroup:
    # Adopted from https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/trainer/ray/launcher.py

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type,
        pg,
        num_gpus_per_actor=1,
    ):
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self._ray_actor_type = ray_actor_type
        self._pg = pg

        self._initate_group(num_gpus_per_actor)

    def _initate_group(self, num_gpus_per_actor):
        pg = self._pg
        world_size = self._num_nodes * self._num_gpus_per_node

        # Create a placement group
        if pg:
            master_actor = self._ray_actor_type.options(
                num_cpus=num_gpus_per_actor*4,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self._ray_actor_type.options(
                num_cpus=num_gpus_per_actor, num_gpus=num_gpus_per_actor
            ).remote(world_size, 0, 0, None, None)
        self._handlers = [master_actor]

        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self._ray_actor_type.options(
                        num_cpus=num_gpus_per_actor*4,
                        num_gpus=num_gpus_per_actor,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self._ray_actor_type.options(
                        num_cpus=num_gpus_per_actor, num_gpus=num_gpus_per_actor
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                self._handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._handlers]

    def get_parallel_size(self):
        actor = self._handlers[0]
        self.tensor_model_parallel_size = ray.get(actor.get_tensor_model_parallel_size.remote())
        self.data_parallel_size = ray.get(actor.get_data_parallel_size.remote())
        self.pipeline_model_parallel_size = ray.get(actor.get_pipeline_model_parallel_size.remote())
        return self.tensor_model_parallel_size, self.data_parallel_size, self.pipeline_model_parallel_size

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def generate_sequences(self, *args, **kwargs):
        refs = self.async_run_method("generate_sequences", *args, **kwargs)
        # each data parallel group only return one
        return_refs = []
        for i in range(0, self.tensor_model_parallel_size * self.data_parallel_size, self.tensor_model_parallel_size):
            return_refs.append(refs[i])
        # the return size of `return_refs` equal dp_rank
        return return_refs

    def generate_sequences_with_prompt(self, batch_prompt):
        return_refs = []
        dp_ranks = ray.get([actor.get_data_paralle_rank.remote() for actor in self._handlers])
        for i, actor in enumerate(self._handlers):
            rank = dp_ranks[i]
            ref = actor.generate_sequences.remote(batch_prompt[rank])
            if i < self.tensor_model_parallel_size * self.data_parallel_size and i % self.tensor_model_parallel_size == 0:
                return_refs.append(ref)
        # the return size of `return_refs` equal dp_rank
        return return_refs

    def get_batch_prompt(self, step):
        return_refs = []
        for i in range(0, self.tensor_model_parallel_size * self.data_parallel_size, self.tensor_model_parallel_size):
            return_refs.append(self._handlers[i].get_batch_prompt.remote())
        # the return size of `return_refs` equal dp_rank
        return return_refs

    def forward(self, seq, attention_mask, batch_size, seq_length, return_output_log_probs=True):
        return_refs = []
        dp_ranks = ray.get([actor.get_data_paralle_rank.remote() for actor in self._handlers])
        for i, actor in enumerate(self._handlers):
            rank = dp_ranks[i]
            ref = actor.forward.remote(seq[rank], attention_mask[rank], batch_size[rank], seq_length[rank], return_output_log_probs)
            if i < self.tensor_model_parallel_size * self.data_parallel_size and i % self.tensor_model_parallel_size == 0:
                return_refs.append(ref)
        # the return size of `return_refs` equal dp_rank
        return return_refs

    def forward_value_reward(self, seq, attention_mask, batch_size, prompt_length, seq_length, PAD_ID):
        return_refs = []
        dp_ranks = ray.get([actor.get_data_paralle_rank.remote() for actor in self._handlers])
        for i, actor in enumerate(self._handlers):
            # pos = i % (self.tensor_model_parallel_size * self.data_parallel_size) // self.tensor_model_parallel_size
            rank = dp_ranks[i]
            ref = actor.forward_value_reward.remote(seq[rank], attention_mask[rank], batch_size[rank], prompt_length[rank], seq_length[rank], PAD_ID)
            if i < self.tensor_model_parallel_size * self.data_parallel_size and i % self.tensor_model_parallel_size == 0:
                return_refs.append(ref)
        # the return size of `return_refs` equal dp_rank
        return return_refs

    def forward_value(self, seq, attention_mask, batch_size, prompt_length, seq_length):
        return_refs = []
        dp_ranks = ray.get([actor.get_data_paralle_rank.remote() for actor in self._handlers])
        for i, actor in enumerate(self._handlers):
            rank = dp_ranks[i]
            ref = actor.forward_value.remote(seq[rank], attention_mask[rank], batch_size[rank], prompt_length[rank], seq_length[rank])
            if i < self.tensor_model_parallel_size * self.data_parallel_size and i % self.tensor_model_parallel_size == 0:
                return_refs.append(ref)
        # the return size of `return_refs` equal dp_rank
        return return_refs

    def add_exp_dataset(self, batch_exp):
        return_refs = []
        dp_ranks = ray.get([actor.get_data_paralle_rank.remote() for actor in self._handlers])
        for i, actor in enumerate(self._handlers):
            rank = dp_ranks[i]
            ref = actor.add_exp_dataset.remote(batch_exp[rank])
            if i == 0:
                return_refs.append(ref)
        return return_refs
