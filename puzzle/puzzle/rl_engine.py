import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.core.enums import ModelType
from megatron.model import Float16Module
from megatron.training import get_megatron_optimizer, get_optimizer_param_scheduler

from puzzle.utils.model.reward_model import RewardWrapper

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

class RLHFEngine():

    def __init__(self, model_provider_func, model_type, tokenizer):
        self.tokenizer = tokenizer

        self.actor, self.actor_optimizer, self.actor_opt_param_scheduler = \
            self._init_actor(model_provider_func, model_type)

        self.ref = self._init_ref(model_provider_func, model_type)

        self.critic, self.critic_optimizer, self.critic_opt_param_scheduler = \
            self._init_critic(model_provider_func, model_type)

        self.reward = self._init_reward(model_provider_func, model_type)

    def _init_actor(self, model_provider_func, model_type,
                    no_wd_decay_cond=None,
                    scale_lr_cond=None,
                    lr_mult=1.0):
        model = get_model(model_provider_func, model_type)
        optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        return model, optimizer, opt_param_scheduler

    def _init_ref(self, model_provider_func, model_type):
        model = get_model(model_provider_func, model_type, wrap_with_ddp=False)
        return model

    def _init_critic(self, model_provider_func, model_type,
                    no_wd_decay_cond=None,
                    scale_lr_cond=None,
                    lr_mult=1.0):
        model = get_reward_model(model_provider_func, model_type)
        optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        return model, optimizer, opt_param_scheduler

    def _init_reward(self, model_provider_func, model_type):
        model = get_reward_model(model_provider_func, model_type, wrap_with_ddp=False)
        return model
