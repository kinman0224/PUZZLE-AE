import torch

from megatron import get_args
from megatron.core.utils import get_attr_wrapped_model

def get_configs(config):
    args = get_args()
    num_layers = getattr(config, "num_layers",
                         getattr(config, "n_layer", None))
    hidden_size = getattr(config, "hidden_size",
                          getattr(config, "n_embd", None))
    vocab_size = getattr(args, "padded_vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size


def calculate_flops(config, checkpoint_activations_factor, batch_size, seq_length):
    num_layers, hidden_size, vocab_size = get_configs(config)

    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size *
                           seq_length * num_layers * (hidden_size**2)) * (
                               1.0 + (seq_length / (6.0 * hidden_size)) +
                               (vocab_size /
                                (16.0 * num_layers * hidden_size)))
    return flops_per_iteration

def calculate_gen_flops(config, batch_size, seq_length, gen_exp_time, gpus_per_model):
    num_layers, hidden_size, vocab_size = get_configs(config)
    gen_flops_per_iteration = (
            24 * batch_size * seq_length * num_layers *
            (hidden_size**2)) * (
                1.0 + (seq_length / (6.0 * hidden_size)) +
                (vocab_size /
                 (16.0 * num_layers * hidden_size)))
    gen_tflops = gen_flops_per_iteration / (gen_exp_time * gpus_per_model *
                                                (10**12))
    return gen_tflops, gen_flops_per_iteration

def print_throughput(trainer, gen_exp_time, train_time, e2e_time):
    args = get_args()

    gpus_per_model = torch.distributed.get_world_size()
    seq_length = args.max_answer_seq_len + args.max_prompt_seq_len
    batch_size = args.global_batch_size
    samples_per_second = batch_size / e2e_time

    actor_checkpoint_activations_factor = 4 if args.recompute_granularity is not None else 3
    critic_checkpoint_activations_factor = 4 if args.recompute_granularity is not None else 3

    actor_config = get_attr_wrapped_model(trainer.actor_model[0], 'config')
    actor_train_flops_per_iteration = calculate_flops(actor_config, actor_checkpoint_activations_factor, batch_size, seq_length)
    critic_config = get_attr_wrapped_model(trainer.critic_model[0], 'config')
    critic_train_flops_per_iteration = calculate_flops(critic_config, critic_checkpoint_activations_factor, batch_size, seq_length)

    total_train_flops = actor_train_flops_per_iteration + critic_train_flops_per_iteration
    train_tflops = total_train_flops / (train_time * gpus_per_model * (10**12))

    gen_tflops, gen_flops_per_iteration = calculate_gen_flops(actor_config,
                                                              batch_size,
                                                              seq_length,
                                                              gen_exp_time,
                                                              gpus_per_model)

    total_flops_per_iteration = total_train_flops + gen_flops_per_iteration
    total_tflops = total_flops_per_iteration / (e2e_time * gpus_per_model * (10**12))

    print(
        f"End-to-End => Latency: {e2e_time:>6.2f}s, TFLOPs: {total_tflops:>6.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )
    print(
        f"Generation => Latency: {gen_exp_time:>6.2f} s, TFLOPs: {gen_tflops:>6.2f}, Answer Seq. Length: {args.max_answer_seq_len}"
        )
    print(
        f"Training   => Latency: {train_time:>6.2f} s, TFLOPs: {train_tflops:>6.2f}"
        )