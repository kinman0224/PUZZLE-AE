from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model

def set_model_mpu(model, _mpu):
    for module in model:
        get_attr_wrapped_model(module, "set_mpu")(_mpu)

def get_model_mpu_index(model):
    if type(model) is list:
        model = model[0]
    return get_attr_wrapped_model(model, "get_mpu")()._INDEX

def apply_model_mpu(model):
    idx = get_model_mpu_index(model)
    parallel_state.switch_mpu_by_index(idx)
