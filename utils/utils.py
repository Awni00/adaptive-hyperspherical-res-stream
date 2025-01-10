import  math
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


# region: AttributeDict

class AttributeDict(Dict):
    """
    A drop-in replacement for a Python dictionary, with the additional functionality to access and modify keys
    through attribute lookup for convenience.

    Based on a class in the pytorch lightning utilities subpackage
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # recursively convert nested dictionaries to AttributeDicts
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttributeDict(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delattr__(self, item: str) -> None:
        if item not in self:
            raise KeyError(item)
        del self[item]

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        # rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        rows = []
        for n in sorted(self.keys()):
            val = self[n]
            if isinstance(val, AttributeDict):
                val = val.__repr__()
                # add newline at begining and tab at start of each line
                val = "\n\t" + val.replace("\n", "\n\t")
            rows.append(tmp_name.format(f'"{n}":', val))
        return "\n".join(rows)

    def todict(self) -> dict:
        """
        Convert the `AttributeDict` to a dictionary.

        Returns
        -------
        dict
            The dictionary.
        """
        return {k: (v.todict() if isinstance(v, AttributeDict) else v) for k, v in self.items()}

# endregion

# region: dictionary utilities

def flatten_dict(dd, separator='.', prefix=''):
    """flattens a nested dictionary"""

    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def get_unique_dict_vals(dict_list):
    """
    given a list of dictionaries, return a (flattened) dictionary of unique values for each key

    Parameters:
    ----------
        dict_list (list): list of dictionaries

    Returns:
    -------
        dict: dictionary of unique values for each key. keys are strings, values are sets
    """

    config_key_vals = dict()

    for dict_ in dict_list:
        dict_ = flatten_dict(dict_)

        for key, value in dict_.items():
            if key not in config_key_vals:
                config_key_vals[key] = set()
            if isinstance(value, list):
                value = tuple(value)
            config_key_vals[key].add(value)

    return config_key_vals

# endregion

# region: general helpers and utilities

def print_gpu_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

def format_large_number(number):
    if number < 1e3:
        return str(number)
    elif number < 1e6:
        return f'{number/1e3:.1f}K'
    elif number < 1e9:
        return f'{number/1e6:.1f}M'
    else:
        return f'{number/1e9:.1f}B'

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cast_tuple(t, length = 1):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert len(out) == length
    return out

# endregion

# region tensor and pytorch utils

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# endregion

# region Training utilities, e.g., learning rate scheduling, etc.

def get_cosine_schedule_with_warmup(optimizer, max_lr, lr_decay_steps, warmup_iters=None, min_lr=None):
    """
    Returns a learning rate schedule that increases linearly to max_lr in the first warmup_steps,
    then decays via a cosine curve back down to min_lr by lr_decay_steps (then stays constant).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.
    min_lr : float
        The minimum learning rate. Default is 0.
    max_lr : float
        The maximum learning rate.
    warmup_steps : int
        The number of steps to increase the learning rate linearly in first phase. If None, defaults to 0.05 * lr_decay_steps.
    lr_decay_steps : int
        The total number of steps to decay lr, after which lr is constant at min_lr.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler.
    """

    warmup_iters = warmup_iters if warmup_iters is not None else int(lr_decay_steps * 0.05)
    min_lr = min_lr if min_lr is not None else max_lr * 0.01

    # NOTE: pytorch schedulers scale the optimizer's lr by the value returned,
    # so we need to return ratio between desired lr at given step and the optimizer's lr param
    def lr_lambda(it):

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return (it + 1) / (warmup_iters + 1)

        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_steps:
            return min_lr / max_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_steps - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1

        return (min_lr + coeff * (max_lr - min_lr)) / max_lr

    return LambdaLR(optimizer, lr_lambda)
