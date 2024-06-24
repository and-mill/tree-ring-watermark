# Tool to inspect the unets of pipes


import torch

from optim_utils import *
from io_utils import *

import matplotlib.pyplot as plt

from andmill_utils import MODELS, get_pipe


class WeightTensorFinder:
    def __init__(self, obj, max_depth=4):
        self.obj = obj
        self.max_depth = max_depth

    def __iter__(self):
        yield from self._find_weight_tensors(self.obj, [])

    def _find_weight_tensors(self, obj, path):
        if self.max_depth is not None and len(path) > self.max_depth:
            return

        if isinstance(obj, torch.Tensor):# and 'weight' in path[-1]:
            yield '.'.join(path), obj
        elif hasattr(obj, '__dict__'):
            for attr in dir(obj):
                if not attr.startswith('__') and hasattr(obj, attr):
                    new_path = path + [attr]
                    try:
                        sub_obj = getattr(obj, attr)
                        yield from self._find_weight_tensors(sub_obj, new_path)
                    except Exception:
                        pass


# MAIN
for model in MODELS:

    # Create an instance of the finder
    pipe = get_pipe(model)
    finder = WeightTensorFinder(pipe.unet, max_depth=6)

    # Use the finder to retrieve tensors which are pobably weights
    print(f"Found weight tensors for {model}:")
    for key, tensor in finder:
        print(f'\t{key}, {tensor.shape}')
