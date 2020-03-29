from typing import Tuple, List
from functools import partial

import torch
from torch import nn

__all__ = ['DeepSupervisionWrapper']


class DeepSupervisionWrapper(nn.Module):

    def __init__(self, module: nn.Module, auxiliary_modules: List[Tuple[nn.Module, nn.Module]]):
        super().__init__()

        self.module = module

        self.layers = [layer for layer, _module in auxiliary_modules]
        self.auxiliary = nn.ModuleList([
            module
            for _layer, module in auxiliary_modules
        ])

    def forward(self, input):
        if self.training:
            aux_outputs = [None for _ in range(len(self.layers))]
            hooks = []

            for id, (layer, auxiliary) in enumerate(zip(self.layers, self.auxiliary)):
                hook_fn = partial(
                    auxiliary_hook, aux_outputs=aux_outputs, aux_id=id, auxiliary_module=auxiliary)
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)

            output = self.module(input)
            for hook in hooks:
                hook.remove()
            return output, aux_outputs
        else:
            return self.module(input)


def auxiliary_hook(_module, _input, output, aux_outputs, aux_id, auxiliary_module):
    aux_outputs[aux_id] = auxiliary_module(output)
