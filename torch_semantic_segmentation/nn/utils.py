from functools import partial
import torch

__all__ = ['CaptureOutput']


class CaptureOutput(object):

    def __init__(self, module, layers):
        self.module = module
        self.layers = layers

    def __enter__(self):
        self.outputs = {}

        self._hooks = []

        for name, layer in self.layers.items():
            hook_fn = partial(auxiliary_hook, outputs=self.outputs, name=name)
            hook = layer.register_forward_hook(hook_fn)
            self._hooks.append(hook)

        return self

    def __exit__(self, _type, _value, _tb):
        for hook in self._hooks:
            hook.remove()
        del self._hooks


def auxiliary_hook(_module, _input, output, outputs, name):
    outputs[name] = output
