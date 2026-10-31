"""Non-reentrant activation checkpointing, including Transformers 4.28 Llama.

Full and Mask reuse decoder parameters in a single loss/backward graph. Reentrant
checkpointing creates separate nested backwards and may invoke ZeRO-2's gradient
reduction hooks twice for the same parameter. Do not detach the shared region
tokens or alter the loss to work around this: retain the graph and use PyTorch's
non-reentrant checkpoint implementation instead.
"""
import inspect
from functools import partial, update_wrapper
from types import FunctionType, MethodType

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.llama.modeling_llama import LlamaModel


class _Namespace:
    """Delegate attributes, overriding only an explicitly supplied name."""
    def __init__(self, original, **overrides):
        self._original = original
        self._overrides = overrides

    def __getattr__(self, name):
        return self._overrides.get(name, getattr(self._original, name))


def enable_non_reentrant_checkpointing(model):
    enable = model.gradient_checkpointing_enable
    if 'gradient_checkpointing_kwargs' in inspect.signature(enable).parameters:
        enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        return

    # Transformers 4.28 hardcodes torch.utils.checkpoint.checkpoint inside
    # LlamaModel.forward and cannot accept gradient_checkpointing_kwargs.
    # Rebind only this instance's forward globals, without modifying torch,
    # the Transformers installation, model weights, or other model instances.
    enable()
    patched = 0
    for module in model.modules():
        if not isinstance(module, LlamaModel):
            continue
        if getattr(module, '_reg2rg_non_reentrant', False):
            patched += 1
            continue
        original = module.forward.__func__
        if 'torch.utils.checkpoint.checkpoint(' not in inspect.getsource(original):
            raise RuntimeError('Unknown legacy Llama checkpoint implementation; refusing a silent fallback')
        namespace = dict(original.__globals__)
        checkpoint_namespace = _Namespace(torch.utils.checkpoint,
                                          checkpoint=partial(checkpoint, use_reentrant=False))
        namespace['torch'] = _Namespace(torch, utils=_Namespace(torch.utils, checkpoint=checkpoint_namespace))
        forward = FunctionType(original.__code__, namespace, original.__name__,
                               original.__defaults__, original.__closure__)
        forward.__kwdefaults__ = original.__kwdefaults__
        update_wrapper(forward, original)
        module.forward = MethodType(forward, module)
        module._reg2rg_non_reentrant = True
        patched += 1
    if not patched:
        raise RuntimeError('No LlamaModel found for non-reentrant checkpointing')
    print(f'[checkpoint] non-reentrant=True; legacy Llama instances={patched}', flush=True)
