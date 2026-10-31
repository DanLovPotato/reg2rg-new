"""Tiny CPU regression test against real Llama/LoRA; no pretrained weights."""
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Model.checkpoint_compat import enable_non_reentrant_checkpointing


def run(mode):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=32, use_cache=False,
    )), LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, lora_dropout=0.0))
    model.enable_input_require_grads()
    model.train()
    if mode == 'legacy':
        model.gradient_checkpointing_enable()
    elif mode == 'fixed':
        enable_non_reentrant_checkpointing(model)
        enable_non_reentrant_checkpointing(model)  # idempotent
    keys = list(model.state_dict())
    counts = Counter()
    handles = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            def record(grad, name=name):
                counts[name] += 1
                return grad
            handles.append(parameter.register_hook(record))
    # Both branches receive differentiable shared region tokens. Do not detach.
    region = torch.randn(1, 3, 16, requires_grad=True)
    full_image = torch.randn(1, 2, 16, requires_grad=True)
    mask_image = torch.randn(1, 2, 16, requires_grad=True)
    full_input = torch.cat([full_image, region], dim=1)
    mixed = region.clone()
    mixed[:, 0, :] = 0
    mask_input = torch.cat([mask_image, mixed], dim=1)
    labels = torch.tensor([[2, 3, 4, 5, 6]])
    a = model(inputs_embeds=full_input, labels=labels).loss
    b = model(inputs_embeds=mask_input, labels=labels).loss
    loss = a + 0.8 * b
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    grads.update(region=region.grad.clone(), full_image=full_image.grad.clone(), mask_image=mask_image.grad.clone())
    assert keys == list(model.state_dict()), 'Checkpoint compatibility must not change state dict keys'
    return loss.detach(), grads, counts


if __name__ == '__main__':
    global_checkpoint = torch.utils.checkpoint.checkpoint
    reference, reference_grads, _ = run('off')
    legacy, _, legacy_counts = run('legacy')
    fixed, fixed_grads, fixed_counts = run('fixed')
    assert torch.utils.checkpoint.checkpoint is global_checkpoint
    torch.testing.assert_close(fixed, reference)
    assert reference_grads.keys() == fixed_grads.keys()
    for name in reference_grads:
        torch.testing.assert_close(fixed_grads[name], reference_grads[name], rtol=1e-4, atol=1e-6)
    assert max(legacy_counts.values()) > 1, legacy_counts
    assert max(fixed_counts.values()) == 1, fixed_counts
    assert fixed_grads['region'].abs().sum() > 0
    print('PASS: matching dual-branch loss and all gradients; shared region gradients preserved;')
    print('legacy max hook calls=', max(legacy_counts.values()), 'fixed=', max(fixed_counts.values()))
