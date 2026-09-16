"""Frozen fVLM organ encoder matching ``BlipPretrain.forward_test_win``.

This intentionally wraps, rather than modifies, fVLM's ViT.  Its output is the
256-D normalized image feature produced by fVLM's query-token attention pooling
and organ-specific projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import FVLM_ORGAN_MASK_ID


FVLM_ORGANS = [
    "abdomen", "bone", "breast", "esophagus", "heart", "lung",
    "mediastinum", "thyroid", "trachea and bronchie", "pleura",
]
FVLM_NATIVE_CROP_SIZE = (112, 256, 352)
FVLM_PATCH_SIZE = (16, 16, 32)


class FVLMOrganEncoder(nn.Module):
    """Exact frozen image branch of the 10-organ fVLM evaluation checkpoint."""

    def __init__(self, vit_cls, checkpoint_path):
        super().__init__()
        self.organs = tuple(FVLM_ORGANS)
        self.organ_to_index = {organ: index for index, organ in enumerate(self.organs)}
        self.visual_encoder = vit_cls(
            in_channels=1, img_size=FVLM_NATIVE_CROP_SIZE, patch_size=FVLM_PATCH_SIZE,
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            qkv_bias=True, dropout_rate=0.1,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=4, dropout=0.1, batch_first=True,
        )
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), 768))
        self.vision_projs = nn.ModuleList([nn.Linear(768, 256) for _ in self.organs])
        self._load_finetuned_checkpoint(checkpoint_path)
        self.requires_grad_(False)
        self.eval()

    def _load_finetuned_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        required_prefixes = ("visual_encoder.", "attention.", "query_tokens", "vision_projs.")
        missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in state)]
        if missing:
            raise KeyError(
                "The fVLM checkpoint must be the 10-organ finetuned checkpoint used by "
                f"eval_finetune.py; missing keys for {missing}: {checkpoint_path}"
            )
        own_state = self.state_dict()
        selected = {key: value for key, value in state.items() if key in own_state}
        missing_keys = sorted(set(own_state) - set(selected))
        unexpected = sorted(set(selected) - set(own_state))
        if missing_keys or unexpected:
            raise RuntimeError(
                f"fVLM organ-head checkpoint mismatch; missing={missing_keys}, unexpected={unexpected}"
            )
        self.load_state_dict(selected, strict=True)

    def train(self, mode=True):
        # The fVLM evaluation path always runs in eval mode.  Do not let the
        # enclosing Reg2RG model enable attention dropout on this frozen module.
        return super().train(False)

    def forward(self, images, masks, organ_name):
        """Return fVLM's normalized [batch, 256] organ image embedding.

        ``masks`` is the canonical fVLM crop mask (values use
        ``FVLM_ORGAN_MASK_ID``), aligned exactly with ``images``.
        """
        if organ_name not in self.organ_to_index:
            raise KeyError(f"Unknown fVLM organ: {organ_name!r}")
        organ_id = self.organ_to_index[organ_name]
        mask_value = FVLM_ORGAN_MASK_ID[organ_name]
        image_embeds, hidden_image_embeds = self.visual_encoder(images)
        del image_embeds  # forward_test_win pools the multi-scale features instead.

        organ_mask = masks.squeeze(1).eq(mask_value).float()
        token_mask = F.max_pool3d(
            organ_mask.unsqueeze(1), kernel_size=FVLM_PATCH_SIZE, stride=FVLM_PATCH_SIZE,
        ).flatten(1).bool()
        if not token_mask.any(dim=1).all():
            raise ValueError(f"At least one {organ_name!r} crop has no fVLM patch tokens")

        query = self.query_tokens[organ_id].view(1, 1, -1)
        features = []
        for sample_index, tokens in enumerate(token_mask):
            key_value = torch.cat(
                [level[sample_index, tokens] for level in hidden_image_embeds], dim=0,
            ).unsqueeze(0)
            pooled, _ = self.attention(query, key_value, key_value)
            features.append(pooled.squeeze(0))
        pooled = torch.cat(features, dim=0)
        return F.normalize(self.vision_projs[organ_id](pooled), dim=-1)
