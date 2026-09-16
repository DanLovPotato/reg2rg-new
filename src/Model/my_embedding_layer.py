import torch.nn as nn
import torch.nn.functional as F
import torch
#---Dan---
import os
import sys
#---Dan---
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel
from monai.networks.nets.swin_unetr import SwinTransformer
from .cross_attention import TwoWayTransformer
from .cross_modal_knowledge_enhancer import (
    CrossModalKnowledgeEnhancer,
    RegionWiseLocalKnowledgeEnhancer,
    GlobalKnowledgeEnhancerWithKSAP,
)
from .fvlm_vit.organ_encoder import FVLMOrganEncoder
import numpy as np
import json
CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]
 
SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]
 
# img_size/patch_size that the fVLM checkpoint's visual_encoder was pretrained/finetuned at
# (see fvlm/finetune.py CROP_SIZE/PATCH_SIZE) - (D, H, W) order. Only used to construct the
# module and size its position_embeddings table; PatchEmbeddingBlock.forward() interpolates
# that table to whatever (D, H, W) actually comes in at runtime, so callers aren't limited to
# this exact input shape as long as they pass axes in (D, H, W) order (see forward() below).
FVLM_NATIVE_CROP_SIZE = (112, 256, 352)
FVLM_PATCH_SIZE = (16, 16, 32)


#---Dan---
def get_original_fvlm_vit():
    """Return the exact ViT class imported by fvlm/eval_finetune.py."""
    fvlm_root = os.environ.get(
        "FVLM_SOURCE_ROOT", "/home/wisc/dxiang23/Projects/chestCT/code/fvlm"
    )
    if not os.path.isfile(os.path.join(fvlm_root, "lavis", "models", "blip_models", "vit.py")):
        raise FileNotFoundError(
            "fVLM source tree is required for exact encoder parity; set "
            f"FVLM_SOURCE_ROOT. Looked in: {fvlm_root}"
        )
    if fvlm_root not in sys.path:
        sys.path.insert(0, fvlm_root)
    from lavis.models.blip_models.vit import ViT
    return ViT


def load_fvlm_visual_encoder(vit_module, checkpoint_path):
    """Load only fVLM's visual encoder from its full BlipPretrain checkpoint."""
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint_data['model'] if 'model' in checkpoint_data else checkpoint_data
    prefix = 'visual_encoder.'
    visual_encoder_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not visual_encoder_state:
        raise KeyError(f"No {prefix!r} weights found in fVLM checkpoint: {checkpoint_path}")
    vit_module.load_state_dict(visual_encoder_state, strict=True)
#---Dan---


REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]
 
class MyEmbedding(nn.Module):
    def __init__(self, pretrained_visual_encoder=None, pretrained_finegrain_visual_encoder=None, pretrained_adapter=None, bank_npy_path=None, organ_annotation_path=None, num_embeddings=32000, embedding_dim=4096, perceiver_num=32, vis_dim=768, patch_size=32, frame_patch_size=4, seg_channel=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.torch.randn((num_embeddings, embedding_dim)), requires_grad=True)  # NOTE: will be initialized using the weight from MedLLaMA
        self.image_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.region_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel
        self.region_token_len = perceiver_num + 1
 
        self.vision_encoder = ViT(
            image_size=512,          # image size
            frames=512,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=frame_patch_size,      # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        #---Dan---
        self.use_fvlm = pretrained_finegrain_visual_encoder is not None
        if self.use_fvlm:
            self.finegrain_clip_vision_encoder = FVLMOrganEncoder(
                get_original_fvlm_vit(), pretrained_finegrain_visual_encoder,
            )
            # Not an fVLM module: this is only the bridge from exact, 256-D
            # fVLM feature space to the existing 4096-D Reg2RG token interface.
            self.fvlm_to_llm = nn.Linear(256, embedding_dim)
        #---Dan---
 
        self.mask_encoder = ViT(
            image_size=256,          # image size
            frames=64,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=16,      # frame patch size
            dim=255,
            depth=3,
            heads=8,
            mlp_dim=512,
            channels = 1,
            dropout=0.1,
            emb_dropout=0.1
        )
 
        # load pretrained vision encoder from RadFM
        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location='cpu')
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # frozen the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        #---Dan---
        if self.use_fvlm:
            for param in self.finegrain_clip_vision_encoder.parameters():
                param.requires_grad = False
        #---Dan---
        self.vis_dim = vis_dim
 
        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, num_latents=perceiver_num)
        # load pretrained perceiver and fc from RadFM
        if pretrained_adapter is not None:
            state_dict = torch.load(pretrained_adapter, map_location='cpu')
            self.perceiver.load_state_dict(state_dict['perceiver'])
            # self.fc.load_state_dict(state_dict['fc'])
       
        # self.cross_attn = TwoWayTransformer(
        #     depth=3,
        #     embedding_dim=self.vis_dim,
        #     num_heads=8,
        #     mlp_dim=1024,
        # )
       
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        self.mask_fc = nn.Linear(255, self.embedding_dim)

        # The knowledge bank is optional for lightweight training/debugging.
        self.report_id_to_index = None
        self.report_organ_to_index = None
        if bank_npy_path is not None:
            bank_raw = np.load(bank_npy_path, allow_pickle=True)
            if isinstance(bank_raw, np.lib.npyio.NpzFile):
                if "data" in bank_raw.files:
                    bank_array = bank_raw["data"]
                elif "feats" in bank_raw.files:
                    bank_array = bank_raw["feats"]
                    if bank_array.ndim != 3:
                        raise ValueError(f"feats must have shape [reports, organs, dim], got {bank_array.shape}")
                    patient_ids = bank_raw["patient_ids"].tolist()
                    bank_organs = bank_raw["organs"].tolist()
                    self.report_id_to_index = {report_id: index for index, report_id in enumerate(patient_ids)}
                    self.report_organ_to_index = {organ: index for index, organ in enumerate(bank_organs)}
                else:
                    raise KeyError(f"Unsupported NPZ keys: {bank_raw.files}")
            elif bank_raw.dtype == object:
                bank_array = bank_raw.item()["data"]
            else:
                bank_array = bank_raw
            if bank_array.ndim not in (2, 3):
                raise ValueError(f"report bank must be 2-D or 3-D, got shape {bank_array.shape}")
            self.register_buffer("report_bank", torch.from_numpy(bank_array).float(), persistent=False)
            bank_dim = bank_array.shape[-1]
            self.gke = CrossModalKnowledgeEnhancer(d_model=embedding_dim, bank_dim=bank_dim)
            self.rwlke = RegionWiseLocalKnowledgeEnhancer(organs_list=REGIONS, d_model=embedding_dim, bank_dim=bank_dim)
            self.global_gke = GlobalKnowledgeEnhancerWithKSAP(
                organs_list=REGIONS, d_model=embedding_dim
            )
        else:
            self.register_buffer("report_bank", None, persistent=False)
            self.gke = None
            self.rwlke = None
            self.global_gke = None

        self.organ_annotation_index = None
        self._warned_missing_annotations = set()  # --Dan---
        if organ_annotation_path is not None:
            with open(organ_annotation_path, "r", encoding="utf-8") as annotation_file:
                annotation_raw = json.load(annotation_file)
            if isinstance(annotation_raw, dict):
                annotation_records = annotation_raw.get(
                    "validation", annotation_raw.get("train", annotation_raw)
                )
            else:
                annotation_records = annotation_raw
            if not isinstance(annotation_records, list):
                raise ValueError("organ annotation must contain a train or validation list")
            self.organ_annotation_index = {record["id"]: record for record in annotation_records}

    #---Dan---
    def train(self, mode=True):
        """Keep frozen visual backbones deterministic while adapters train."""
        super().train(mode)
        self.vision_encoder.eval()
        if self.use_fvlm:
            self.finegrain_clip_vision_encoder.eval()
        return self
    #---Dan---


    def _select_organ_annotations(self, sample_ids, region2areas):
        if self.organ_annotation_index is None:
            return None
        if sample_ids is None:
            raise ValueError("sample_ids are required when organ annotations are enabled")

        selected = []
        for sample_id, sample_regions in zip(sample_ids, region2areas):
            record = self.organ_annotation_index.get(sample_id)
            organ_names = sample_regions.values() if isinstance(sample_regions, dict) else sample_regions
            if record is None:
                # --Dan--- organ_annotation.json is missing ~12/24128 sample ids
                # (e.g. train_8039_c_1) that are otherwise valid rows in
                # train_region_report.csv; fall back instead of crashing the run.
                if sample_id not in self._warned_missing_annotations:
                    self._warned_missing_annotations.add(sample_id)
                    print(f"[my_embedding_layer] WARNING: no organ annotation found for "
                          f"sample {sample_id!r}; using empty neighbor lists for this sample.")
                selected.append({organ: [] for organ in organ_names})
                continue
            selected.append({
                organ: record.get(f"{organ}_indices", [])[:5]
                for organ in organ_names
            })
        return selected

    def _lookup_annotation_vectors(self, organ_annotations):
        if organ_annotations is None:
            return None
        if self.report_id_to_index is None or self.report_organ_to_index is None:
            raise ValueError("The report bank must provide feats, patient_ids, and organs")

        organ_report_vectors = []
        bank_dim = self.report_bank.shape[-1]
        for sample_annotations in organ_annotations:
            sample_organ_vectors = {}
            for organ, neighbors in sample_annotations.items():
                vectors = []
                for neighbor in neighbors:
                    report_id = neighbor["id"]
                    report_organ = neighbor.get("organ", organ)
                    report_index = self.report_id_to_index.get(report_id)
                    organ_index = self.report_organ_to_index.get(report_organ)
                    if report_index is None or organ_index is None:
                        continue
                    vectors.append(self.report_bank[report_index, organ_index])
                if vectors:
                    organ_vectors = torch.stack(vectors, dim=0)
                else:
                    organ_vectors = self.report_bank.new_empty((0, bank_dim))
                sample_organ_vectors[organ] = organ_vectors

            organ_report_vectors.append(sample_organ_vectors)

        return organ_report_vectors

    def forward(
        self,
        vision_x,
        mask_x,
        fvlm_mask_x,
        text_input,
        region2areas,
        sample_ids=None,
        precomputed_region_embedding=None,
        return_rwlke_region_embedding=False,
    ):
        """Build the two visual streams used by the Full and Mask prompts.

        Full branch:
            full global CT -> encoder/adapter -> global report attention -> LIFT-GCN
            organ crops/masks -> encoder/adapter -> RWLKE region tokens

        Mask branch:
            masked global CT -> encoder/adapter only
            region tokens are supplied by Trainer: visible organs copy Full RWLKE
            tokens and dropped organs use zero placeholders.
        """
        raw_image = vision_x['image']
        # 
        B, S, C, H, W, D = raw_image.shape

        # Both prompts independently encode their own global CT volume.
        # reshape维度
        vision_temp = rearrange(raw_image, "b S c h w d -> (b S) c h w d")
        # 把上面整理好的 3D CT 数据真正送进 3D Vision Transformer (ViT) 编码器做特征提取
        vision_temp, _ = self.vision_encoder(vision_temp)
        # 当时把 B 和 S 合并成一个维度送进 ViT；现在编码完了，就把这个合并维度重新拆成 B 和 S 两个独立维度，这样才能知道"哪几个 token 属于哪个病人的第几次扫描
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        # Perceiver Resampler 把这些不定长的视觉特征压缩成固定数量（n 个）的紧凑 token，既降低了序列长度、又能适配 LLM 输入长度的要求
        vision_temp = self.perceiver(vision_temp.unsqueeze(2))
        # latent token 数量 = 32， 32 就是"每次扫描固定压缩成 32 个 token
        n = vision_temp.shape[2]
        # 
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n * S)
        # Global features(最上面那个分支)
        image_embedding = self.fc(vision_temp)

        if precomputed_region_embedding is not None:
            # Mask prompt: do not encode crops/masks and do not run GKE, RWLKE or LIFT-GCN.
            if set(vision_x.keys()) != {'image'}:
                raise ValueError("Mask branch vision_x must contain only the global image")
            if mask_x:
                raise ValueError("Mask branch mask_x must be empty")
            if fvlm_mask_x:
                raise ValueError("Mask branch fvlm_mask_x must be empty")
            if precomputed_region_embedding.size(0) != B:
                raise ValueError("Precomputed region embedding batch size does not match image batch")
            if precomputed_region_embedding.size(-1) != self.embedding_dim:
                raise ValueError("Precomputed region embedding has the wrong feature dimension")

            enhanced_image_embedding = image_embedding
            rwlke_region_embedding = precomputed_region_embedding
        else:
            # Full prompt: organ crops and masks are required to produce RWLKE tokens.
            region_embeddings = {
                area: tensor for area, tensor in vision_x.items() if area != 'image'
            }
            if not region_embeddings:
                raise ValueError("Full branch requires at least one organ crop")
            missing_masks = set(region_embeddings) - set(mask_x)
            if missing_masks:
                raise KeyError(f"Missing organ masks for: {sorted(missing_masks)}")
            missing_fvlm_masks = set(region_embeddings) - set(fvlm_mask_x)
            if missing_fvlm_masks:
                raise KeyError(f"Missing fVLM crop masks for: {sorted(missing_fvlm_masks)}")

            mask_embeddings = {}
            for area, region_tensor in region_embeddings.items():
                vision_temp = rearrange(
                    region_tensor, "b S c h w d -> (b S) c h w d"
                )
                #---Dan---
                # This is the original fine-grained region-encoder location.
                # It always uses fVLM, never the RadFM global-image encoder.
                if not self.use_fvlm:
                    raise RuntimeError(
                        "pretrained_finegrained_visual_encoder is required for "
                        "Reg2RG region encoding"
                    )
                #---Dan---
                # center_crop_like_fvlm_eval grows the crop past FVLM_NATIVE_CROP_SIZE
                # (matching fvlm/eval_finetune.py's own center_crop) whenever an organ's
                # mask bounding box exceeds it, then pads only up to the next multiple of
                # FVLM_PATCH_SIZE - so crops are >= native size, divisible by patch size,
                # but not always exactly native size. PatchEmbeddingBlock interpolates its
                # position-embedding table to whatever (D, H, W) actually comes in.
                crop_dims = vision_temp.shape[2:]
                shape_ok = (
                    vision_temp.shape[1] == 1
                    and len(crop_dims) == len(FVLM_NATIVE_CROP_SIZE)
                    and all(d >= n for d, n in zip(crop_dims, FVLM_NATIVE_CROP_SIZE))
                    and all(d % p == 0 for d, p in zip(crop_dims, FVLM_PATCH_SIZE))
                )
                if not shape_ok:
                    raise ValueError(
                        f"fVLM expects (1, D, H, W) crops with D,H,W >= "
                        f"{FVLM_NATIVE_CROP_SIZE} and divisible by {FVLM_PATCH_SIZE}, "
                        f"got {tuple(vision_temp.shape[1:])} for {area!r}"
                    )
                #---Dan---
                fvlm_mask = rearrange(
                    fvlm_mask_x[area], "b S c h w d -> (b S) c h w d"
                )
                # This 256-D feature is exactly fVLM's image branch output:
                # ViT -> multi-scale masked query attention -> organ projection
                # -> L2 normalization.  Only the adapter afterwards is Reg2RG-specific.
                fvlm_feature = self.finegrain_clip_vision_encoder(
                    vision_temp, fvlm_mask, area,
                )
                region_embeddings[area] = self.fvlm_to_llm(fvlm_feature)
                region_embeddings[area] = rearrange(
                    region_embeddings[area], "(b s) d -> b s 1 d", b=B, s=S
                ).expand(-1, -1, self.region_token_len - 1, -1)
                region_embeddings[area] = rearrange(
                    region_embeddings[area], "b s n d -> b (s n) d"
                )

                mask_embedding, _ = self.mask_encoder(mask_x[area])
                mask_embedding = torch.mean(mask_embedding, dim=1)
                mask_embeddings[area] = self.mask_fc(mask_embedding)

            for area in region_embeddings:
                region_embeddings[area] = torch.cat(
                    [region_embeddings[area], mask_embeddings[area].unsqueeze(1)],
                    dim=1,
                )
                if region_embeddings[area].size(1) != self.region_token_len:
                    raise ValueError(
                        f"Expected {self.region_token_len} tokens for {area!r}, "
                        f"got {region_embeddings[area].size(1)}"
                    )

            max_region = len(region_embeddings)
            vision_region_embedding = image_embedding.new_zeros(
                (B, self.region_token_len * max_region, self.embedding_dim)
            )
            for batch_index, sample_regions in enumerate(region2areas):
                for region_index in range(len(sample_regions)):
                    area = sample_regions[region_index]
                    start = region_index * self.region_token_len
                    end = start + self.region_token_len
                    vision_region_embedding[batch_index, start:end] = (
                        region_embeddings[area][batch_index]
                    )

            organ_annotations = self._select_organ_annotations(
                sample_ids, region2areas
            )
            organ_report_vectors = self._lookup_annotation_vectors(
                organ_annotations
            )
            if self.report_bank is not None:
                # Region stream: RWLKE output is kept separate and is the source
                # copied into the Mask prompt for all visible organs.
                rwlke_region_embedding = self.rwlke(
                    vision_region_embedding,
                    region2areas,
                    organ_report_vectors,
                )

                # Global stream: report attention followed by LIFT-GCN.
                enhanced_image_embedding = self.gke(
                    image_embedding, organ_report_vectors
                )
                local_features = {}
                for organ in REGIONS:
                    sample_features = []
                    organ_is_present = False
                    for sample_index, sample_regions in enumerate(region2areas):
                        organ_names = (
                            list(sample_regions.values())
                            if isinstance(sample_regions, dict)
                            else list(sample_regions)
                        )
                        if organ in organ_names:
                            region_index = organ_names.index(organ)
                            start = region_index * self.region_token_len
                            sample_features.append(
                                rwlke_region_embedding[
                                    sample_index,
                                    start:start + self.region_token_len,
                                ]
                            )
                            organ_is_present = True
                        else:
                            sample_features.append(
                                torch.zeros_like(
                                    rwlke_region_embedding[
                                        sample_index, :self.region_token_len
                                    ]
                                )
                            )
                    if organ_is_present:
                        local_features[organ] = torch.stack(
                            sample_features, dim=0
                        )

                graph_global, _ = self.global_gke(local_features)
                enhanced_image_embedding = (
                    enhanced_image_embedding + graph_global.unsqueeze(1)
                )
            else:
                enhanced_image_embedding = image_embedding
                rwlke_region_embedding = vision_region_embedding

        embedding_weight = torch.cat(
            [self.weight, self.image_token_weight, self.region_token_weight],
            dim=0,
        ).unsqueeze(0).repeat(B, 1, 1)
        embedding_weight = torch.cat(
            [
                embedding_weight,
                enhanced_image_embedding,
                rwlke_region_embedding,
            ],
            dim=1,
        )

        text_input_one_hot = F.one_hot(
            text_input, embedding_weight.shape[1]
        ).to(device=text_input.device, dtype=embedding_weight.dtype)
        output = torch.matmul(text_input_one_hot, embedding_weight)

        if return_rwlke_region_embedding:
            return output, rwlke_region_embedding
        return output
