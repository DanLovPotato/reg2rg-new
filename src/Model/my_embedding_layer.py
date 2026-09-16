import torch.nn as nn
import torch.nn.functional as F
import torch
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from .fvlm_vit.vit import ViT as FvlmViT
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


# def load_fvlm_visual_encoder(vit_module, checkpoint_path):
#     """Loads the visual_encoder.* weights from a fvlm finetune.py checkpoint (which stores
#     the whole BlipPretrain model's state_dict under "model") into a standalone fVLM ViT.
#     """
#     ckpt = torch.load(checkpoint_path, map_location='cpu')
#     state_dict = ckpt['model'] if 'model' in ckpt else ckpt
#     prefix = 'visual_encoder.'
#     sub_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
#     vit_module.load_state_dict(sub_state_dict, strict=True)


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
        ############ finegrain_clip 用微调好的 fVLM 视觉编码器（对比学习预训练，见
        # fvlm/finetune.py），而不是随机初始化的 vit_3d.ViT
        # self.finegrain_clip_vision_encoder = FvlmViT(
        #     in_channels=1,
        #     img_size=FVLM_NATIVE_CROP_SIZE,
        #     patch_size=FVLM_PATCH_SIZE,
        #     hidden_size=vis_dim,
        #     mlp_dim=3072,
        #     num_layers=12,
        #     num_heads=12,
        #     qkv_bias=True,
        #     dropout_rate=0.1,
        # )
        ##################
 
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
        #####
        # load fine-tuned fVLM visual_encoder weights (extracted from the full BlipPretrain
        # checkpoint saved by fvlm/finetune.py)
        # if pretrained_finegrain_visual_encoder is not None:
        #     load_fvlm_visual_encoder(self.finegrain_clip_vision_encoder, pretrained_finegrain_visual_encoder)
        #     #####
 
        # frozen the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        ######    
        # for param in self.finegrain_clip_vision_encoder.parameters():
        #     param.requires_grad = False
        ######
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
        B, S, C, H, W, D = raw_image.shape

        # Both prompts independently encode their own global CT volume.
        vision_temp = rearrange(raw_image, "b S c h w d -> (b S) c h w d")
        vision_temp, _ = self.vision_encoder(vision_temp)
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        vision_temp = self.perceiver(vision_temp.unsqueeze(2))
        n = vision_temp.shape[2]
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n * S)
        image_embedding = self.fc(vision_temp)

        if precomputed_region_embedding is not None:
            # Mask prompt: do not encode crops/masks and do not run GKE, RWLKE or LIFT-GCN.
            if set(vision_x.keys()) != {'image'}:
                raise ValueError("Mask branch vision_x must contain only the global image")
            if mask_x:
                raise ValueError("Mask branch mask_x must be empty")
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

            mask_embeddings = {}
            for area, region_tensor in region_embeddings.items():
                vision_temp = rearrange(
                    region_tensor, "b S c h w d -> (b S) c h w d"
                )
                assert vision_temp.shape[1] == 3, (
                    f"RadFM vision encoder expects 3 channels, got {vision_temp.shape[1]} "
                    f"for region {area!r} with shape {tuple(vision_temp.shape)}"
                )
                vision_temp, _ = self.vision_encoder(vision_temp)
                vision_temp = rearrange(
                    vision_temp, "(b s) v d -> b s v d", b=B, s=S
                )
                vision_temp = self.perceiver(vision_temp.unsqueeze(2))
                region_n = vision_temp.shape[2]
                vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
                vision_temp = rearrange(
                    vision_temp, "(b T) d -> b T d", b=B, T=region_n * S
                )
                region_embeddings[area] = self.fc(vision_temp)

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
