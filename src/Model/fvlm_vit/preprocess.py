"""fVLM eval-compatible crop helpers for Reg2RG's processed fVLM volumes."""

import torch
import torch.nn.functional as F


CROP_SIZE = (112, 256, 352)  # (D, H, W)
PATCH_SIZE = (16, 16, 32)  # (D, H, W) - matches fvlm/finetune.py's PATCH_SIZE


#---Dan---
# Matches fvlm/finetune.py's on-disk merged-mask IDs. Pleura shares lung's mask.
FVLM_ORGAN_MASK_ID = {
    "abdomen": 1,
    "bone": 2,
    "breast": 3,
    "esophagus": 4,
    "heart": 5,
    "lung": 6,
    "mediastinum": 7,
    "thyroid": 8,
    "trachea and bronchie": 9,
    "pleura": 6,
}


def center_crop_like_fvlm_eval(
    image, mask, crop_size=CROP_SIZE, patch_size=PATCH_SIZE,
    return_mask=False, mask_value=1,
):
    """Exact tensor crop/pad path used by fvlm/eval_finetune.py.

    When an organ's mask bounding box exceeds crop_size in some dimension, the
    crop window is grown to fully contain it (matching eval_finetune.py's
    center_crop()), so the final pad target is the next multiple of patch_size
    rather than the fixed crop_size - matching eval_finetune.py's
    DivisiblePadd(k=PATCH_SIZE). Padding to a fixed crop_size instead would go
    negative here and silently truncate part of the organ back out.
    """
    coords = torch.nonzero(mask[0] > 0, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("fVLM processed mask has no voxels for this organ")

    z_min, y_min, x_min = coords.min(dim=0).values.long()
    z_max, y_max, x_max = coords.max(dim=0).values.long()
    crop_d = max(crop_size[0], (z_max - z_min).item())
    crop_h = max(crop_size[1], (y_max - y_min).item())
    crop_w = max(crop_size[2], (x_max - x_min).item())
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2
    _, depth, height, width = image.shape

    x_start = max(0, (center_x - crop_w // 2).item())
    x_end = min(width, x_start + crop_w)
    x_start = max(0, x_end - crop_w) if x_end - x_start < crop_w else x_start
    y_start = max(0, (center_y - crop_h // 2).item())
    y_end = min(height, y_start + crop_h)
    y_start = max(0, y_end - crop_h) if y_end - y_start < crop_h else y_start
    z_start = max(0, (center_z - crop_d // 2).item())
    z_end = min(depth, z_start + crop_d)
    z_start = max(0, z_end - crop_d) if z_end - z_start < crop_d else z_start

    image = image[:, z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_mask = mask[:, z_start:z_end, y_start:y_end, x_start:x_end]
    pad_d, pad_h, pad_w = (
        -(-current // step) * step - current
        for step, current in zip(patch_size, image.shape[1:])
    )
    image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
    if not return_mask:
        return image
    cropped_mask = F.pad(
        cropped_mask.to(dtype=torch.float32), (0, pad_w, 0, pad_h, 0, pad_d),
        mode="constant", value=0.0,
    )
    return image, cropped_mask * mask_value
#---Dan---


#---Dan---
# ── 在线复现 fvlm/preprocess.py 的离线流水线 ──────────────────────────────
# 让训练/推理只依赖 RadGenome 的原始数据（{split}_preprocessed + {split}_region_mask），
# 不再需要事先跑一遍 fvlm/preprocess.py 生成 processed_{split}_{images,masks}。
# 下面的常量和步骤顺序必须和 fvlm/preprocess.py 逐条对齐，否则冻结的 fVLM 编码器
# 会看到与其微调时不同的输入分布。

# fvlm/preprocess.py::ORGANS —— 只有 9 个，没有 pleura。RadGenome 的 pleura.nii.gz
# 与 lung.nii.gz 逐体素相同，单通道整数 mask 无法同时编码两者，所以 pleura 不写入
# 自己的 id，改由 FVLM_ORGAN_MASK_ID 指回 lung 的 6。
FVLM_MASK_SOURCE_ORGANS = [
    "abdomen", "bone", "breast", "esophagus", "heart",
    "lung", "mediastinum", "thyroid", "trachea and bronchie",
]

# fvlm_original/data/resize.py 的 ref_spacing。CROP_SIZE 的深度轴（112 体素）只有在
# 重采样到 3mm 层厚之后才对应预训练窗口假设的 ~336mm 解剖范围。
REF_SPACING = (1.0, 1.0, 3.0)

# fvlm/preprocess.py 在器官并集 bbox 外扩的边距
EXTEND_D = 5
EXTEND_HW = 20


def merge_organ_masks(mask_paths, loader):
    """把每器官一个的二值 {organ}.nii.gz 合成单通道多类别体积（0=背景，1..9=器官）。

    id = FVLM_MASK_SOURCE_ORGANS.index(organ) + 1，与 fvlm/preprocess.py 一致。
    重叠处后写覆盖先写，循环顺序因此不能改。
    """
    merged = None
    for organ_id, organ in enumerate(FVLM_MASK_SOURCE_ORGANS):
        path = mask_paths.get(organ)
        if path is None:
            continue
        organ_mask = loader({"label": path})["label"]
        if merged is None:
            # 从真实加载的 mask 克隆而非 zeros，保留 affine/spacing 元数据供重采样使用
            merged = organ_mask.clone()
            merged[:] = 0
        merged[organ_mask > 0] = organ_id + 1
    return merged


def build_fvlm_volume(image_path, mask_paths, image_loader, mask_loader):
    """在线生成 fVLM 分支所需的 (image, label)，等价于 fvlm/preprocess.py 的产物。

    Args:
        image_path: RadGenome 原始 CT 的 .nii.gz 路径
        mask_paths: {organ_name: {organ}.nii.gz 路径}，至少覆盖 FVLM_MASK_SOURCE_ORGANS
        image_loader / mask_loader: MONAI LoadImaged(ensure_channel_first=True)

    Returns:
        (image, label)，形状 [1, D, H, W]，D/H/W 各自 >= CROP_SIZE。
        image 已做 [-1150, 350] → [0, 1] 的强度窗；label 是 1..9 的整数器官编号。
    """
    from monai import transforms

    label = merge_organ_masks(mask_paths, mask_loader)
    if label is None:
        raise FileNotFoundError(f"没有任何器官 mask 可用于合并: {image_path}")

    data = image_loader({"image": image_path})
    image = data["image"]
    label.meta["filename_or_obj"] = image_path
    data["label"] = label

    # ① 先在原始轴序下重采样到 REF_SPACING（image/label 共用同一网格，目标尺寸相同）
    affine = image.meta["affine"]
    spacing = tuple(abs(affine[i, i].item()) for i in range(3))
    _, x, y, z = image.shape
    scale = [spacing[i] / REF_SPACING[i] for i in range(3)]
    target_size = [int(x * scale[0]), int(y * scale[1]), int(z * scale[2])]
    data = transforms.Compose([
        transforms.Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
        transforms.Resized(keys=["label"], spatial_size=target_size, mode="nearest"),
    ])(data)

    # ② 转成 (C, D, H, W) 轴序 + 强度窗（label 是整数编号，不做强度变换）
    data = transforms.Compose([
        transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True,
        ),
    ])(data)

    image, label = data["image"], data["label"]
    organ_ids_before = label.unique()

    # ③ 裁到器官并集的 bbox + 边距
    coords = torch.nonzero(label[0] > 0, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError(f"合并后的 mask 没有任何器官体素: {image_path}")
    lo = coords.min(dim=0).values
    hi = coords.max(dim=0).values
    margin = torch.tensor([EXTEND_D, EXTEND_HW, EXTEND_HW])
    lo = torch.maximum(lo - margin, torch.zeros(3, dtype=lo.dtype))
    hi = torch.minimum(
        hi + margin, torch.tensor(image.shape[1:], dtype=hi.dtype)
    )
    image = image[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    label = label[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    # 与 fvlm/preprocess.py 相同的完整性检查：裁剪不能把某个器官整个切掉
    assert torch.all(organ_ids_before == label.unique()), f"裁剪丢失器官: {image_path}"

    # ④ 补零到至少 CROP_SIZE（只往大补，不往小裁）
    data = transforms.Compose([
        transforms.SpatialPadd(keys=["image"], spatial_size=CROP_SIZE,
                               mode="constant", constant_values=0),
        transforms.SpatialPadd(keys=["label"], spatial_size=CROP_SIZE,
                               mode="constant", constant_values=0),
    ])({"image": image, "label": label})

    return data["image"], data["label"]
#---Dan---
