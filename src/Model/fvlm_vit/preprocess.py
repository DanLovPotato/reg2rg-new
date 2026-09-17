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
