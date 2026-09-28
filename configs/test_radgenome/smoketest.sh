# 在完整验证集上跑 test_radgenome.py 推理。
# ckpt 是自己训的 pytorch_model_10052.bin，带 fVLM organ branch 和 knowledge bank。

# Device settings — EDIT to match the server (run `nvidia-smi` there first)
cuda_devices="3"

# EDIT this one line to wherever you place data + checkpoints on the server;
# everything below is derived from it.
remote_base="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT"

# Paths — mirrors the folder layout under chestCT/ here:
# llama_weights/, Reg2RG_weights/, smoke_data/reg2rg_data/dataset/...
lang_encoder_path="$remote_base/weights/llama_weights"
tokenizer_path="$remote_base/weights/llama_weights"
pretrained_visual_encoder="$remote_base/weights/Reg2RG_weights/RadFM_vit3d.pth"
pretrained_finegrained_visual_encoder="$remote_base/weights/fvlm_weights/finetuned_9_8/checkpoint_040.pth"
pretrained_adapter="$remote_base/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"
ckpt_path="$remote_base/outputs/pytorch_model_10052.bin"
data_folder="$remote_base/data/dataset/valid_preprocessed"
mask_folder="$remote_base/data/dataset/valid_region_mask"
report_file="$remote_base/data/dataset/radgenome_files/validation_region_report.csv"
monai_cache_dir="$remote_base/data/dataset/cache"
result_path="$remote_base/Reg2RG/results/Reg2RG_radgenome/inference_valid_full.csv"

# Knowledge bank：标注是 valid 样本，检索库是训练集报告
organ_annotation_path="$remote_base/data/dataset/EK_files_val_unique/organ_annotation.json"
bank_npy_path="$remote_base/data/dataset/EK_files_val_unique/organ_report_embeddings.npz"
