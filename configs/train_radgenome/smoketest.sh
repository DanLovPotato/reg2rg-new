# Smoke-test config: verify train_radgenome.py runs end-to-end on the
# 500-case validation subset (used here as a stand-in for train data —
# this matches the ModelArguments/DataArguments defaults in
# src/args/train_radgenome/jhcpu7.py, which also point at the "valid_*" files).

# Experiment settings
experiment_name="Reg2RG_8_25"
bf16=True

# Device settings — EDIT to match the server (run `nvidia-smi` there first)
cuda_devices="0,1,2"

# Torchrun settings
master_port=25368

# EDIT this one line to wherever you place data + checkpoints on the server;
# everything below is derived from it.
remote_base="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT"

# Paths — mirrors the folder layout under chestCT/ here:
# weights/llama_weights/, weights/Reg2RG_weights/, data/smoke_CTimagedata/reg2rg_data/dataset/...
lang_encoder_path="$remote_base/weights/llama_weights"
tokenizer_path="$remote_base/weights/llama_weights"
pretrained_visual_encoder="$remote_base/weights/Reg2RG_weights/RadFM_vit3d.pth"
pretrained_finegrained_visual_encoder="$remote_base/weights/fvlm_weights/finetuned_9_8/checkpoint_040.pth"
pretrained_adapter="$remote_base/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"
# Canonical fVLM processed-data root (processed_{train,valid}_{images,masks}) - required
# whenever pretrained_finegrained_visual_encoder is set; see radgenome_dataset_train.py.
fvlm_processed_root="$remote_base/data/dataset"

# data_folder="$remote_base/data/smoke_CTimagedata/reg2rg_data/dataset/train_preprocessed"
# mask_folder="$remote_base/data/smoke_CTimagedata/reg2rg_data/dataset/train_region_mask"
# report_file="$remote_base/data/smoke_CTimagedata/reg2rg_data/dataset/radgenome_files/train_region_report.csv"
# monai_cache_dir="$remote_base/data/smoke_CTimagedata/reg2rg_data/cache"
data_folder="$remote_base/data/dataset/train_preprocessed"
mask_folder="$remote_base/data/dataset/train_region_mask"
report_file="$remote_base/data/dataset/radgenome_files/train_region_report.csv"
monai_cache_dir="$remote_base/data/dataset/cache"

organ_annotation_path="$remote_base/data/dataset/EK_files_train/organ_annotation.json"
bank_npy_path="$remote_base/data/dataset/EK_files_train/organ_report_embeddings.npz"
output_dir="$remote_base/outputs/$experiment_name"
deepspeed_config="../ds_configs/stage2.json"

# Training settings — kept minimal for a fast smoke test, not a real training run
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=20
gradient_accumulation_steps=8 # 8的倍数
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=1
weight_decay=0.0
warmup_steps=0
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=4
logging_steps=1
