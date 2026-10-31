# Reg2RG training with the organ-branch perceiver/fc fine-tuned in radfm_perceiver_align
# (step145000) and the matching 4096-d knowledge bank (radfm_perceiver_align/KE_generation).
#   organ branch : ViT (frozen) -> organ_perceiver (step145000, frozen) -> organ_fc (step145000, trained)
#   full-CT branch: unchanged (RadFM perceiver + randomly initialised fc, both trained)
# Training hyper-parameters follow repro/chtc_training/run_full.sh.
# Usage (from scripts/): bash train_radgenome.sh perceiver_step145000

# Experiment settings
experiment_name="Reg2RG_organperceiver_step145000_run2"
bf16=True

# Device settings — EDIT to match the server (run `nvidia-smi` there first)
cuda_devices="0,1,2"

# Torchrun settings
master_port=25368

remote_base="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT"

lang_encoder_path="$remote_base/weights/llama_weights"
tokenizer_path="$remote_base/weights/llama_weights"
pretrained_visual_encoder="$remote_base/weights/Reg2RG_weights/RadFM_vit3d.pth"
pretrained_finegrained_visual_encoder="$remote_base/weights/fvlm_weights/finetuned/checkpoint_040.pth"  # unused
pretrained_adapter="$remote_base/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"  # full-CT branch perceiver

# Organ-branch perceiver/fc — must be the same step the bank was built with
align_ckpt_dir="$remote_base/outputs/radfm_perceiver_align/run_20261021"
organ_perceiver_path="$align_ckpt_dir/perceiver-step145000.pth"
organ_fc_path="$align_ckpt_dir/fc-step145000.pth"
freeze_organ_perceiver=True
freeze_organ_fc=False

data_folder="$remote_base/data/dataset/train_preprocessed"
mask_folder="$remote_base/data/dataset/train_region_mask"
report_file="$remote_base/data/dataset/radgenome_files/train_region_report.csv"
monai_cache_dir="$remote_base/data/dataset/cache"

bank_dir="$remote_base/data/dataset/KE_bank_4096_step145000"
bank_npy_path="$bank_dir/organ_report_embeddings.npz"
organ_annotation_path="$bank_dir/train/organ_annotation.json"
output_dir="$remote_base/outputs/$experiment_name"
deepspeed_config="../ds_configs/stage2.json"

# Training settings
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=5  # 3 GPUs x 1 x 5 = 15, close to run_full.sh's 2 x 1 x 8 = 16
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=20  # keep every epoch (optimizer states of older epochs are stripped by scripts/strip_old_ckpt_states.sh)
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
logging_steps=1
