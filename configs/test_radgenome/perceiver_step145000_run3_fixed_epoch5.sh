# Validation generation for a model trained with configs/train_radgenome/perceiver_step145000_run3_fixed.sh.
# The organ perceiver/fc paths are only needed to build the modules; their weights
# are then overwritten by ckpt_path (strict load).
# Usage (from scripts/): bash test_radgenome.sh perceiver_step145000_run3_fixed_epoch5

# Device settings — EDIT to match the server (run `nvidia-smi` there first)
cuda_devices="3"

remote_base="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT"

lang_encoder_path="$remote_base/weights/llama_weights"
tokenizer_path="$remote_base/weights/llama_weights"
pretrained_visual_encoder="$remote_base/weights/Reg2RG_weights/RadFM_vit3d.pth"
pretrained_adapter="$remote_base/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"

align_ckpt_dir="$remote_base/outputs/radfm_perceiver_align/run_20261021"
organ_perceiver_path="$align_ckpt_dir/perceiver-step145000.pth"
organ_fc_path="$align_ckpt_dir/fc-step145000.pth"

# EDIT: the checkpoint written by the training run (filled in by scripts/queue_run3_fixed_tests.sh)
ckpt_path="$remote_base/outputs/Reg2RG_organperceiver_step145000_run3_fixed/checkpoint-4825/pytorch_model.bin"  # run3_fixed epoch 5

data_folder="$remote_base/data/dataset/valid_preprocessed"
mask_folder="$remote_base/data/dataset/valid_region_mask"
report_file="$remote_base/data/dataset/radgenome_files/validation_region_report.csv"
monai_cache_dir="$remote_base/data/dataset/cache"

# Same bank npz as training; validation's own annotation (neighbours are train patients)
bank_dir="$remote_base/data/dataset/KE_bank_4096_step145000"
bank_npy_path="$bank_dir/organ_report_embeddings.npz"
organ_annotation_path="$bank_dir/val/organ_annotation.json"
result_path="$remote_base/outputs/Reg2RG_organperceiver_step145000_run3_fixed/results/valid_reports_run3_fixed_epoch5.csv"
max_samples="400"
