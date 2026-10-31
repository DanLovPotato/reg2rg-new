#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
S=/staging/t/txu98
OUT="${VALIDATION_OUTPUT_ROOT:-$S/reg2rg_validation_15070_10organs_20260930}"
mkdir -p "$OUT" "$ROOT/data/dataset" "$ROOT/python_packages" "$ROOT/cache"
exec > >(tee -a "$OUT/inference.log") 2>&1
trap 'status=$?; echo "$status" > "$OUT/last_exit_code.txt"' EXIT
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/python_packages" HF_HOME="$ROOT/cache/hf" TRANSFORMERS_OFFLINE=1
tar -xzf code.tar.gz
tar -xzf "$S/python_packages/reg2rg_torch201_ds0126_v1.tar.gz" -C "$ROOT/python_packages"
run() { apptainer exec --nv --bind "$ROOT:$ROOT" --bind "$S:$S" "$S/reg2rg-pytorch2.0.1-cu117.sif" "$@"; }
run python -m pip install --no-deps --no-index --target "$ROOT/python_packages" "$S/python_wheels/torch_geometric-2.6.1-py3-none-any.whl"
run python -c 'import torch, transformers, peft, torch_geometric; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
for archive in "$S"/radgenome_valid_ct/radgenome_valid_ct_batch_*.tar; do
  tar -xf "$archive" -C "$ROOT/data"
done
tar -xzf "$S/radgenome_valid_companion/valid_region_mask.tar.gz" -C "$ROOT/data/dataset"
tar -xf "$S/radgenome_valid_companion/radgenome_valid_reports.tar" -C "$ROOT/data/dataset"
tar -xf "$S/reg2rg-models.tar"
tar -xf "$S/reg2rg_train_full_2gpu_20260921/6246686.0/checkpoint-15070.tar" checkpoint-15070/pytorch_model.bin
cp "$ROOT/code/code_revision.json" "$OUT/code_revision.json"
export VALIDATION_ROOT="$ROOT" VALIDATION_OUT="$OUT"
cd "$ROOT/code"
run python "$ROOT/${VALIDATION_ENTRY:-entry.py}" \
 --lang_encoder_path "$ROOT/models/Llama-2-7b-chat-hf" \
 --tokenizer_path "$ROOT/models/Llama-2-7b-chat-hf" \
 --pretrained_visual_encoder "$ROOT/models/Reg2RG/RadFM_vit3d.pth" \
 --pretrained_adapter "$ROOT/models/Reg2RG/RadFM_perceiver_fc.pth" \
 --ckpt_path "$ROOT/checkpoint-15070/pytorch_model.bin" \
 --data_folder "$ROOT/data/dataset/valid_preprocessed" \
 --mask_folder "$ROOT/data/dataset/valid_region_mask" \
 --report_file "$ROOT/data/dataset/radgenome_files/validation_region_report.csv" \
 --monai_cache_dir "$ROOT/cache/monai" \
 --bank_npy_path "$S/report_bank/train_10organs_20260915/organ_report_embeddings.npz" \
 --organ_annotation_path "$S/report_bank/validation_10organs_20260930/organ_annotation.json" \
 --result_path "$OUT/radgenome_validation_reports.csv"
date -Is > "$OUT/INFERENCE_COMPLETE"
