#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
STAGING=/staging/t/txu98
OUT="$STAGING/reg2rg_train_full_2gpu_20260921/${1:?job_id}"
SIF="$STAGING/reg2rg-pytorch2.0.1-cu117.sif"
mkdir -p "$OUT" "$ROOT/bank" "$ROOT/python_packages" "$ROOT/cache"
exec > >(tee -a "$OUT/training.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$OUT/last_exit_code.txt"' EXIT
echo "START $(date -Is) job=$1 host=$(hostname)"
nvidia-smi
tar -xzf code.tar.gz
cp "$ROOT/code/code_revision.json" "$OUT/code_revision.json"
cat "$OUT/code_revision.json"
cp "$STAGING/report_bank/train_10organs_20260915/organ_report_embeddings.npz" "$ROOT/bank/"
cp "$STAGING/report_bank/train_10organs_20260915/organ_annotation.json" "$ROOT/bank/"
export PYTHONPATH="$ROOT/python_packages"
export PATH="$ROOT/python_packages/bin:$PATH"
printf '%s  %s\n' b7df083f9d80a8ed04d6261de893171f496805a044f0ac29b1a59f143de15b4c "$STAGING/python_packages/reg2rg_torch201_ds0126_v1.tar.gz" | sha256sum -c -
tar -xzf "$STAGING/python_packages/reg2rg_torch201_ds0126_v1.tar.gz" -C "$ROOT/python_packages"
export HF_HOME="$ROOT/cache/hf" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1
export TRAIN_ROOT="$ROOT" TRAIN_OUTPUT="$OUT"
run() { apptainer exec --nv --bind "$ROOT:$ROOT" --bind "$STAGING:$STAGING" "$SIF" "$@"; }
run python -m pip install --no-deps --no-index --target "$ROOT/python_packages" "$STAGING/python_wheels/torch_geometric-2.6.1-py3-none-any.whl"
run python -c 'import torch,transformers,deepspeed,torch_geometric; assert torch.cuda.device_count()==2; print("VERSIONS",torch.__version__,transformers.__version__,deepspeed.__version__); print("GPUS",[torch.cuda.get_device_name(i) for i in range(2)])'
tar -xf "$STAGING/reg2rg-models.tar"
run python prepare_smoke.py "$ROOT"
cp "$ROOT/preflight.json" "$OUT/smoke_preflight.json"
cd "$ROOT/code"
common=(--bf16 True
 --lang_encoder_path "$ROOT/models/Llama-2-7b-chat-hf"
 --tokenizer_path "$ROOT/models/Llama-2-7b-chat-hf"
 --pretrained_visual_encoder "$ROOT/models/Reg2RG/RadFM_vit3d.pth"
 --pretrained_adapter "$ROOT/models/Reg2RG/RadFM_perceiver_fc.pth"
 --data_folder "$ROOT/dataset/train_preprocessed"
 --mask_folder "$ROOT/dataset/train_region_mask"
 --report_file "$ROOT/dataset/radgenome_files/train_region_report.csv"
 --monai_cache_dir "$ROOT/cache/monai"
 --bank_npy_path "$ROOT/bank/organ_report_embeddings.npz"
 --organ_annotation_path "$ROOT/bank/organ_annotation.json"
 --deepspeed ds_configs/stage2.json --per_device_train_batch_size 1
 --learning_rate 5e-5 --weight_decay 0.0 --evaluation_strategy no
 --dataloader_num_workers 4 --logging_steps 1 --logging_nan_inf_filter False
 --report_to none --seed 42 --data_seed 42)
# Fresh initialization for this gate. Its weights are never used by full training.
run torchrun --standalone --nnodes=1 --nproc_per_node=2 src/train_radgenome.py \
 "${common[@]}" --output_dir "$ROOT/smoke_checkpoints" \
 --gradient_accumulation_steps 8 --max_steps 3 --num_train_epochs 1 \
 --warmup_steps 0 --lr_scheduler_type constant \
 --save_strategy steps --save_steps 3 --save_total_limit 1 --run_name direct_gcn_2gpu_gate
run python "$ROOT/verify_smoke.py" "$ROOT/smoke_checkpoints" "$OUT" 2
echo "TWO_GPU_GATE_PASSED $(date -Is)"
run python "$ROOT/prepare_full.py" "$ROOT"
cp "$ROOT/full_manifest.json" "$OUT/"
export RESUME_CURRENT_RUN=""
if [ -f "$OUT/latest_checkpoint.json" ]; then
    checkpoint_name=$(run python -c 'import json,os; from pathlib import Path; p=Path(os.environ["TRAIN_OUTPUT"]); print(json.loads((p/"latest_checkpoint.json").read_text())["archive"])')
    mkdir -p "$ROOT/checkpoints"
    tar -xf "$OUT/$checkpoint_name" -C "$ROOT/checkpoints"
    export RESUME_CURRENT_RUN="$ROOT/checkpoints/${checkpoint_name%.tar}"
fi
printf '{"status":"RUNNING"}\n' > "$OUT/attempt_status.json"
set +e
run torchrun --standalone --nnodes=1 --nproc_per_node=2 "$ROOT/train_entry.py" \
 "${common[@]}" --output_dir "$ROOT/checkpoints" \
 --gradient_accumulation_steps 8 --num_train_epochs 10 --max_steps -1 \
 --warmup_steps 20 --lr_scheduler_type constant_with_warmup \
 --save_strategy steps --save_steps 250 --save_total_limit 2 --run_name direct_gcn_10organ_2gpu_full
status=$?
set -e
if [ "$status" -ne 0 ] && run python -c 'import json,os; from pathlib import Path; assert json.loads((Path(os.environ["TRAIN_OUTPUT"])/"attempt_status.json").read_text())["status"]=="CONTINUE"'; then
    exit 85
fi
exit "$status"
