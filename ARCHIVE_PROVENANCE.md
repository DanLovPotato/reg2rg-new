# Full-report CE F1 0.2441: historical training-code archive

Branch: `full_correct_F_20260920` (name requested by the user).
Archived on 2026-10-02. The branch name is not the training or evaluation date.

## Which model is this?

- CHTC training job: **6246686.0**.
- Actual final checkpoint: **checkpoint-15070**, not checkpoint-15072.
- Checkpoint archive: `/staging/t/txu98/reg2rg_train_full_2gpu_20260921/6246686.0/checkpoint-15070.tar`.
- Base branch: `feature/direct-gcn-fusion-20260916`.
- Base commit: `2f75ae67073343ec954b4de9c9242d8c2ecbe75d`.
- The actual training payload also contains the non-reentrant Llama gradient-checkpointing compatibility patch, recorded in `code_revision.json`.
- The source files in the repository root match the actual submitted `code/` payload byte-for-byte. This is **not** the later local `reg2rg-new-main` FineGrainedCLIP revision, and is not the official retraining job 6527314.

Training used **Full + Mask**, with `full_loss + 0.8 * mask_loss`, shared visible-region RWLKE tokens, and direct LIFT/GCN fusion. **Full-only refers to validation generation, not to the training loss.**

## Training configuration

The authoritative historical launcher is `repro/chtc_training/run_full.sh`, not an older root debug configuration.

- 24,126 report-eligible training samples; 10 epochs.
- 2 GPUs, batch size 1 per GPU, gradient accumulation 8; effective batch 16.
- LR 5e-5, weight decay 0, warmup 20, constant-with-warmup scheduler, seed/data_seed 42.
- BF16, DeepSpeed ZeRO-2, initialization from Llama-2/RadFM only.
- Ten-organ training bank: `/staging/t/txu98/report_bank/train_10organs_20260915/`.
- Training NPZ SHA-256: `8b229026ae3bbd7d8b9ac61c010e57ce845e00d5e98a990bd68853accc6280b8`.
- Training JSON SHA-256: `989d339f2d748c962d765ad1d794405b56cdaed78c6be71c0d53f290baa68625`.

The launcher performs a separate smoke gate before fresh full training. It only resumes a checkpoint belonging to the same submitted run. It does not use Box-9042 or another previously trained Reg2RG model.

## Validation and recorded metrics

- Full-only validation job: **6421418.0**.
- Historical launcher and entry point: `repro/chtc_validation_full/run.sh` and `full_entry.py`.
- This job strictly loaded checkpoint-15070 and generated **1,564 Full reports** in natural case order, skipping Mask generation.
- The validation JSON was `/staging/t/txu98/report_bank/validation_10organs_20260930/organ_annotation.json`, paired with the training NPZ above.
- Validation JSON SHA-256: `fe4441b4b686e3bd586f91a7cedbea7a5320eaa47709029ee4681970a6a90013`.
- Full report CSV SHA-256: `9170de5c29eb9b9a94d0c118e5b6b1f30e61124a3a7ab07ee81c10de6c08fc9f`.

Whole-report RadBERT CE, threshold 0.5, max length 512, region prefixes removed:

| Aggregation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Micro | 0.4737442922 | 0.1644215531 | **0.2441176471** |
| Macro | 0.3812227386 | 0.1348110173 | 0.1926210665 |
| Weighted | 0.4499781487 | 0.1644215531 | 0.2341327959 |

The CE classifier checksum is `3ce74736ea6900313c8ecb2e3d287c13186420ccc9c0b14e6978ffa2b38bc249`. **That is the RadBERT evaluator checksum, not the Reg2RG model checksum.**

Aggregate CE, per-finding CE and NLG tables are in `repro/metrics/`. These are the measurements of the historical generated CSV, not a guarantee that rerunning inference is bitwise deterministic. The historical Full-only wrapper does not explicitly set an inference seed.

## Payload integrity and use

Both original training and Full-only inference payloads have SHA-256:

`35c5cf2f08e33170abde5905d13962bd1c3f59f4860c111c2bb76eaa81727ce2`

They are retained under their respective `repro/chtc_*/code.tar.gz` paths. Finder AppleDouble entries (`._*`) are metadata and are retained only inside the original tarballs, not as source files. Scripts retain historical CHTC paths and are archived without modification. External data, companion archives, bank files, container, base models and trained weights remain required; **none of those large assets or report texts are uploaded by this archive**.

Run `python repro/verify_source.py` to verify that the root source matches the historical payload. No existing GitHub branch is overwritten by this archive.
