#!/bin/bash
# Keeps every epoch's pytorch_model.bin but deletes the DeepSpeed optimizer/model-state
# folder (global_step*, ~36 GB) of every checkpoint OLDER than the newest complete one,
# so a long run can still be resumed from its last checkpoint.
# A checkpoint counts as complete when it has pytorch_model.bin, trainer_state.json,
# the "latest" marker and its global_step* folder. Checkpoints newer than that (i.e.
# still being written) are never touched.
# Usage: bash strip_old_ckpt_states.sh <output_dir> <training_screen_name>
out="$1"; screen_name="$2"
while true; do
    newest_step=0
    for c in $(ls -d "$out"/checkpoint-* 2>/dev/null); do
        step=${c##*-}
        if [ -f "$c/pytorch_model.bin" ] && [ -f "$c/trainer_state.json" ] && [ -f "$c/latest" ] \
           && ls -d "$c"/global_step* >/dev/null 2>&1 && [ "$step" -gt "$newest_step" ]; then
            newest_step=$step
        fi
    done
    for c in $(ls -d "$out"/checkpoint-* 2>/dev/null); do
        step=${c##*-}
        if [ "$newest_step" -gt 0 ] && [ "$step" -lt "$newest_step" ] && [ -f "$c/pytorch_model.bin" ] \
           && ls -d "$c"/global_step* >/dev/null 2>&1; then
            rm -rf "$c"/global_step* && echo "$(date) stripped optimizer states from $c (newest complete: $newest_step)"
        fi
    done
    screen -ls | grep -q "$screen_name" || break
    sleep 600
done
echo "$(date) training screen gone; stopping"
