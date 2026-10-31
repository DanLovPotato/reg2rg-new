# Wait for the epoch-3 test to finish and for run2's epoch-5 checkpoint to be fully written,
# then test epoch 5 on all validation cases.
REPO=/home/wisc/dxiang23/Projects/chestCT/code/Reg2RG-debug-current-full_correct_F_20260920_1027
O=/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/Reg2RG_organperceiver_step145000_run2
while screen -ls | grep -q reg2rg_test_r2e3; do sleep 120; done
echo "$(date) epoch3 test finished"
while true; do
    c=$(ls -d $O/checkpoint-80[0-9][0-9] 2>/dev/null | head -1)
    if [ -n "$c" ] && [ -f "$c/pytorch_model.bin" ] && [ -f "$c/latest" ] && [ -f "$c/trainer_state.json" ]; then break; fi
    sleep 120
done
sleep 120  # let the save settle
echo "$(date) epoch5 checkpoint ready: $c ($(stat -c %s $c/pytorch_model.bin) bytes)"
sed -i "s|checkpoint-EPOCH5|$(basename $c)|" $REPO/configs/test_radgenome/perceiver_step145000_run2_epoch5.sh
cd $REPO/scripts
screen -dmS reg2rg_test_r2e5 bash -c 'source /opt/miniforge3/etc/profile.d/conda.sh; conda activate reg2rg; bash test_radgenome.sh perceiver_step145000_run2_epoch5 2>&1 | tee ../test_run2_epoch5.log'
echo "$(date) epoch5 test launched"
