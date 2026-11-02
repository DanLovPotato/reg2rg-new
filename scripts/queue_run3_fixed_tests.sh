# Sequentially test run3_fixed epochs 5, 6 (full), 7 on GPU 3, each as soon as that
# epoch's checkpoint is fully written. Mirrors scripts/queue_run2_tests.sh.
REPO=/home/wisc/dxiang23/Projects/chestCT/code/Reg2RG-debug-current-full_correct_F_20260920_1027
O=/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/Reg2RG_organperceiver_step145000_run3_fixed
C=$REPO/configs/test_radgenome

for epoch in 5 6 7; do
    while true; do
        c=$(ls -d $O/checkpoint-* 2>/dev/null | sort -t- -k2 -n | sed -n "${epoch}p")
        if [ -n "$c" ] && [ -f "$c/pytorch_model.bin" ] && [ -f "$c/latest" ] && [ -f "$c/trainer_state.json" ]; then break; fi
        screen -ls | grep -q reg2rg_train3 || { [ -z "$c" ] && { echo "$(date) training ended before epoch $epoch"; exit 0; }; }
        sleep 300
    done
    sleep 120
    sed -e "s|checkpoint-4825|$(basename $c)|" -e "s|run3_fixed_epoch5|run3_fixed_epoch$epoch|" $C/perceiver_step145000_run3_fixed_epoch5.sh > $C/perceiver_step145000_run3_fixed_epoch$epoch.sh
    [ "$epoch" = 6 ] && sed -i "/^max_samples=/d" $C/perceiver_step145000_run3_fixed_epoch$epoch.sh  # epoch 6: all 1564 cases
    # wait for GPU3 to be free of any other reg2rg test job before starting
    while screen -ls | grep -qE "reg2rg_test_r2q|reg2rg_test_r3q"; do sleep 120; done
    echo "$(date) testing epoch $epoch: $c"
    cd $REPO/scripts
    screen -dmS reg2rg_test_r3q bash -c "source /opt/miniforge3/etc/profile.d/conda.sh; conda activate reg2rg; bash test_radgenome.sh perceiver_step145000_run3_fixed_epoch$epoch 2>&1 | tee ../test_run3_fixed_epoch$epoch.log"
    sleep 30
    while screen -ls | grep -q reg2rg_test_r3q; do sleep 120; done
    echo "$(date) epoch $epoch test finished (Tracebacks: $(grep -c Traceback ../test_run3_fixed_epoch$epoch.log))"
done
echo "$(date) queue done"
