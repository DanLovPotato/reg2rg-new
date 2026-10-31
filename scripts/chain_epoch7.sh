set -u
R=/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs
REPO=/home/wisc/dxiang23/Projects/chestCT/code/Reg2RG-debug-current-full_correct_F_20260920_1027
mkdir -p $R/Reg2RG_organperceiver_step145000_epoch7
cp -p $R/Reg2RG_organperceiver_step145000/checkpoint-11258/pytorch_model.bin $R/Reg2RG_organperceiver_step145000_epoch7/ \
  && cmp $R/Reg2RG_organperceiver_step145000/checkpoint-11258/pytorch_model.bin $R/Reg2RG_organperceiver_step145000_epoch7/pytorch_model.bin \
  && echo "$(date) epoch7 weights copied and verified" || { echo "$(date) COPY FAILED"; exit 1; }
while screen -ls | grep -q reg2rg_test_ep6; do sleep 120; done
echo "$(date) epoch6 test finished; Tracebacks: $(grep -c Traceback $REPO/test_epoch6.log)"
cd $REPO/scripts
screen -dmS reg2rg_test_ep7 bash -c 'source /opt/miniforge3/etc/profile.d/conda.sh; conda activate reg2rg; bash test_radgenome.sh perceiver_step145000_epoch7 2>&1 | tee ../test_epoch7.log'
echo "$(date) epoch7 test launched in screen reg2rg_test_ep7"
