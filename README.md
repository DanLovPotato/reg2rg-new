# Reg2RG 🚀
*Official repository for the IEEE TMI paper **“Large Language Model with Region‑Guided Referring and Grounding for CT Report Generation”***

> 这份fork加了"用`radfm_perceiver_align`项目单独训好的organ-branch perceiver/fc替换掉原版"这个能力，下面"🧩 Organ-Branch Perceiver 替换"这一节是这份fork特有的内容，其余部分是原作者的通用说明，照常适用。

## 📦 Installation
```bash
git clone https://github.com/zhi-xuan-chen/Reg2RG.git
cd Reg2RG
conda create -n reg2rg python=3.9
conda activate reg2rg
pip install -r requirements.txt
```

## 📁 Data Preparation
The **RadGenome‑ChestCT** dataset used in this work can be downloaded from the Hugging Face hub:
<https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT>

## 🗄️ Model Checkpoints
- **Base LLM:** [Llama‑2‑7b‑chat‑hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- **Pre‑trained & auxiliary checkpoints:** [Reg2RG](https://huggingface.co/Trusure/Reg2RG/tree/main)

## 🧩 Organ-Branch Perceiver 替换

需要的东西：这份Reg2RG仓库 + `radfm_perceiver_align`仓库（同级目录，负责训练organ perceiver/fc和对应的4096维knowledge bank）。

### 怎么跑（以`perceiver_step145000`这套配置为例）

1. **训练**（把organ perceiver冻住、fc继续联合训练，10个epoch）：
   ```bash
   cd scripts
   bash train_radgenome.sh perceiver_step145000
   ```
   对应`configs/train_radgenome/perceiver_step145000.sh`，里面`freeze_organ_perceiver=True`、`freeze_organ_fc=False`——perceiver权重保持不动（照抄`radfm_perceiver_align`训出来的），fc在Reg2RG这边继续跟着联合训练微调。

2. **验证/测试**（用训练中间或最终存下来的checkpoint，在validation集上生成报告）：
   先复制一份`configs/test_radgenome/perceiver_step145000.sh`，把里面的`ckpt_path`改成你想测的那个`outputs/Reg2RG_organperceiver_step145000/checkpoint-XXXX/pytorch_model.bin`，然后：
   ```bash
   cd scripts
   bash test_radgenome.sh [你改好的配置文件名]
   ```
   生成结果存到配置里`result_path`指定的csv，再按下面"📊 Evaluation"那节的步骤算NLG/CE指标。

3. **新训一个自己的organ perceiver/fc**：参考`radfm_perceiver_align`仓库自己的README，训完之后，把这份Reg2RG配置里的`organ_perceiver_path`/`organ_fc_path`/`bank_npy_path`/`organ_annotation_path`都换成新checkpoint对应的路径（四个必须是同一个checkpoint step配套生成的，不能混用）。

## 🤖 Inference
After the data and checkpoints are ready, you can inference the model on **RadGenome‑ChestCT** with our pre‑trained weights.

1. Create a custom configuration file `[your_config_name].sh` in `configs/test_radgenome/` (you can copy and modify `configs/test_radgenome/jhcpu7.sh`).
2. Run:
   ```bash
   cd scripts
   bash test_radgenome.sh [your_config_name]
   ```

## 📊 Evaluation
After inference, evaluate the generated reports with the scripts in the `evaluation/` directory.
> **All scripts read file paths from variables *inside* the script.
> Please open each file and set the path (e.g. `results_path`) to your own CSV results before running.**

### Whole‑Report Evaluation

1. **Strip region prefixes**
   ```bash
   python evaluation/rm_region_text.py
   ```
2. **Compute Natural Language Generation (NLG) metrics**
   ```bash
   python evaluation/hf_nlg_evaluation.py
   ```
3. **Compute Clinical Efficacy (CE) metrics**
   ```bash
   python evaluation/ce_evaluator_ct2rep/ce_evaluation.py
   ```

### Region‑Level Evaluation

1. **Region prediction accuracy**
   ```bash
   python evaluation/region_pred_acc.py
   ```
2. **Split into region‑specific reports**
   ```bash
   python evaluation/parse_report_to_region.py
   ```
3. **Compute region‑level NLG metrics**
   ```bash
   python evaluation/hf_nlg_evaluation_region.py
   ```
4. **Compute region‑level CE metrics**
   ```bash
   python evaluation/ce_evaluator_ct2rep/ce_evaluation_region.py
   ```

## 🏋️‍♂️ Training
You can fine‑tune the model on your own dataset using our pre‑trained checkpoint.

1. Add your dataset class to `src/Dataset/`, following the style of the existing datasets.
2. Create a config `[your_config_name].sh` in `configs/train_radgenome/` (copy and adjust `configs/train_radgenome/jhcpu7.sh`).
3. Run the training script:
   ```bash
   cd scripts
   bash train_radgenome.sh [your_config_name]
   ```

## 📄 Citation
If you find our work useful, please consider citing it:

```bibtex
@article{chen2025reg2rg,
  title   = {Large Language Model with Region-Guided Referring and Grounding for CT Report Generation},
  author  = {Chen, Zhixuan and Bie, Yequan and Jin, Haibo and Chen, Hao},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025},
  publisher = {IEEE}
}
```

## 🙏 Acknowledgements
This project builds on the [RadFM](https://github.com/chaoyi-wu/RadFM) repository and the [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model.
We sincerely thank the original authors for their invaluable contributions to the community.
