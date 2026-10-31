# Reg2RG

这份fork加了"用`radfm_perceiver_align`项目单独训好的organ-branch perceiver/fc替换掉原版"的能力。

## 环境

```bash
conda create -n reg2rg python=3.9
conda activate reg2rg
pip install -r requirements.txt
```

## 怎么跑

需要的东西：这份Reg2RG仓库 + `radfm_perceiver_align`仓库（同级目录，负责训练organ perceiver/fc和对应的4096维knowledge bank）。

以`perceiver_step145000`这套配置为例：

1. **训练**（organ perceiver冻住，fc继续联合训练，10个epoch）：
   ```bash
   cd scripts
   bash train_radgenome.sh perceiver_step145000
   ```
   对应`configs/train_radgenome/perceiver_step145000.sh`，里面`freeze_organ_perceiver=True`、`freeze_organ_fc=False`。

2. **验证/测试**（用训练中间或最终存下来的checkpoint，在validation集上生成报告）：
   复制一份`configs/test_radgenome/perceiver_step145000.sh`，把`ckpt_path`改成想测的`outputs/Reg2RG_organperceiver_step145000/checkpoint-XXXX/pytorch_model.bin`，然后：
   ```bash
   cd scripts
   bash test_radgenome.sh [改好的配置文件名]
   ```
   结果存到配置里`result_path`指定的csv，再跑下面"算指标"的步骤。

3. **换一套新的organ perceiver/fc**：训好之后，把Reg2RG配置里的`organ_perceiver_path`/`organ_fc_path`/`bank_npy_path`/`organ_annotation_path`都换成新checkpoint对应的路径（四个必须是同一个checkpoint step配套生成的，不能混用）。

## 算指标

生成完报告之后（上面第2步的`result_path` csv），用`evaluation/`里的脚本算分：
> 每个脚本的路径（比如`results_path`）都是写在脚本开头的变量里，跑之前先改成自己的csv路径。

1. **区域预测准确率**
   ```bash
   python evaluation/region_pred_acc.py
   ```
2. **拆成按区域分开的报告**
   ```bash
   python evaluation/parse_report_to_region.py
   ```
3. **算区域级NLG指标**（BLEU/METEOR/ROUGE/CIDEr）
   ```bash
   python evaluation/hf_nlg_evaluation_region.py
   ```
4. **算区域级临床效能（CE）指标**
   ```bash
   python evaluation/ce_evaluator_ct2rep/ce_evaluation_region.py
   ```
