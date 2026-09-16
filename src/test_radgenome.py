import tqdm.auto as tqdm
import os
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.Reg2RG import Reg2RG
from Dataset.radgenome_dataset_test import RadGenomeDataset_Test
# from args.test_combined_region_radgenome.superpod import ModelArguments, DataArguments
from args.test_radgenome.jhcpu7 import ModelArguments, DataArguments
import torch
from torch.utils.data import DataLoader
from safetensors import safe_open
import random
import numpy as np
import pandas as pd

REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]
RESULT_COLUMNS = [
    "AccNum",
    "Question",
    "Dropped_regions",
    "GT_combined_report",
    "Mask_GT_combined_report",
    "Full_pred_combined_report",
    "Mask_pred_combined_report",
]

@dataclass
class DataCollator(object):

    def _collate_branch(self, instances, is_masked_branch):
        for instance in instances:
            if instance["is_masked_branch"] != is_masked_branch:
                raise ValueError("Mixed Full and Mask samples in one branch batch")

        lang_xs = [instance['lang_x'] for instance in instances]
        vision_xs = [instance['vision_x'] for instance in instances]
        mask_xs = [instance['mask_x'] for instance in instances]
        fvlm_mask_xs = [instance['fvlm_mask_x'] for instance in instances]
        images = torch.cat(
            [vision['image'].unsqueeze(0) for vision in vision_xs], dim=0
        )

        if is_masked_branch:
            collated_vision_xs = {'image': images}
            collated_mask_xs = {}
            collated_fvlm_mask_xs = {}
        else:
            vision_temp = {area: [] for area in REGIONS}
            mask_temp = {area: [] for area in REGIONS}
            fvlm_mask_temp = {area: [] for area in REGIONS}
            vision_shape = next(
                tensor.shape for area, tensor in vision_xs[0].items()
                if area != 'image'
            )
            mask_shape = next(iter(mask_xs[0].values())).shape
            fvlm_mask_shape = next(iter(fvlm_mask_xs[0].values())).shape
            useless_regions = []

            for area in REGIONS:
                area_is_present = False
                for sample_index in range(len(vision_xs)):
                    if area in vision_xs[sample_index]:
                        vision_temp[area].append(vision_xs[sample_index][area])
                        mask_temp[area].append(mask_xs[sample_index][area])
                        fvlm_mask_temp[area].append(fvlm_mask_xs[sample_index][area])
                        area_is_present = True
                    else:
                        vision_temp[area].append(torch.zeros(vision_shape))
                        mask_temp[area].append(torch.zeros(mask_shape))
                        fvlm_mask_temp[area].append(torch.zeros(fvlm_mask_shape))
                if not area_is_present:
                    useless_regions.append(area)

            for area in useless_regions:
                vision_temp.pop(area)
                mask_temp.pop(area)
                fvlm_mask_temp.pop(area)
            useful_regions = list(vision_temp.keys())
            collated_vision_xs = {
                area: torch.cat(
                    [tensor.unsqueeze(0) for tensor in vision_temp[area]], dim=0
                )
                for area in useful_regions
            }
            collated_vision_xs['image'] = images
            collated_mask_xs = {
                area: torch.cat(
                    [tensor.unsqueeze(0) for tensor in mask_temp[area]], dim=0
                )
                for area in useful_regions
            }
            collated_fvlm_mask_xs = {
                area: torch.cat([tensor.unsqueeze(0) for tensor in fvlm_mask_temp[area]], dim=0)
                for area in useful_regions
            }

        return {
            'sample_ids': [instance['sample_id'] for instance in instances],
            'lang_x': torch.cat(
                [tensor.unsqueeze(0) for tensor in lang_xs], dim=0
            ),
            'vision_x': collated_vision_xs,
            'mask_x': collated_mask_xs,
            'fvlm_mask_x': collated_fvlm_mask_xs,
            'region2area': [instance['region2area'] for instance in instances],
            'dropped_regions': [instance['dropped_regions'] for instance in instances],
            'question': [instance['question'] for instance in instances],
            'gt_combined_report': [
                instance['gt_combined_report'] for instance in instances
            ],
        }

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            'acc_num': [instance['acc_num'] for instance in instances],
            'full': self._collate_branch(
                [instance['full'] for instance in instances], False
            ),
            'mask': self._collate_branch(
                [instance['mask'] for instance in instances], True
            ),
        }

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
# 预处理数据

def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    (model_args, data_args) = parser.parse_args_into_dataclasses()
    print(model_args.ckpt_path)
    
    # 判断结果保存路径是否存在，不存在则创建
    os.makedirs(os.path.dirname(os.path.abspath(data_args.result_path)), exist_ok=True)
    os.makedirs(data_args.monai_cache_dir, exist_ok=True)
    if os.path.exists(data_args.result_path):
        df = pd.read_csv(data_args.result_path)
        if list(df.columns) != RESULT_COLUMNS:
            raise ValueError(
                f"Existing result file has the old single-output schema: "
                f"{data_args.result_path}. Use a new result_path or migrate/remove "
                "the old file before running dual-output inference."
            )
        inferenced_id = df["AccNum"].tolist()
    else:
        df = pd.DataFrame(columns=RESULT_COLUMNS)
        df.to_csv(data_args.result_path, index=False)
        inferenced_id = []

    completed_count = len(df)
    if data_args.max_samples is not None and completed_count >= data_args.max_samples:
        print(f"Already generated {completed_count} reports; nothing to do.")
        return

    print("Setup Data")
    Test_dataset = RadGenomeDataset_Test(
        text_tokenizer=model_args.tokenizer_path,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        cache_dir=data_args.monai_cache_dir,
        inferenced_id=inferenced_id,
        #---Dan---
        use_fvlm=model_args.pretrained_finegrained_visual_encoder is not None,
        # Canonical fVLM processed data; do not derive it from Reg2RG's raw-data root.
        fvlm_processed_root=data_args.fvlm_processed_root,
        #---Dan---
    )

    Test_dataloader = DataLoader(
        Test_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=DataCollator(),
        drop_last=False,
    )

    print("Setup Model")

    model = Reg2RG(
        text_tokenizer_path=model_args.tokenizer_path,
        lang_model_path=model_args.lang_encoder_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        #---Dan---
        pretrained_finegrain_visual_encoder=model_args.pretrained_finegrained_visual_encoder,
        #---Dan---
        pretrained_adapter=model_args.pretrained_adapter,
        bank_npy_path=data_args.bank_npy_path,
        organ_annotation_path=data_args.organ_annotation_path,
    )

    ckpt = torch.load(model_args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    print("load ckpt")
    model = model.cuda()
    print("model to cuda")

    model.eval()

    for sample in tqdm.tqdm(Test_dataloader):
        acc_num = sample["acc_num"][0]
        full = sample['full']
        mask = sample['mask']

        full_reports, full_region_embedding = model.generate(
            full['lang_x'].cuda(),
            {area: tensor.cuda() for area, tensor in full['vision_x'].items()},
            {area: tensor.cuda() for area, tensor in full['mask_x'].items()},
            {area: tensor.cuda() for area, tensor in full['fvlm_mask_x'].items()},
            full['region2area'],
            sample_ids=full['sample_ids'],
            return_rwlke_region_embedding=True,
        )
        masked_region_embedding = full_region_embedding.clone()
        for batch_index, dropped_regions in enumerate(mask['dropped_regions']):
            if full['region2area'][batch_index] != mask['region2area'][batch_index]:
                raise ValueError("Full and Mask prompts must use identical region order")
            for region_index, area in mask['region2area'][batch_index].items():
                if area not in dropped_regions:
                    continue
                region_token_len = model.embedding_layer.region_token_len
                start = region_index * region_token_len
                end = start + region_token_len
                if end > masked_region_embedding.size(1):
                    raise IndexError(
                        f"Region slot {region_index} exceeds the Full RWLKE embedding"
                    )
                masked_region_embedding[batch_index, start:end, :] = 0.0

        mask_reports = model.generate(
            mask['lang_x'].cuda(),
            {area: tensor.cuda() for area, tensor in mask['vision_x'].items()},
            {},
            {},
            mask['region2area'],
            sample_ids=mask['sample_ids'],
            precomputed_region_embedding=masked_region_embedding,
        )

        print('AccNum: ', acc_num)
        print('Dropped regions: ', mask['dropped_regions'][0])
        print('Full GT report: ', full['gt_combined_report'][0])
        print('Full prediction: ', full_reports[0])
        print('Mask GT report: ', mask['gt_combined_report'][0])
        print('Mask prediction: ', mask_reports[0])
        
        new_data = pd.DataFrame([[
            acc_num,
            full['question'][0],
            ','.join(mask['dropped_regions'][0]),
            full['gt_combined_report'][0],
            mask['gt_combined_report'][0],
            full_reports[0],
            mask_reports[0],
        ]], columns=RESULT_COLUMNS)
        new_data.to_csv(data_args.result_path, mode='a', header=False, index=False)
        completed_count += 1
        if data_args.max_samples is not None and completed_count >= data_args.max_samples:
            print(f"Reached max_samples={data_args.max_samples}")
            break


if __name__ == "__main__":
    main()
