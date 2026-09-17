import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer
from dataclasses import dataclass, field
from Model.Reg2RG import Reg2RG
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
from args.train_radgenome.jhcpu7 import ModelArguments, DataArguments, TrainingArguments
import numpy as np
import torch              
import random
 
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
REGION_TOKEN_LEN = 33
 
@dataclass
class DataCollator(object):

    def _collate_branch(
        self, instances: Sequence[Dict], is_masked_branch: bool
    ) -> Dict[str, torch.Tensor]:
        for instance in instances:
            if instance["is_masked_branch"] != is_masked_branch:
                raise ValueError("Mixed Full and Mask samples in one branch batch")

        lang_xs, vision_xs, mask_xs, fvlm_mask_xs, region2areas, attention_masks, labels = tuple(
            [instance[key] for instance in instances]
            for key in (
                'lang_x', 'vision_x', 'mask_x', 'fvlm_mask_x', 'region2area',
                'attention_mask', 'label'
            )
        )
        sample_ids = [instance["sample_id"] for instance in instances]
        dropped_regions = [instance["dropped_regions"] for instance in instances]

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim = 0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim = 0)

        images = torch.cat([vision['image'].unsqueeze(0) for vision in vision_xs], dim = 0)

        if is_masked_branch:
            # Mask 分支只有 masked global CT；region embedding 由 Trainer 稍后注入。
            collated_vision_xs = {'image': images}
            collated_mask_xs = {}
            collated_fvlm_mask_xs = {}
        else:
            vision_temp = {area: [] for area in REGIONS}
            mask_temp = {area: [] for area in REGIONS}
            fvlm_mask_temp = {area: [] for area in REGIONS}
            vision_shape = next(
                tensor.shape
                for area, tensor in vision_xs[0].items()
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

        return dict(
            sample_ids=sample_ids,
            lang_x=lang_xs,
            vision_x=collated_vision_xs,
            mask_x=collated_mask_xs,
            fvlm_mask_x=collated_fvlm_mask_xs,
            region2area = region2areas,
            dropped_regions=dropped_regions,
            attention_mask=attention_masks,
            labels = labels,
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        # each dataset item is {"full": {...}, "mask": {...}}; collate the two
        # branches separately so CustomTrainer can forward/backward them independently
        return dict(
            full=self._collate_branch(
                [instance['full'] for instance in instances],
                is_masked_branch=False,
            ),
            mask=self._collate_branch(
                [instance['mask'] for instance in instances],
                is_masked_branch=True,
            ),
        )


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Full prompt：完整 image tokens + Full RWLKE region tokens，生成 full report。
        full_inputs = {
            key: value
            for key, value in inputs['full'].items()
            if key != 'dropped_regions'
        }
        full_output = model(
            **full_inputs,
            return_rwlke_region_embedding=True,
        )
        full_rwlke_region_embedding = full_output['rwlke_region_embedding']

        # Mask prompt：masked image tokens + 混合 region tokens。
        # 未遮挡器官复制 Full RWLKE token，被遮挡器官使用 33 个全零占位 token。
        masked_region_embedding = full_rwlke_region_embedding.clone()
        dropped_regions_batch = inputs['mask']['dropped_regions']
        region2areas_batch = inputs['mask']['region2area']
        region_token_len = REGION_TOKEN_LEN

        for batch_index, dropped_regions in enumerate(dropped_regions_batch):
            if inputs['full']['region2area'][batch_index] != region2areas_batch[batch_index]:
                raise ValueError("Full and Mask prompts must use identical region order")
            dropped_regions = set(dropped_regions)
            for region_index, area in region2areas_batch[batch_index].items():
                if area not in dropped_regions:
                    continue
                start = region_index * region_token_len
                end = start + region_token_len
                if end > masked_region_embedding.size(1):
                    raise IndexError(
                        f"Region slot {region_index} exceeds the Full RWLKE embedding"
                    )
                masked_region_embedding[batch_index, start:end, :] = 0.0

        mask_inputs = {
            key: value
            for key, value in inputs['mask'].items()
            if key != 'dropped_regions'
        }
        mask_output = model(
            **mask_inputs,
            precomputed_region_embedding=masked_region_embedding,
        )

        full_loss = full_output['loss']
        mask_loss = mask_output['loss']
        loss = full_loss + 0.8 * mask_loss

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(flush=True)
            print(
                f"total_loss: {loss.detach().item():.4f} | "
                f"full_loss: {full_loss.detach().item():.4f} | "
                f"mask_loss: {mask_loss.detach().item():.4f}",
                flush=True,
            )

        if return_outputs:
            return loss, {'full': full_output, 'mask': mask_output}
        return loss

    def log(self, logs):
        # Hide Trainer's rounded epoch field without changing training state.
        current_epoch = self.state.epoch
        try:
            self.state.epoch = None
            super().log(logs)
        finally:
            self.state.epoch = current_epoch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
def main():
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
 
    print("Setup Model")
    model = Reg2RG(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        #---Dan---
        pretrained_finegrain_visual_encoder=model_args.pretrained_finegrained_visual_encoder,
        #---Dan---
        pretrained_adapter=model_args.pretrained_adapter,
        bank_npy_path=data_args.bank_npy_path,
        organ_annotation_path = data_args.organ_annotation_path,

    )

    print("Setup Data")
    print("!!!!")
    Train_dataset = RadGenomeDataset_Train(
        text_tokenizer=model.text_tokenizer,
        image_padding_tokens=model.image_padding_tokens,
        region_padding_tokens=model.region_padding_tokens,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        cache_dir=data_args.monai_cache_dir,
        #---Dan---
        use_fvlm=model_args.pretrained_finegrained_visual_encoder is not None,
        # Canonical fVLM processed data; do not derive it from Reg2RG's raw-data root.
        fvlm_processed_root=data_args.fvlm_processed_root,
        #---Dan---
    )
   
    trainer = CustomTrainer(model=model,
                      train_dataset = Train_dataset,
                      args = training_args,
                      data_collator=DataCollator(),
                      )
 
    trainer.train()
    trainer.save_state()
     
if __name__ == "__main__":
    main()
