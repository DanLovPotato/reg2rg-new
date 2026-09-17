import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_vit3d.pth")
    #---Dan---
    pretrained_finegrained_visual_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "10-organ fVLM finetuned checkpoint_XXX.pth used by eval_finetune.py; enables the frozen exact fVLM organ branch."},
    )
    #---Dan---
    pretrained_adapter: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_perceiver_fc.pth")
    ckpt_path: Optional[str] = field(   
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/pytorch_model.bin")

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_preprocessed')
    mask_folder: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_region_mask')
    # Separate from data_folder so fVLM always sees its canonical 9-label masks.
    fvlm_processed_root: Optional[str] = field(
        default=None,
        metadata={"help": "Canonical fVLM data root containing processed_{train,valid}_{images,masks}; required when fVLM is enabled."},
    )
    report_file: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    monai_cache_dir: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache')
    result_path: Optional[str] = field(default='/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports.csv')
    bank_npy_path: Optional[str] = field(default=None)
    organ_annotation_path: Optional[str] = field(default=None)
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Stop after this many total reports have been written."},
    )
