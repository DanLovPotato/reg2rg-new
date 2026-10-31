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
    pretrained_adapter: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_perceiver_fc.pth")
    ckpt_path: Optional[str] = field(   
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/pytorch_model.bin")
    # --Dan--- organ-branch perceiver/fc fine-tuned in radfm_perceiver_align
    # (perceiver-stepN.pth / fc-stepN.pth). Unset = organ branch shares the
    # full-CT perceiver/fc (original Reg2RG).
    organ_perceiver_path: Optional[str] = field(default=None)
    organ_fc_path: Optional[str] = field(default=None)
    freeze_organ_perceiver: bool = field(default=True)
    freeze_organ_fc: bool = field(default=False)

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_preprocessed')
    mask_folder: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_region_mask')
    report_file: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    monai_cache_dir: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache')
    result_path: Optional[str] = field(default='/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports.csv')
    bank_npy_path: Optional[str] = field(default=None)
    organ_annotation_path: Optional[str] = field(default=None)
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Stop after this many total reports have been written."},
    )
