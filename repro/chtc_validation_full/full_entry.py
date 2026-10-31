"""Full-only inference with the unchanged, strictly loaded dual-branch model."""
import os
import sys
import re
import json
import hashlib
from pathlib import Path
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader

root = Path(os.environ['VALIDATION_ROOT'])
out = Path(os.environ['VALIDATION_OUT'])
sys.path.insert(0, str(root / 'code/src'))
import test_radgenome as original

COLUMNS = ['AccNum', 'Question', 'GT_combined_report', 'Full_pred_combined_report']

class FullDataset(original.RadGenomeDataset_Test):
    def prepare_samples(self):
        samples = super().prepare_samples()
        samples.sort(key=lambda r: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', r['accnum'])])
        return samples

    def __getitem__(self, index):
        sample = self.data[index]
        regions = {k: v[1] for k, v in sample.items() if k not in ('image', 'accnum')}
        masks = {k: v[0] for k, v in sample.items() if k not in ('image', 'accnum')}
        crops, mask_tensors = self.mask_img_to_tensor(sample['image'], masks)
        regions = {k: v for k, v in regions.items() if k in crops}
        full = self._build_branch(list(regions), regions, crops, mask_tensors,
                                  dropped_regions=set(), sample_id=sample['accnum'].removesuffix('.nii.gz'))
        return {'acc_num': sample['accnum'], 'full': full}

class FullCollator(original.DataCollator):
    def __call__(self, rows):
        return {'acc_num': [r['acc_num'] for r in rows],
                'full': self._collate_branch([r['full'] for r in rows], False)}

def main():
    args, data_args = transformers.HfArgumentParser((original.ModelArguments, original.DataArguments)).parse_args_into_dataclasses()
    annotation = Path(data_args.organ_annotation_path)
    assert hashlib.sha256(annotation.read_bytes()).hexdigest() == 'fe4441b4b686e3bd586f91a7cedbea7a5320eaa47709029ee4681970a6a90013'
    records = json.loads(annotation.read_text())['validation']
    expected = {r['id'] for r in records}
    assert len(records) == len(expected) == 1564
    path = Path(data_args.result_path)
    if path.exists():
        completed = pd.read_csv(path)
        assert list(completed.columns) == COLUMNS and completed.AccNum.is_unique
    else:
        completed = pd.DataFrame(columns=COLUMNS)
        completed.to_csv(path, index=False)
    os.makedirs(data_args.monai_cache_dir, exist_ok=True)
    dataset = FullDataset(text_tokenizer=args.tokenizer_path, data_folder=data_args.data_folder,
                          mask_folder=data_args.mask_folder, csv_file=data_args.report_file,
                          cache_dir=data_args.monai_cache_dir, inferenced_id=completed.AccNum.tolist())
    actual = {r['accnum'].removesuffix('.nii.gz') for r in dataset.samples}
    assert actual | {s.removesuffix('.nii.gz') for s in completed.AccNum} == expected
    loader = DataLoader(dataset, batch_size=1, num_workers=4, prefetch_factor=2,
                        pin_memory=True, shuffle=False, collate_fn=FullCollator())
    model = original.Reg2RG(text_tokenizer_path=args.tokenizer_path, lang_model_path=args.lang_encoder_path,
                            pretrained_visual_encoder=args.pretrained_visual_encoder,
                            pretrained_adapter=args.pretrained_adapter, bank_npy_path=data_args.bank_npy_path,
                            organ_annotation_path=data_args.organ_annotation_path)
    weights = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(weights, strict=True)
    del weights
    model.cuda().eval()
    print('Strict checkpoint load OK; Full-only generation; no Mask samples or Mask forwards.', flush=True)
    for batch in original.tqdm.tqdm(loader):
        full = batch['full']
        with torch.inference_mode():
            reports = model.generate(full['lang_x'].cuda(),
                {k: v.cuda() for k, v in full['vision_x'].items()},
                {k: v.cuda() for k, v in full['mask_x'].items()},
                full['region2area'], sample_ids=full['sample_ids'])
        pd.DataFrame([[batch['acc_num'][0], full['question'][0],
                       full['gt_combined_report'][0], reports[0]]], columns=COLUMNS).to_csv(
                           path, mode='a', index=False, header=False)
        print('Saved Full report:', batch['acc_num'][0], flush=True)
    result = pd.read_csv(path)
    assert len(result) == result.AccNum.nunique() == 1564
    assert set(result.AccNum.str.replace('.nii.gz', '', regex=False)) == expected
    assert result.Full_pred_combined_report.fillna('').str.strip().ne('').all()
    result.to_csv(out / 'full_reports.csv', index=False)
    print('COMPLETE: 1564 Full reports.', flush=True)

if __name__ == '__main__':
    main()
