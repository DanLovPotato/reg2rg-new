"""Extract all training archives and validate the upstream dataset cache schema."""
import csv
import json
import pickle
import shutil
import tarfile
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1]).resolve()
staging = Path('/staging/t/txu98')
organs = ['abdomen', 'bone', 'breast', 'esophagus', 'heart', 'lung',
          'mediastinum', 'pleura', 'thyroid', 'trachea and bronchie']

def unpack(path, destination):
    print('Extracting', path, flush=True)
    with tarfile.open(path) as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            assert target == destination or destination in target.parents
            assert member.isfile() or member.isdir(), member.name
            archive.extract(member, destination)

archives = sorted((staging / 'radgenome_first500').glob('radgenome_ct_batch_*.tar'))
archives += sorted((staging / 'radgenome_train_remaining').glob('radgenome_ct_remaining_batch_*.tar'))
assert len(archives) == 966, len(archives)  # 20 existing + ceil(23629 / 25)
assert shutil.disk_usage(root).free > 1_200_000_000_000, 'Insufficient scratch for full CT set'
for archive in archives:
    unpack(archive, root)
dataset = root / 'dataset'
unpack(staging / 'radgenome_train_companion/train_region_mask.tar.gz', dataset)
unpack(staging / 'radgenome_train_companion/radgenome_train_reports.tar', dataset)
images = sorted((dataset / 'train_preprocessed').glob('*/*/*.nii.gz'))
assert len(images) == 24129, len(images)
by_name = {p.name: p for p in images}
assert len(by_name) == len(images)
df = pd.read_csv(dataset / 'radgenome_files/train_region_report.csv')
samples = []
for name, group in df.groupby('Volumename', sort=True):
    regions = {row.Anatomy: row.Sentence for row in group.itertuples() if row.Anatomy in organs}
    if not regions:
        continue
    assert name in by_name, f'Missing CT: {name}'
    sample = {'image': str(by_name[name])}
    for organ, sentence in regions.items():
        mask = dataset / 'train_region_mask' / ('seg_' + name.removesuffix('.nii.gz')) / (organ + '.nii.gz')
        assert mask.is_file() and mask.stat().st_size > 0, f'Missing mask: {mask}'
        sample[organ] = [str(mask), sentence]
    samples.append(sample)
assert len(samples) >= 24000, len(samples)
with (root / 'code/src/Dataset/train_samples.pkl').open('wb') as f:
    pickle.dump(samples, f)
annotation = json.loads((root / 'bank/organ_annotation.json').read_text())
ids = {r['id'] for r in annotation['train']}
missing_bank = [Path(s['image']).name.removesuffix('.nii.gz') for s in samples
                if Path(s['image']).name.removesuffix('.nii.gz') not in ids]
manifest = {'ct_count': len(images), 'train_samples': len(samples),
            'missing_annotation_ids': missing_bank,
            'missing_annotation_policy': 'Unmodified upstream: empty retrieval lists, not dropped samples',
            'epochs': 10, 'world_size': 2, 'per_device_batch': 1, 'gradient_accumulation_steps': 8,
            'effective_batch': 16, 'initialization': 'Llama/RadFM pretrained only',
            'commit': '2f75ae67073343ec954b4de9c9242d8c2ecbe75d'}
(root / 'full_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
