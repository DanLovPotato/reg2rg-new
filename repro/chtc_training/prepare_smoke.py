import hashlib
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1]).resolve()
staging = Path('/staging/t/txu98')
bank = root / 'bank'
expected = {
    'organ_report_embeddings.npz': '8b229026ae3bbd7d8b9ac61c010e57ce845e00d5e98a990bd68853accc6280b8',
    'organ_annotation.json': '989d339f2d748c962d765ad1d794405b56cdaed78c6be71c0d53f290baa68625',
}
for name, digest in expected.items():
    assert hashlib.sha256((bank / name).read_bytes()).hexdigest() == digest, name
z = np.load(bank / 'organ_report_embeddings.npz', allow_pickle=False)
organs = ['abdomen', 'bone', 'breast', 'esophagus', 'heart', 'lung',
          'mediastinum', 'pleura', 'thyroid', 'trachea and bronchie']
assert z['feats'].shape == (24126, 10, 256)
assert set(z['organs'].tolist()) == set(organs)
ids = set(z['patient_ids'].tolist())
annotations = {r['id']: r for r in json.loads((bank / 'organ_annotation.json').read_text())['train']}

def extract(archive, members):
    for member in members:
        target = (root / member.name).resolve()
        assert root in target.parents and member.isfile(), member.name
        archive.extract(member, root)

with tarfile.open(root / 'radgenome_first500_masks_reports.tar') as ar:
    members = ar.getmembers()
    reports = [m for m in members if m.name.endswith('/train_region_report.csv')]
    assert len(reports) == 1
    extract(ar, reports)
    report_path = root / reports[0].name
    df = pd.read_csv(report_path)
    grouped = {key: group for key, group in df.groupby('Volumename')}
    mask_names = {m.name for m in members if m.isfile()}
    with tarfile.open(staging / 'radgenome_first500/radgenome_ct_batch_0.tar') as ct:
        candidates = sorted((m for m in ct.getmembers() if m.name.endswith('.nii.gz')), key=lambda m: m.name)
        selected = []
        selected_masks = set()
        samples = []
        for m in candidates:
            name = Path(m.name).name
            sample_id = name.removesuffix('.nii.gz')
            if name not in grouped or sample_id not in annotations or sample_id not in ids:
                continue
            sample = {'image': str(root / m.name)}
            masks = set()
            for row in grouped[name].itertuples():
                if row.Anatomy in organs:
                    mask = f'dataset/train_region_mask/seg_{sample_id}/{row.Anatomy}.nii.gz'
                    if mask not in mask_names:
                        raise RuntimeError(f'Missing mask: {mask}')
                    masks.add(mask)
                    sample[row.Anatomy] = [str(root / mask), row.Sentence]
            if len(sample) == 1:
                continue
            for organ in organs:
                for neighbor in annotations[sample_id].get(f'{organ}_indices', [])[:5]:
                    assert neighbor['id'] in ids and neighbor.get('organ', organ) in organs
            selected.append(m)
            selected_masks.update(masks)
            samples.append(sample)
            if len(samples) == 12:
                break
        assert len(samples) == 12, len(samples)
        extract(ct, selected)
    extract(ar, [m for m in members if m.name in selected_masks])

# Match the upstream dataset's cache schema; create once before the 3 ranks
# start, avoiding concurrent cache writes and any stale upstream paths.
cache = root / 'code/src/Dataset/train_samples.pkl'
with cache.open('wb') as f:
    pickle.dump(samples, f)
manifest = {'commit': '2f75ae67073343ec954b4de9c9242d8c2ecbe75d',
            'bank_sha256': expected, 'bank_shape': list(z['feats'].shape),
            'bank_organs': z['organs'].tolist(), 'sample_count': len(samples),
            'sample_ids': [Path(s['image']).name for s in samples],
            'report_path': str(report_path), 'initialization': 'Llama/RadFM only; no Reg2RG checkpoint',
            'world_size': 2, 'max_steps': 3, 'gradient_accumulation_steps': 8}
(root / 'preflight.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
