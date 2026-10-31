"""Lightweight check: imports/config only; never constructs a model."""
import importlib
import importlib.metadata
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'code/src'))
for name in ['torch', 'transformers', 'accelerate', 'peft', 'deepspeed',
             'pydantic', 'torch_geometric', 'monai', 'nibabel', 'pandas']:
    importlib.import_module(name)
    print(name, importlib.metadata.version(name), flush=True)
import train_radgenome
from transformers.deepspeed import HfDeepSpeedConfig
HfDeepSpeedConfig(str(root / 'code/ds_configs/stage2.json'))
print('PASS: upstream training entry point and HF DeepSpeed config import', flush=True)
