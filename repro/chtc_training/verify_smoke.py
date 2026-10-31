import json
import math
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:3])
world_size = int(sys.argv[3])
state = json.loads((root / 'trainer_state.json').read_text())
assert state['global_step'] == 3, state['global_step']
losses = [r['loss'] for r in state['log_history'] if 'loss' in r]
assert len(losses) == 3 and all(math.isfinite(x) for x in losses), losses
checkpoint = root / 'checkpoint-3'
assert (checkpoint / 'trainer_state.json').is_file()
assert any(p.suffix in ('.bin', '.safetensors') for p in checkpoint.iterdir())
optim = list(checkpoint.rglob('*optim_states.pt'))
assert len(optim) == world_size, [str(p) for p in optim]
result = {'status': 'PASS', 'steps': 3, 'losses': losses,
          'optimizer_shards': len(optim), 'world_size': world_size,
          'note': 'Smoke only; not a model-quality evaluation or full training run.'}
(output / 'smoke_result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
