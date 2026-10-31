"""Run the unchanged upstream trainer with finite-loss and durable-save guards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers

sys.path.insert(0, str(Path(os.environ['TRAIN_ROOT']) / 'code/src'))
import train_radgenome as upstream

publication = Path(os.environ['TRAIN_OUTPUT'])
started = time.monotonic()
timed_stop = False

class DurableCheckpoint(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global timed_stop
        stop = torch.tensor([int(time.monotonic() - started > 6 * 86400)], device=args.device)
        torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item():
            timed_stop = True
            control.should_training_stop = True
            control.should_save = True
        if state.global_step >= state.max_steps:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        torch.distributed.barrier()
        failed = torch.zeros(1, dtype=torch.int32, device=args.device)
        if state.is_world_process_zero:
            try:
                name = f'checkpoint-{state.global_step}'
                temporary = publication / (name + '.tar.partial')
                destination = publication / (name + '.tar')
                subprocess.run(['tar', '-cf', str(temporary), '-C', args.output_dir, name], check=True)
                temporary.replace(destination)
                marker = publication / 'latest_checkpoint.json.partial'
                marker.write_text(json.dumps({'archive': destination.name, 'global_step': state.global_step}))
                marker.replace(publication / 'latest_checkpoint.json')
                # Retain two recoverable snapshots from this run only.
                previous = sorted(publication.glob('checkpoint-*.tar'), key=lambda p: int(p.stem.split('-')[-1]))
                for old in previous[:-2]:
                    old.unlink()
                print('PUBLISHED_CHECKPOINT', destination, flush=True)
            except Exception as exc:
                print('CHECKPOINT_PUBLICATION_FAILED', repr(exc), flush=True)
                failed.fill_(1)
        torch.distributed.broadcast(failed, src=0)
        if failed.item():
            raise RuntimeError('Durable checkpoint publication failed')

class GuardedTrainer(upstream.CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(DurableCheckpoint())

    def compute_loss(self, *args, **kwargs):
        output = super().compute_loss(*args, **kwargs)
        loss = output[0] if isinstance(output, tuple) else output
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Non-finite Full/Mask training loss')
        return output

    def train(self, *args, **kwargs):
        resume = os.environ.get('RESUME_CURRENT_RUN', '')
        if resume:
            kwargs['resume_from_checkpoint'] = resume
        return super().train(*args, **kwargs)

upstream.CustomTrainer = GuardedTrainer
upstream.main()
if torch.distributed.get_rank() == 0:
    (publication / 'attempt_status.json').write_text(json.dumps({'status': 'CONTINUE' if timed_stop else 'COMPLETE'}))
torch.distributed.barrier()
# Shell maps torchrun's worker failure status to a controlled resubmission
# using the explicit status file; no old model checkpoint is ever loaded.
if timed_stop:
    sys.exit(85)
