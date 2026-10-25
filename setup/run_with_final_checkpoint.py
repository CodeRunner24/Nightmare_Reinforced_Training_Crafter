"""Run upstream main unchanged; save the final training state on normal return."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import elements

checkpoints = []
OriginalCheckpoint = elements.Checkpoint


class ObservedCheckpoint(OriginalCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        checkpoints.append(self)


elements.Checkpoint = ObservedCheckpoint
runpy.run_path(str(ROOT / 'dreamerv3' / 'main.py'), run_name='__main__')
for checkpoint in checkpoints:
    if not {'step', 'agent', 'replay'} <= checkpoint._saveables.keys():
        continue
    # The upstream loop otherwise saves only on a wall-clock interval.
    checkpoint.replay.save_wait = True
    print(f'FINAL CHECKPOINT: step={int(checkpoint.step)}', flush=True)
    checkpoint.save()
    info = {
        'final_step': int(checkpoint.step),
        'optimizer_calls_this_process': int(checkpoint.agent.n_updates),
        'checkpoint': str(checkpoint.latest()),
        'replay': checkpoint.replay.stats(),
    }
    chunk = next((c for c in checkpoint.replay.chunks.values() if c.data), None)
    if chunk:
        info['replay_fields'] = {
            name: {'shape_per_transition': list(value.shape[1:]),
                   'dtype': str(value.dtype),
                   'bytes_per_transition': value.nbytes // value.shape[0]}
            for name, value in chunk.data.items()}
        info['raw_replay_bytes_per_transition'] = sum(
            v['bytes_per_transition'] for v in info['replay_fields'].values())
    folder = Path(str(checkpoint._directory)).parent
    (folder / 'final_state.json').write_text(json.dumps(info, indent=2) + '\n')
    print('FINAL STATE:', json.dumps(info, indent=2), flush=True)
