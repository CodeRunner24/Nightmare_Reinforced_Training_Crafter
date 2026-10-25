"""Installation checks only: no policy, training, rollout, or benchmark."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import timeit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tqdm import tqdm

started = timeit.default_timer()
report = {'python': sys.version, 'checks': {}, 'training_started': False}
modules = ['numpy', 'jax', 'chex', 'ninjax', 'optax', 'elements', 'portal',
           'scope', 'av', 'crafter', 'embodied', 'dreamerv3.agent']
for name in tqdm(modules, desc='Checking installed imports', unit='module'):
    before = timeit.default_timer()
    importlib.import_module(name)
    report['checks'][name] = round(timeit.default_timer() - before, 3)
    tqdm.write(f'OK import {name}')

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from embodied.envs.crafter import Crafter
import embodied.jax.internal as internal

# Use the upstream setup path, with version-specific optional XLA tuning disabled.
internal.setup(platform='cuda', gpuflags=False, transfer_guard=False)
devices = jax.devices('cuda')
assert devices, 'No JAX GPU device detected'
result = jax.jit(lambda x: x + 1)(jnp.ones((2, 2))).block_until_ready()
assert np.array_equal(np.asarray(result), np.full((2, 2), 2))
report['gpu'] = [str(device) + ' ' + device.device_kind for device in devices]
print('OK JAX GPU:', report['gpu'], flush=True)

# Reset only to verify packaged game assets and the Dreamer adapter. No game steps.
env = Crafter('reward', seed=0)
obs = env.step({'reset': True, 'action': 0})
assert obs['image'].shape == (64, 64, 3) and obs['is_first']
Image.fromarray(obs['image']).resize((512, 512), Image.Resampling.NEAREST).save(
    ROOT / 'setup' / 'crafter_initial.png')
report['crafter'] = {'image_shape': list(obs['image'].shape),
                     'actions': int(env.act_space['action'].high),
                     'reset_only': True, 'game_steps': 0}
env.close()
report['versions'] = {
    name: metadata.version(name)
    for name in ['jax', 'jaxlib', 'numpy', 'crafter', 'ninjax', 'optax']}
report['elapsed_seconds'] = round(timeit.default_timer() - started, 3)
(ROOT / 'setup' / 'environment-check.json').write_text(
    json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
