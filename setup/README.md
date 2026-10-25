# DreamerV3 + Crafter local environment

Repository: https://github.com/danijar/dreamerv3

Checked-out revision: `e01491fad6434b2245a3b8ca201dd7faedcc458c`.

This environment is installed under `.venv-wsl` and runs in WSL2 Ubuntu, not
Windows Python. The managed Python interpreter is in the WSL user's uv directory.
The upstream training code and `requirements.txt` are unchanged.

## Open the installed environment

From PowerShell:

```powershell
wsl -d Ubuntu
```

Then inside Ubuntu:

```bash
cd /mnt/c/Users/atak6/Downloads/Dream
source setup/activate.sh
```

## Reinstall

```bash
bash setup/install-wsl.sh
```

`requirements-crafter-rtx50.txt` is a Crafter-specific dependency set. It uses
JAX 0.6.2 with CUDA 12 wheels instead of upstream JAX 0.4.33 and its old CUDA
compiler constraint. NumPy stays at 1.26.4. Exact installed versions are recorded
in `requirements-crafter-rtx50.lock.txt` after installation.

Other benchmark suites such as Atari ROMs, Minecraft, DMLab, and MuJoCo are not
needed for Crafter and are not installed by this setup.

## Installation verification

```bash
python setup/check_environment.py
```

This imports dependencies, uses the upstream JAX initialization helper, checks a
tiny GPU calculation, and resets the Crafter adapter once to load its assets.
It does not construct or train an agent, advance the game, collect a dataset,
or measure training speed. Outputs:

- `environment-check.json`: installation check results and timings.
- `crafter_initial.png`: initial game screen, enlarged with nearest-neighbor.
- `check.log`: terminal output from the initial check.
- `install.log`: installation output.

The installation check itself performs no training. Subsequently, a 300-step
debug smoke test and a bounded real-model Crafter pilot were completed.
Measured real-model results are in
`logs/crafter_real_profile_02/REPORT_TR.md`, alongside telemetry and a PNG plot.
No full reproduction run has been performed.
