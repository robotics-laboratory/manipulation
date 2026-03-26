# Torch Installation Fix for Isaac Sim Container

## The Error

When running any script via `isaaclab.sh -p`, every Isaac Sim extension
failed to load with:

```
ModuleNotFoundError: No module named 'torch'
```

This cascaded through ~15 extensions (isaacsim.core.simulation_manager,
isaacsim.core.prims, isaacsim.sensors.camera, isaaclab_tasks, etc.),
making the entire environment unusable.

## Root Cause

Isaac Sim 5.1.0 bundles its own copy of PyTorch at:

```
/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/
```

Isaac Sim's extension loader adds `pip_prebundle/` to `sys.path`, so all
extensions import torch from there (not from the standard `site-packages`).

The bundled torch also has a matching `.dist-info` entry in site-packages,
so pip considers it "already installed".

### What went wrong

When we ran:

```bash
/isaac-sim/python.sh -m pip install "lerobot[smolvla]>=0.4.0"
```

LeRobot depends on torch. Since pip saw torch as already installed (via
the `.dist-info`), it did **not** reinstall torch to site-packages.
However, installing lerobot's other dependencies **partially corrupted**
the bundled `pip_prebundle/torch/` directory.

Result: the bundled torch was broken, and there was no fallback in
site-packages.

> **Note:** lerobot's install also causes two more problems — numpy 2.x
> breaking Omniverse's C extensions, and packaging 25.0 breaking pip itself.
> See [numpy-fix-notes.md](numpy-fix-notes.md) for the full chain.

### Why the first fix attempt failed (commit 2349e41)

Simply removing `pip_prebundle/torch` before pip install didn't help
because pip still saw torch's `.dist-info` as present and skipped
reinstalling it. The bundled copy was gone, site-packages had nothing.

### Why the second fix attempt failed (commit f57521b)

The symlink approach:

```bash
TORCH_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
ln -s "$TORCH_SITE/torch" /path/to/pip_prebundle/torch
```

Failed because `site.getsitepackages()[0]/torch` didn't exist — pip never
installed torch there (it thought torch was already installed).

### Why the third fix attempt failed (commit 2e34e2e)

Using `pip install "torch>=2.0.0"` (unpinned) installed the latest torch
from PyPI. The default PyPI wheel is built against newer CUDA libraries
and requires `libcusparseLt.so.0`, which is **not present** in the Isaac
Sim 5.1.0 container. Build error:

```
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
```

## The Working Fix (commit 6f1a855, refined in current Dockerfile)

All steps below are merged into a single `RUN` layer in the Dockerfile:

### 1. Uninstall bundled torch (removes .dist-info)

```bash
/isaac-sim/python.sh -m pip uninstall -y torch
```

This removes the `.dist-info` metadata so pip no longer thinks torch is
installed.

### 2. Remove the physical bundled copy

```bash
rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch
```

### 3. Install the correct torch version for Isaac Sim's CUDA

Isaac Sim 5.1.0 ships CUDA 12.8 libraries. We must install the matching
torch wheel:

```bash
/isaac-sim/python.sh -m pip install \
    "torch==2.7.0" --extra-index-url https://download.pytorch.org/whl/cu128
```

Then install lerobot with `--no-deps` and its dependencies manually (see
[numpy-fix-notes.md](numpy-fix-notes.md) for why `--no-deps` is required):

```bash
/isaac-sim/python.sh -m pip install --no-deps "lerobot[smolvla]==0.4.3"
/isaac-sim/python.sh -m pip install \
    "datasets>=4.0.0,<4.2.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
    ...  # full list in Dockerfile and numpy-fix-notes.md
```

### 4. Symlink for Isaac Sim extensions

Isaac Sim's extension loader only looks in `pip_prebundle/`, so we
symlink it to the freshly installed torch:

```bash
TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
ln -s "$TORCH_DIR" /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch
```

### 5. Verify

```bash
/isaac-sim/python.sh -c "import torch; print(f'torch {torch.__version__} OK')"
```

## Key Takeaways

1. **Isaac Sim bundles torch with dist-info** — pip thinks it's installed
   even if the files are corrupted. You must `pip uninstall` first.

2. **Pin torch to match Isaac Sim's CUDA** — Isaac Sim 5.1.0 uses CUDA
   12.8. The default PyPI torch wheel targets different CUDA libs. Use
   `torch==2.7.0` from `https://download.pytorch.org/whl/cu128`.

3. **Symlink pip_prebundle → site-packages** — Isaac Sim extensions only
   look in `pip_prebundle/`. A symlink to the real install is the cleanest
   fix.

4. **Use `torch.__file__` to find the install path** — More reliable than
   `site.getsitepackages()` since Isaac Sim's Python environment is
   non-standard.

## Namespace Package vs inspect.getfile (tensordict crash)

### The Error

After the torch 2.7.0 install, running any script that imports `rsl_rl`
crashes with:

```
TypeError: <module 'isaaclab' (<_frozen_importlib_external.NamespaceLoader ...>)> is a built-in module
```

### Root Cause

`tensordict` calls `torch.compiler.allow_in_graph` at module scope, which
triggers `import torch._dynamo`. Deep inside `torch._dynamo`, the decorator
`@torch.library.register_fake` calls `inspect.getframeinfo` to record source
info. `inspect.getframeinfo` → `inspect.getmodule` → `inspect.getfile`
walks the call stack and encounters the `isaaclab` module.

`isaaclab` is a **namespace package** (loaded via `NamespaceLoader`, no
`__file__` attribute). `torch.package.package_importer` has monkey-patched
`inspect.getfile`, but its fallback to the original `getfile` raises
`TypeError` for modules without `__file__`.

### The Fix

Set `isaaclab.__file__` before `rsl_rl` (and thus `tensordict`) is imported:

```python
import isaaclab as _isaaclab_ns
if not getattr(_isaaclab_ns, "__file__", None):
    _isaaclab_ns.__file__ = next(iter(_isaaclab_ns.__path__), __file__)
```

Applied to all scripts in `scripts/rsl_rl/` that import `rsl_rl`:
`train.py`, `play.py`, `collect_trajectories.py`, `collect_bc_dataset.py`,
`train_bc.py`.

## References

- https://github.com/isaac-sim/IsaacLab/issues/2652
- https://github.com/isaac-sim/IsaacLab/issues/3788
- https://github.com/isaac-sim/IsaacLab/discussions/3770
