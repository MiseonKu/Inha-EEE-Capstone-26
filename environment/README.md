# Environment

This folder contains dependency specifications for rerunning the submitted preprocessing and ML scripts.

## Tested CUDA Environment

The final GPU KRR scripts require CUDA. The package was verified on:

```text
GPU: NVIDIA GeForce RTX 2070 SUPER, 8192 MiB
NVIDIA driver: 610.62
PyTorch: 2.5.1+cu121
CUDA reported by PyTorch: 12.1
torch.cuda.is_available(): True
```

Verification evidence is stored outside the submitted package at:

```text
05_goal_and_log/2026-06-25/evidence/submission_package_cuda_env.txt
```

## Conda Setup

```powershell
conda env create -f environment\environment.yml
conda activate capstone-reproduction
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Pip Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r environment\requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If CUDA is unavailable, CPU-only runs may still help inspect data loading or preprocessing, but they must not be used as final metric reproduction evidence.
