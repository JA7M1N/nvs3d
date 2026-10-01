#!/bin/bash
# colab_setup.sh — set up nvs3d in Colab/Kaggle (assumes torch is pre-installed)
set -e

echo "=== nvs3d Colab/Kaggle setup ==="

# Install the package in editable mode without its torch dependency
pip install -e . --no-deps

# Install only the small missing dependencies (no torch, no torchvision)
pip install -r requirements-colab.txt

echo ""
echo "=== Environment check ==="
python -c "
import torch
print(f'torch        = {torch.__version__}')
print(f'cuda_avail   = {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu_name     = {torch.cuda.get_device_name(0)}')
else:
    print('gpu_name     = N/A (no GPU)')
"

echo ""
echo "=== Setup complete ==="
