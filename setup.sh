#!/bin/bash

echo "Installing PyTorch CPU version first..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Checking for CUDA availability..."
cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")

if [ "$cuda_available" = "True" ]; then
    echo "CUDA detected, installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements-cuda.txt
else
    echo "No CUDA detected, using CPU version..."
    pip install -r requirements-cpu.txt
fi