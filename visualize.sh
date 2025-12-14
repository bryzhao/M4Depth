#!/bin/bash
# M4Depth visualization script with proper CUDA library paths

cd /home/bryan/dev/final_project/M4Depth
source /home/bryan/dev/final_project/m4depth_env/bin/activate

NVIDIA_BASE="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages/nvidia"

export PYTHONPATH="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cuda_nvrtc/lib:${NVIDIA_BASE}/cuda_runtime/lib:$LD_LIBRARY_PATH"
export PATH="${NVIDIA_BASE}/cuda_nvcc/bin:$PATH"

# Run visualization with test data
# Options:
#   --num_samples: number of images to generate (default 10)
#   --stride: skip every N samples for variety (default 1)
#   --shuffle: randomize sample selection
#   --use_train_data: use training data instead of test
#   --output_dir: where to save images (default: visualizations)
#   --ckpt_dir: checkpoint directory (default: ./ckpt)

python3 visualize_predictions.py \
    --num_samples=30 \
    --stride=100 \
    --output_dir=./visualizations \
    --ckpt_dir=./checkpoints \
    "$@"
