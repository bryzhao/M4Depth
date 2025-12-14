#!/bin/bash
# M4Depth training script with proper CUDA library paths

cd /home/bryan/dev/final_project/M4Depth
source /home/bryan/dev/final_project/m4depth_env/bin/activate

NVIDIA_BASE="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages/nvidia"

export PYTHONPATH="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cuda_nvrtc/lib:${NVIDIA_BASE}/cuda_runtime/lib:$LD_LIBRARY_PATH"
export PATH="${NVIDIA_BASE}/cuda_nvcc/bin:$PATH"

python3 main.py --mode=train --dataset=midair --seq_len=4 --db_seq_len=8 \
  --arch_depth=6 --ckpt_dir=./checkpoints --log_dir=./checkpoints/summaries \
  --records=data/midair/train_data/ --enable_validation --batch_size=3
