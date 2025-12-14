#!/bin/bash
# M4Depth evaluation script - runs on test set and prints metrics

cd /home/bryan/dev/final_project/M4Depth
source /home/bryan/dev/final_project/m4depth_env/bin/activate

NVIDIA_BASE="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages/nvidia"

export PYTHONPATH="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cuda_nvrtc/lib:${NVIDIA_BASE}/cuda_runtime/lib:$LD_LIBRARY_PATH"
export PATH="${NVIDIA_BASE}/cuda_nvcc/bin:$PATH"

# Create symlinks so eval mode finds the checkpoint
# (eval looks in ckpt_dir/best, but our weights are in ckpt_dir/train)
mkdir -p ./checkpoints/best
if [ ! -L "./checkpoints/best/checkpoint" ]; then
    echo "Setting up checkpoint symlinks..."
    for f in ./checkpoints/train/*; do
        ln -sf "../train/$(basename $f)" "./checkpoints/best/$(basename $f)" 2>/dev/null
    done
fi

echo "============================================"
echo "Running evaluation on MidAir TEST set"
echo "============================================"

python3 main.py \
    --mode=eval \
    --dataset=midair \
    --seq_len=4 \
    --arch_depth=6 \
    --ckpt_dir=./checkpoints \
    --records_path=./data/midair/test_data \
    --log_dir=./checkpoints/summaries/eval \
    "$@"

echo ""
echo "============================================"
echo "Metrics saved to: ./checkpoints/perfs-midair.txt"
echo "============================================"
