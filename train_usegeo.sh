#!/bin/bash
# UseGeo fine-tuning script - fine-tunes MidAir-pretrained model on real-world UseGeo data
# This is Phase 2 of the RBE 577 Final Project

cd /home/bryan/dev/final_project/M4Depth
source /home/bryan/dev/final_project/m4depth_env/bin/activate

NVIDIA_BASE="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages/nvidia"

export PYTHONPATH="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cuda_nvrtc/lib:${NVIDIA_BASE}/cuda_runtime/lib:$LD_LIBRARY_PATH"
export PATH="${NVIDIA_BASE}/cuda_nvcc/bin:$PATH"

echo "============================================"
echo "Phase 2: UseGeo Fine-tuning"
echo "============================================"
echo "Training samples: 385"
echo "Test samples: 97"
echo "Starting from: MidAir checkpoint (Phase 1)"
echo "============================================"

python3 train_usegeo.py \
    --epochs=30 \
    --batch_size=2 \
    --lr=1e-5 \
    --from_midair_ckpt \
    --midair_ckpt_dir=./checkpoints \
    --output_dir=./checkpoints_usegeo \
    --log_dir=./checkpoints_usegeo/summaries \
    "$@"

echo ""
echo "============================================"
echo "Training complete!"
echo "Checkpoints saved to: ./checkpoints_usegeo"
echo "============================================"
