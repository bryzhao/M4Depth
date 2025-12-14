#!/bin/bash
# UseGeo visualization script
# Generates side-by-side comparisons of RGB, predicted depth, ground truth, and error

cd /home/bryan/dev/final_project/M4Depth
source /home/bryan/dev/final_project/m4depth_env/bin/activate

NVIDIA_BASE="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages/nvidia"

export PYTHONPATH="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cuda_nvrtc/lib:${NVIDIA_BASE}/cuda_runtime/lib:$LD_LIBRARY_PATH"
export PATH="${NVIDIA_BASE}/cuda_nvcc/bin:$PATH"

# Default: visualize with fine-tuned model on test data
CKPT_DIR="${1:-./checkpoints_usegeo}"
OUTPUT_DIR="${2:-visualizations_usegeo}"
NUM_SAMPLES="${3:-10}"

echo "============================================"
echo "UseGeo Visualization"
echo "============================================"
echo "Checkpoint: $CKPT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "============================================"

python3 visualize_usegeo.py \
    --ckpt_dir="$CKPT_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --num_samples="$NUM_SAMPLES" \
    --stride=5

echo ""
echo "============================================"
echo "Visualizations saved to: $OUTPUT_DIR/"
echo "============================================"
