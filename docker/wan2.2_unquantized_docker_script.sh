#!/bin/bash
# Test script for Wan 2.2 unquantized with 10 and 20 steps
# This script should be run inside the Docker container

set -e

# Configuration
# Try multiple possible model paths
if [ -d "/workspace/docker/models/Wan2.2-T2V-A14B" ]; then
    MODEL_PATH="/workspace/docker/models/Wan2.2-T2V-A14B"
elif [ -d "/workspace/docker/models/wan2.2-14b" ]; then
    MODEL_PATH="/workspace/docker/models/wan2.2-14b"
elif [ -d "/workspace/docker/Wan2.2-T2V-A14B" ]; then
    MODEL_PATH="/workspace/docker/Wan2.2-T2V-A14B"
else
    MODEL_PATH="/workspace/docker/models/Wan2.2-T2V-A14B"  # Default, will check later
fi
OUTPUT_DIR="/workspace/results"
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
SIZE="480*832"
WAN2_2_PATH="/workspace/wan2.2"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Wan 2.2 Unquantized Benchmark Test"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Prompt: $PROMPT"
echo "Size: $SIZE"
echo ""

# Install required Python packages
echo "[INFO] Installing required Python packages..."
pip install easydict imageio ftfy decord opencv-python einops librosa soundfile av peft dashscope --quiet

# Install Flash Attention for ROCm
echo "[INFO] Installing Flash Attention for ROCm..."
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
if [ ! -d "/workspace/flash-attention" ]; then
    git clone https://github.com/ROCm/flash-attention.git /workspace/flash-attention
    cd /workspace/flash-attention
    git checkout main_perf
    pip install . --quiet
    cd /workspace
fi

# Check if wan2.2 directory exists
if [ ! -d "$WAN2_2_PATH" ]; then
    echo "[ERROR] Wan2.2 directory not found at $WAN2_2_PATH"
    echo "[INFO] Cloning Wan2.2 repository..."
    git clone https://github.com/Wan-Video/Wan2.2.git "$WAN2_2_PATH"
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARNING] Model path $MODEL_PATH does not exist"
    echo "[INFO] Please ensure the model is mounted or downloaded"
fi

# Test 1: 10 steps
echo ""
echo "=========================================="
echo "Test 1: 10 steps"
echo "=========================================="
TIMESTAMP_10=$(date +%Y%m%d_%H%M%S)
VIDEO_FILE_10="$OUTPUT_DIR/wan22_10steps_${TIMESTAMP_10}.gif"

cd "$WAN2_2_PATH"
python3 generate.py \
    --task t2v-A14B \
    --size "$SIZE" \
    --sample_steps 10 \
    --ckpt_dir "$MODEL_PATH" \
    --offload_model False \
    --convert_model_dtype \
    --prompt "$PROMPT" \
    --save_file "$VIDEO_FILE_10"

if [ -f "$VIDEO_FILE_10" ]; then
    echo "[SUCCESS] 10-step test completed. Output: $VIDEO_FILE_10"
else
    echo "[ERROR] 10-step test failed. Output file not found."
    exit 1
fi

# Wait a bit between tests
echo "[INFO] Waiting 10 seconds before next test..."
sleep 10

# Test 2: 20 steps
echo ""
echo "=========================================="
echo "Test 2: 20 steps"
echo "=========================================="
TIMESTAMP_20=$(date +%Y%m%d_%H%M%S)
VIDEO_FILE_20="$OUTPUT_DIR/wan22_20steps_${TIMESTAMP_20}.gif"

cd "$WAN2_2_PATH"
python3 generate.py \
    --task t2v-A14B \
    --size "$SIZE" \
    --sample_steps 20 \
    --ckpt_dir "$MODEL_PATH" \
    --offload_model False \
    --convert_model_dtype \
    --prompt "$PROMPT" \
    --save_file "$VIDEO_FILE_20"

if [ -f "$VIDEO_FILE_20" ]; then
    echo "[SUCCESS] 20-step test completed. Output: $VIDEO_FILE_20"
else
    echo "[ERROR] 20-step test failed. Output file not found."
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "10-step output: $VIDEO_FILE_10"
echo "20-step output: $VIDEO_FILE_20"
echo ""


