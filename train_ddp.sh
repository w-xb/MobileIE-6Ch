#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train_ddp.sh [gpu_list] [num_gpus]
#
# Examples:
#   bash train_ddp.sh "0" 1
#   bash train_ddp.sh "0,1" 2
#   bash train_ddp.sh "0,1,2,3" 4

GPU_LIST="${1:-0,1,2,3}"
NUM_GPUS="${2:-4}"

echo "Using GPUs: ${GPU_LIST}"
echo "Number of processes: ${NUM_GPUS}"
echo "Model: MobileIE-6Ch"
echo "Config: config/lle.yaml"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate mobileie
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate mobileie
else
    echo "Conda was not found. Please activate the mobileie environment before running this script."
fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    main_ddp.py \
    -task train \
    -model_task lle \
    -use_6channel \
    -device cuda
