#!/bin/bash

# Example usage: ./run_dataset_generation.sh key_here

# Dataset and model configuration
REAL_DATASET='MS-COCO-unique'
DIFFUSION_MODEL='stabilityai/stable-diffusion-xl-base-1.0-inpainting'

# Total indices range
START_INDEX=37630
END_INDEX=75260

# Number of GPUs
NUM_GPUS=10

# Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

# Calculate the number of indices per GPU (add 1 for inclusive range)
RANGE_PER_GPU=$(( ($END_INDEX - $START_INDEX + 1) / $NUM_GPUS ))

# Loop to create tasks for each GPU
for (( i=0; i<$NUM_GPUS; i++ )); do
    gpu_start_index=$(( START_INDEX + i * RANGE_PER_GPU ))
    if [[ $i -eq $(( NUM_GPUS - 1 )) ]]; then
        # Last GPU takes any remainder
        gpu_end_index=$END_INDEX
    else
        gpu_end_index=$(( gpu_start_index + RANGE_PER_GPU - 1 ))
    fi

    # Run the command with calculated indices
    pm2 start generate_synthetic_image_dataset.py --name "$REAL_DATASET $DIFFUSION_MODEL i2i $i" --no-autorestart -- \
        --hf_org 'bitmind' \
        --real_image_dataset_name "$REAL_DATASET" \
        --diffusion_model "$DIFFUSION_MODEL" \
        --download_annotations \
        --generate_synthetic_images \
        --upload_synthetic_images \
        --i2i \
        --hf_token "$HF_TOKEN" \
        --start_index $gpu_start_index \
        --end_index $gpu_end_index \
        --gpu_id $i
done