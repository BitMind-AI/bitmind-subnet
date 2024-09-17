#!/bin/bash

# Example usage: ./run_dataset_generation.sh key_here

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

# Calculate the number of indices per GPU
RANGE_PER_GPU=$(( ($END_INDEX - $START_INDEX) / $NUM_GPUS ))

# Loop to create tasks for each GPU
for (( i=0; i<$NUM_GPUS; i++ )); do
    gpu_start_index=$(( START_INDEX + i * RANGE_PER_GPU ))
    if [[ $i -eq $(( NUM_GPUS - 1 )) ]]; then
        # Ensure the last GPU takes any remainder
        gpu_end_index=$END_INDEX
    else
        gpu_end_index=$(( gpu_start_index + RANGE_PER_GPU - 1 ))
    fi

    # Run the command with calculated indices
    pm2 start generate_synthetic_dataset.py --name "mscoco $i" --no-autorestart -- \
        --hf_org 'bitmind' \
        --real_image_dataset_name 'MS-COCO-unique' \
        --diffusion_model 'stabilityai/stable-diffusion-xl-base-1.0' \
        --download_annotations \
        --generate_synthetic_images \
        --upload_synthetic_images \
        --hf_token "$HF_TOKEN" \
        --start_index $gpu_start_index \
        --end_index $gpu_end_index \
        --gpu_id $i
done