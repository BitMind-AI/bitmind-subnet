#!/bin/bash

# --input_dir is a directory of mp4 files
# --frames_dir is where the extracted png frames will be stored
# --dataset_dir is where the huggingface dataset (containing paths to frames) will be stored
# once your dataset is created, you can add its local path to base_miner/config.py for training
python base_miner/datasets/create_video_dataset.py --input_dir ~/.cache/sn34/real/video \
       --frames_dir ~/.cache/sn34/train_frames \
       --dataset_dir ~/.cache/sn34/train_dataset/real_frames \
       --num_videos 500 \
       --frame_rate 5 \
       --max_frames 24 \
       --dataset_name real_frames \
       --overwrite 
