#!/bin/bash
nohup python3 -m torch.distributed.launch --nproc_per_node=2 train_detector.py --ddp > ucf_ddp.log 2>&1 &
