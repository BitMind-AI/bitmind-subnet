#!/bin/bash
pkill -f "train_detector.py --ddp"
pkill -f "torch.distributed.launch"

# force kill for stubborn processes
sleep 1
pkill -9 -f "train_detector.py --ddp"
pkill -9 -f "torch.distributed.launch"
