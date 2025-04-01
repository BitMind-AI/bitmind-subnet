#!/usr/bin/env python3
"""W&B Cache Cleaning Script"""
import os
import sys
import glob
import shutil
import time
import argparse
from datetime import datetime

def clean_wandb_cache(wandb_dir, hours=1):
    """Cleans wandb runs except recent ones and latest-run."""
    if not os.path.exists(wandb_dir):
        print(f"W&B directory not found: {wandb_dir}")
        return
    
    run_dirs = [d for d in glob.glob(os.path.join(wandb_dir, "run-*")) if os.path.isdir(d)]
    
    if not run_dirs:
        print("No W&B runs found.")
        return
    
    # Keep recent runs
    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    recent_runs = [d for d in run_dirs if os.path.getmtime(d) > cutoff_time]
    
    # Preserve latest-run target
    latest_run_link = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_run_link) and os.path.isdir(latest_run_link):
        try:
            latest_run_target = os.path.realpath(latest_run_link)
            if latest_run_target not in recent_runs and latest_run_target in run_dirs:
                recent_runs.append(latest_run_target)
                print(f"Preserving latest-run: {os.path.basename(latest_run_target)}")
        except Exception as e:
            print(f"Error with latest-run: {e}")
    
    print(f"Keeping {len(recent_runs)} runs (modified in last {hours} hours):")
    for run in recent_runs:
        print(f"  - {os.path.basename(run)}")
    
    # Remove old runs
    runs_removed = 0
    space_freed = 0
    
    for run_dir in run_dirs:
        if run_dir not in recent_runs:
            try:
                dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                              for dirpath, _, filenames in os.walk(run_dir)
                              for filename in filenames)
                space_freed += dir_size
                shutil.rmtree(run_dir)
                runs_removed += 1
            except Exception as e:
                print(f"Error removing {run_dir}: {e}")
    
    print(f"Cleaned {runs_removed} runs, freed {space_freed / (1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean W&B cache directory")
    parser.add_argument("--dir", default="./wandb", help="W&B directory path (default: ./wandb)")
    parser.add_argument("--hours", type=int, default=1, help="Keep runs newer than this many hours (default: 1)")
    args = parser.parse_args()
    
    clean_wandb_cache(args.dir, args.hours)
