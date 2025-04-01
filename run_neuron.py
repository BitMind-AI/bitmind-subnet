"""
Thank you to Namoray of SN19 for their autoupdate implementation!
"""

import os
import sys
import subprocess
import time
import argparse
import glob
import shutil
from datetime import datetime

# self heal restart interval
RESTART_INTERVAL_HOURS = 3


def should_update_local(local_commit, remote_commit):
    return local_commit != remote_commit


def clean_wandb_cache_except_current():
    """
    Cleans all wandb runs from the cache except for the most recent ones.
    Specifically handles the 'latest-run' symlink and keeps any runs from the last hour.
    """
    wandb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb")

    if not os.path.exists(wandb_dir):
        print("W&B cache directory not found.")
        return

    run_dirs = [
        d for d in glob.glob(os.path.join(wandb_dir, "run-*")) if os.path.isdir(d)
    ]

    if not run_dirs:
        print("No W&B runs found in cache.")
        return

    one_hour_ago = time.time() - 3600
    recent_runs = []  # [d for d in run_dirs if os.path.getmtime(d) > one_hour_ago]

    # check if there's a latest-run symlink and make sure we preserve its target
    latest_run_link = os.path.join(wandb_dir, "latest-run")
    latest_run_target = None

    if os.path.exists(latest_run_link) and os.path.isdir(latest_run_link):
        try:
            latest_run_target = os.path.realpath(latest_run_link)
            if latest_run_target not in recent_runs and latest_run_target in run_dirs:
                recent_runs.append(latest_run_target)
                print(
                    f"Preserving latest-run target: {os.path.basename(latest_run_target)}"
                )
        except Exception as e:
            print(f"Error resolving latest-run symlink: {e}")

    print(f"Keeping {len(recent_runs)} recent W&B runs:")
    for run in recent_runs:
        run_time = datetime.fromtimestamp(os.path.getmtime(run))
        print(f"  - {os.path.basename(run)} (from {run_time})")

    # Remove all other run directories
    runs_removed = 0
    space_freed = 0

    for run_dir in run_dirs:
        if run_dir not in recent_runs:
            try:
                dir_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(run_dir)
                    for filename in filenames
                )
                space_freed += dir_size
                shutil.rmtree(run_dir)
                runs_removed += 1
            except Exception as e:
                print(f"Error removing {run_dir}: {e}")

    # Don't delete artifacts directory, just report on its size
    artifacts_dir = os.path.join(wandb_dir, "artifacts")
    if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
        try:
            artifacts_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(artifacts_dir)
                for filename in filenames
            )
            print(
                f"W&B artifacts directory size: {artifacts_size / (1024*1024):.2f} MB"
            )
        except Exception as e:
            print(f"Error calculating artifacts directory size: {e}")

    # Don't delete log files
    print(
        f"Cleaned {runs_removed} W&B runs, freed approximately {space_freed / (1024*1024):.2f} MB"
    )


def run_auto_update_self_heal(neuron_type, auto_update, self_heal, clean_wandb):
    if clean_wandb:
        print("Pruning wanbd cache")
        clean_wandb_cache_except_current()

    last_restart_time = time.time()
    while True:
        time.sleep(60)
        if auto_update:
            current_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
            local_commit = subprocess.getoutput("git rev-parse HEAD")
            os.system("git fetch")
            remote_commit = subprocess.getoutput(
                f"git rev-parse origin/{current_branch}"
            )
            if should_update_local(local_commit, remote_commit):
                print("Local repo is not up-to-date. Updating...")
                reset_cmd = "git reset --hard " + remote_commit
                process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                if error:
                    print("Error in updating:", error)
                else:
                    print(
                        "Updated local repo to latest version: {}",
                        format(remote_commit),
                    )

                    print("Running the autoupdate steps...")
                    # Trigger shell script. Make sure this file path starts from root
                    os.system(f"./autoupdate_{neuron_type}_steps.sh")
                    time.sleep(20)
                    print("Finished running the autoupdate steps 😎")
                    print("Restarting neuron")
                    os.system(f"./start_{neuron_type}.sh")
            else:
                print("Repo is up-to-date.")
        if self_heal:
            # Check if it's time to restart the PM2 process
            if time.time() - last_restart_time >= RESTART_INTERVAL_HOURS * 3600:
                if clean_wandb:
                    clean_wandb_cache_except_current()
                os.system(f"./start_{neuron_type}.sh")
                last_restart_time = time.time()  # Reset the timer after the restart


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bittensor neuron run script with optional self-healing and auto-update."
    )
    parser.add_argument("--validator", action="store_true")
    parser.add_argument("--miner", action="store_true")
    parser.add_argument(
        "--no-self-heal",
        action="store_true",
        help="Disable the automatic restart of the PM2 process",
    )
    parser.add_argument(
        "--no-auto-update",
        action="store_true",
        help="Disable the automatic update of the local repository",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache before starting validator",
    )
    parser.add_argument(
        "--no-clean-wandb", action="store_true", help="Disable cleaning the wandb cache"
    )
    args = parser.parse_args()

    if not (args.miner ^ args.validator):
        print(
            f"Usage: python {__file__}"
            + "--validator | --miner [--no-self-heal --no-auto-update --no-clean-wandb]"
        )
        sys.exit(1)

    neuron_type = "miner" if args.miner else "validator"

    if args.clear_cache and args.validator:
        os.system(f"./start_{neuron_type}.sh --clear-cache")
    else:
        os.system(f"./start_{neuron_type}.sh")

    if not args.no_auto_update or not args.no_self_heal:
        run_auto_update_self_heal(
            neuron_type,
            auto_update=not args.no_auto_update,
            self_heal=not args.no_self_heal,
            clean_wandb=not args.no_clean_wandb,
        )
