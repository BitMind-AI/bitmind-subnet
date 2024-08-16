"""
Thank you to Namoray of SN19 for their autoupdate implementation!
"""
import os
import subprocess
import time
import argparse

# self heal restart interval
RESTART_INTERVAL_HOURS = 6


def should_update_local(local_commit, remote_commit):
    return local_commit != remote_commit


def run_auto_update_self_heal(neuron_type, network, auto_update, self_heal):
    last_restart_time = time.time()

    while True:
        time.sleep(60)

        if auto_update:
            current_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
            local_commit = subprocess.getoutput("git rev-parse HEAD")
            os.system("git fetch")
            remote_commit = subprocess.getoutput(f"git rev-parse origin/{current_branch}")

            if should_update_local(local_commit, remote_commit):
                print("Local repo is not up-to-date. Updating...")
                reset_cmd = "git reset --hard " + remote_commit
                process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                if error:
                    print("Error in updating:", error)
                else:
                    print("Updated local repo to latest version: {}", format(remote_commit))
                    
                    print("Running the autoupdate steps...")
                    # Trigger shell script. Make sure this file path starts from root
                    os.system(f"./autoupdate_{neuron_type}_steps.sh")
                    time.sleep(20)
                    print("Finished running the autoupdate steps 😎")
                    print("Restarting neuron")
                    os.system(f"./start_{network}_{neuron_type}.sh")
            else:
                print("Repo is up-to-date.")

        if self_heal:
            # Check if it's time to restart the PM2 process
            if time.time() - last_restart_time >= RESTART_INTERVAL_HOURS * 3600:
                os.system(f"./start_{network}_{neuron_type}.sh")
                last_restart_time = time.time()  # Reset the timer after the restart


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bittensor neuron run script with optional self-healing and auto-update.")
    parser.add_argument("--neuron-type", required=True, help="validator or miner")
    parser.add_argument("--network", required=True, help="finney or test")
    parser.add_argument("--no-self-heal", action="store_true", help="Disable the automatic restart of the PM2 process")
    parser.add_argument("--no-auto-update", action="store_true", help="Disable the automatic update of the local repository")

    args = parser.parse_args()
    if args.neuron_type not in ('validator', 'miner'):
        print(f"Usage: python {__file__} --neuron-type 'validator' or 'miner' [--no-self-heal --no-auto-update]")

    os.system(f"./start_{args.network}_{args.neuron_type}.sh")

    if not args.no_auto_update or not args.no_self_heal:
        run_auto_update_self_heal(
            args.neuron_type,
            args.network,
            auto_update=not args.no_auto_update,
            self_heal=not args.no_self_heal)
