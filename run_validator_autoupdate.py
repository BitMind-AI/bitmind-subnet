"""
Thank you to Namoray of SN19 for their autoupdate implementation!
"""
import os
import subprocess
import time

# Set the interval in hours to restart the PM2 process
RESTART_INTERVAL_HOURS = 6.5
PM2_PROCESS_NAME = "bitmind_validator"


def restart_pm2_process():
    print(f"Restarting PM2 process: {PM2_PROCESS_NAME}")
    os.system(f"pm2 delete {PM2_PROCESS_NAME}")
    os.system(f"pm2 start {PM2_PROCESS_NAME}")
    print(f"PM2 process {PM2_PROCESS_NAME} restarted successfully")


def should_update_local(local_commit, remote_commit):
    return local_commit != remote_commit


os.system("./start_mainnet_validator.sh")
time.sleep(10)


def run_auto_updater():
    last_restart_time = time.time()
    while True:
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
                os.system("./autoupdate_validator_steps.sh")
                time.sleep(20)

                print("Finished running the autoupdate steps! Ready to go 😎")

        else:
            print("Repo is up-to-date.")

        time.sleep(60)

        # Check if it's time to restart the PM2 process
        if time.time() - last_restart_time >= RESTART_INTERVAL_HOURS * 3600:
            restart_pm2_process()
            last_restart_time = time.time()  # Reset the timer after the restart

        time.sleep(60)


if __name__ == "__main__":
    run_auto_updater()