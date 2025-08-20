# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs
# Copyright © 2025 BitMind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import signal
import time
import os
import requests
import subprocess
import json
import bittensor as bt
import gas


def get_running_sn34_apps_via_pm2():
    """
    Query PM2 for running processes and return names starting with 'sn34-'.
    """
    try:
        result = subprocess.run([
            "pm2", "jlist"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            bt.logging.warning(f"pm2 jlist failed: {result.stderr}")
            return []
        apps = json.loads(result.stdout)
        if not isinstance(apps, list):
            return []
        return [proc.get('name') for proc in apps if isinstance(proc, dict) and isinstance(proc.get('name'), str) and proc.get('name').startswith('sn34-')]
    except Exception as e:
        bt.logging.warning(f"Failed to parse pm2 jlist: {e}")
        return []


def restart_pm2_services(base_path):
    """
    Restart running sn34-* PM2 services only. Does not restart other processes.
    """
    app_names = get_running_sn34_apps_via_pm2()

    if not app_names:
        bt.logging.warning("No sn34-* apps found; skipping PM2 restarts to avoid affecting other processes")
        return False

    # Restart each service individually
    success = True
    for app_name in app_names:
        try:
            bt.logging.info(f"Restarting {app_name}...")
            subprocess.run(["pm2", "restart", app_name], check=True)
            bt.logging.info(f"Successfully restarted {app_name}")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Failed to restart {app_name}: {e}")
            success = False

    return success


def autoupdate(branch: str = "main", force=False):
    """
    Automatically updates the codebase to the latest version available on the specified branch.

    This function checks the remote repository for the latest version by fetching the VERSION file from the specified branch.
    If the local version is older than the remote version, it performs a git pull to update the local codebase to the latest version.
    After successfully updating, it restarts the application with the updated code.

    Args:
    - branch (str): The name of the branch to check for updates. Defaults to "main".
    - force (bool): Force update even if versions are the same. Defaults to False.

    Note:
    - The function assumes that the local codebase is a git repository and has the same structure as the remote repository.
    - It requires git to be installed and accessible from the command line.
    - The function will restart the application using PM2 services defined in validator.config.js.
    - If the update fails, manual intervention is required to resolve the issue and restart the application.
    """
    bt.logging.info("Checking for updates...")
    try:
        github_url = f"https://raw.githubusercontent.com/BitMind-AI/bitmind-subnet/{branch}/VERSION?ts={time.time()}"
        response = requests.get(
            github_url,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            },
        )
        response.raise_for_status()
        repo_version = response.content.decode().strip()
        latest_version = tuple(map(int, repo_version.split(".")))
        local_version = tuple(map(int, gas.__version__.split(".")))

        bt.logging.info(f"Local version: {local_version}")
        bt.logging.info(f"Latest version: {latest_version}")

        if latest_version > local_version or force:
            bt.logging.info(f"A newer version is available. Updating...")
            base_path = os.path.abspath(__file__)
            while os.path.basename(base_path) != "bitmind-subnet":
                base_path = os.path.dirname(base_path)

            os.system(f"cd {base_path} && git pull")

            with open(os.path.join(base_path, "VERSION")) as f:
                new_version = f.read().strip()
                new_version = tuple(map(int, new_version.split(".")))

                if new_version == latest_version:
                    bt.logging.info("Updated successfully.")

                    # Restart PM2 services using validator.config.js (with robust fallbacks)
                    if restart_pm2_services(base_path):
                        bt.logging.info("All PM2 services restarted successfully.")
                        bt.logging.info(f"Restarting validator")
                        os.kill(os.getpid(), signal.SIGINT)
                    else:
                        bt.logging.error("Failed to restart some PM2 services. Manual restart may be required.")
                else:
                    bt.logging.error("Update failed. Manual update required.")
    except Exception as e:
        bt.logging.error(f"Update check failed: {e}")