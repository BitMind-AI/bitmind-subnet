# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs
# Copyright © 2025 BitMind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import signal
import time
import os
import requests
import bittensor as bt
import bitmind


def autoupdate(branch: str = "main", force=False):
    """
    Automatically updates the codebase to the latest version available on the specified branch.

    This function checks the remote repository for the latest version by fetching the VERSION file from the specified branch.
    If the local version is older than the remote version, it performs a git pull to update the local codebase to the latest version.
    After successfully updating, it restarts the application with the updated code.

    Args:
    - branch (str): The name of the branch to check for updates. Defaults to "main".

    Note:
    - The function assumes that the local codebase is a git repository and has the same structure as the remote repository.
    - It requires git to be installed and accessible from the command line.
    - The function will restart the application using the same command-line arguments it was originally started with.
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
        repo_version = response.content.decode()
        latest_version = int("".join(repo_version.split(".")))
        local_version = int("".join(bitmind.__version__.split(".")))

        bt.logging.info(f"Local version: {bitmind.__version__}")
        bt.logging.info(f"Latest version: {repo_version}")

        if latest_version > local_version or force:
            bt.logging.info(f"A newer version is available. Updating...")
            base_path = os.path.abspath(__file__)
            while os.path.basename(base_path) != "bitmind-subnet":
                base_path = os.path.dirname(base_path)

            os.system(f"cd {base_path} && git pull && chmod +x setup.sh && ./setup.sh")

            with open(os.path.join(base_path, "VERSION")) as f:
                new_version = f.read().strip()
                new_version = int("".join(new_version.split(".")))

                if new_version == latest_version:
                    bt.logging.info("Updated successfully.")

                    bt.logging.info("Restarting generator...")
                    subprocess.run(["pm2", "reload", "sn34-generator"], check=True)

                    bt.logging.info("Restarting proxy...")
                    subprocess.run(["pm2", "reload", "sn34-proxy"], check=True)

                    bt.logging.info(f"Restarting validator")
                    os.kill(os.getpid(), signal.SIGINT)
                else:
                    bt.logging.error("Update failed. Manual update required.")
    except Exception as e:
        bt.logging.error(f"Update check failed: {e}")
