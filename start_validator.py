import argparse
import time
import subprocess


default_address = "wss://bittensor-finney.api.onfinality.io/public-ws"

def update_and_restart(pm2_name, wallet_name, wallet_hotkey, port, network, address):
    print("Updating to the latest version...")
    subprocess.run(["git", "reset", "--hard"])
    result = subprocess.run(["git", "pull"], capture_output=True, text=True)
    if "Already up to date." not in result.stdout:
        subprocess.run(["pip", "install", "-e", "."])
        subprocess.run(["python", "download_data.py"])

    vali_start_command = [
        "pm2", "start", "neurons/validator.py", "--interpreter", "python3", "--name", pm2_name, "--", "--wallet.name", wallet_name, "--wallet.hotkey", wallet_hotkey,
        "--netuid", "34", "--subtensor.network", network, "--axon.port", port
    ]
    if network == 'finney' or address != default_address:
        vali_start_command += ["--subtensor.chain_endpoint", address]

    print("Restarting validator...")
    print(vali_start_command)
    time.sleep(5)
    subprocess.run(["pm2", "delete", pm2_name])    
    subprocess.run(vali_start_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update and restart the validator process",
        epilog="Example usage: python start_validator.py --pm2_name 'net168vali' --wallet_name 'default' --wallet_hotkey 'default' [--address 'wss://...']"
    )

    parser.add_argument("--pm2_name", required=True, help="Name of the PM2 process.")
    parser.add_argument("--wallet_name", required=True, help="Name of the wallet.")
    parser.add_argument("--wallet_hotkey", required=True, help="Hotkey for the wallet.")
    parser.add_argument("--port", default='8091', help="axon.port")    
    parser.add_argument("--network", default="test", help="subtensor.network (finney/test/local)")
    parser.add_argument("--address", default=default_address, help="Subtensor chain_endpoint, defaults to 'wss://bittensor-finney.api.onfinality.io/public-ws' if not provided.")

    args = parser.parse_args()

    try:
        update_and_restart(args.pm2_name, args.wallet_name, args.wallet_hotkey, args.port, args.network, args.address)
    except Exception as e:
        parser.error(f"An error occurred: {e}")
