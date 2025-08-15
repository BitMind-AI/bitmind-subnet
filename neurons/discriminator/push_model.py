#!/usr/bin/env python3
"""
Script to push a zip of multiple ONNX models to Hugging Face and register it on the Bittensor blockchain.

Usage:
    python push_model.py --onnx-dir "path/to/models/directory" [options]
    python push_model.py --model-zip "path/to/models.zip" [options]

Options:
    --wallet-name NAME         Bittensor wallet name (default: default)
    --wallet-hotkey KEY       Bittensor hotkey name (default: default)
    --netuid UID             Subnet UID (default: 34)
    --subtensor-chain-endpoint URL  Subtensor network endpoint
    --retry-delay SECS       Retry delay in seconds (default: 60)

Example:
    python push_model.py --onnx-dir "models/" --wallet-name miner1
    python push_model.py --model-zip "models.zip" --wallet-name miner1
"""
import argparse
import asyncio
import hashlib
import os
import sys
import traceback
import time
from typing import Optional

import bittensor as bt

# Try to import colorama for colored output, fallback to no colors if not available
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORS = True
except ImportError:
    # Fallback colors that do nothing
    class Fore:
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        CYAN = ""
        WHITE = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""
    HAS_COLORS = False

from gas.types import DiscriminatorModelId as ModelId
from gas.utils.chain_model_metadata_store import ChainModelMetadataStore
from gas.protocol.model_uploads import upload_model_zip_presigned
from gas.utils.model_zips import validate_onnx_directory, create_model_zip


MODEL_UPLOAD_ENDPOINT = "https://onnx-models-worker.bitmind.workers.dev/upload"


def print_success(message: str):
    """Print a success message in green"""
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print an error message in red"""
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")


def print_warning(message: str):
    """Print a warning message in yellow"""
    print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")


def print_info(message: str):
    """Print an info message in blue"""
    print(f"{Fore.BLUE}⛽ {message}{Style.RESET_ALL}")


def print_step(step_num: int, total_steps: int, message: str):
    """Print a step message with consistent formatting"""
    print(f"{Fore.CYAN}[{step_num}/{total_steps}] {message}{Style.RESET_ALL}")


def get_hash_of_two_strings(str1: str, str2: str) -> str:
    return str(hash(str1 + str2))  # Simplified hash function


def compute_zip_hash(zip_path: str) -> str:
    """Compute SHA256 hash of a zip file"""
    hash_sha256 = hashlib.sha256()

    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


async def push_model_zip(
    onnx_dir: Optional[str] = None,
    zip_path: Optional[str] = None,
    wallet: bt.wallet = None,
    retry_delay_secs: int = 60,
    netuid: int = 34,
    chain_endpoint: Optional[str] = None,
):
    """Pushes a zip of multiple ONNX models and their configuration to Hugging Face and
    registers it on the Bittensor blockchain."""

    # Step 0: Prepare model zip
    if zip_path is not None:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Model zip file not found: {zip_path}")
        print_info(f"Using provided model zip: {zip_path}")

    elif onnx_dir is not None:
        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

        print_info(f"Creating model zip from directory: {onnx_dir}")
        zip_path = create_model_zip(onnx_dir)
        print_success(f"Created model zip: {zip_path}")
    else:
        raise ValueError("Either --onnx-dir or --model-zip must be provided")

    # Step 1: Upload to cloud inference system
    print()
    print_step(1, 2, " Uploading model zip to cloud inference system...")
    response = upload_model_zip_presigned(wallet, zip_path, MODEL_UPLOAD_ENDPOINT)

    if not response["success"]:
        print_error("Failed to upload model to cloud inference system")
        print_error(f"Failed at step: {response.get('step', 'unknown')}")
        print_error(f"Error: {response.get('error', 'Unknown error')}")
        if 'model_id' in response:
            print_info(f"Model ID: {response['model_id']}")
        return False
    
    print_success("Model zip uploaded successfully!")

    # Step 2: Register on Blockchain
    print_step(2, 2, "Registering model metadata on blockchain...")
    
    if not chain_endpoint:
        chain_endpoint = "finney" if netuid == 34 else "test"

    print_info(f"Connecting to subnet {netuid} via {chain_endpoint}")
    subtensor = bt.subtensor(network=chain_endpoint)
    metadata_store = ChainModelMetadataStore(subtensor, netuid)

    model_key = response.get("r2_key", "")
    if not model_key:
        print_error("Model key not provided in upload response")
        return False

    hash = get_hash_of_two_strings(
        compute_zip_hash(zip_path), wallet.hotkey.ss58_address
    )

    model_id = ModelId(key=model_key, hash=hash)
    print_info(f"Model ID: {model_id}")

    while True:
        try:
            print_info("Storing model metadata on chain...")
            await metadata_store.store_model_metadata(wallet, model_id)
            
            print_info("Verifying metadata on chain...")
            uid = subtensor.get_uid_for_hotkey_on_subnet(
                wallet.hotkey.ss58_address, netuid
            )
            model_metadata = await metadata_store.retrieve_model_metadata(
                uid, wallet.hotkey.ss58_address
            )

            if (
                not model_metadata
                or model_metadata.id.to_compressed_str() != model_id.to_compressed_str()
            ):
                print_error(f"Metadata verification failed")
                print_error(f"Expected: {model_id}")
                print_error(f"Got: {model_metadata}")
                raise ValueError("Metadata verification failed")

            print_success("Model metadata registered on blockchain successfully!")
            return True
            
        except Exception as e:
            print_error(f"Failed to register on blockchain: {e}")
            print_warning(f"Retrying in {retry_delay_secs} seconds...")
            time.sleep(retry_delay_secs)


def main():
    parser = argparse.ArgumentParser(
        description="Push a zip of ONNX models to Hugging Face and register on Bittensor"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--onnx-dir",
        help="Path to directory containing ONNX files (image_detector.onnx, video_detector.onnx)",
    )
    group.add_argument(
        "--model-zip",
        help="Path to pre-existing model zip file",
    )
    parser.add_argument(
        "--wallet-name",
        default="default",
        help="Bittensor wallet name"
    )
    parser.add_argument(
        "--wallet-hotkey",
        default="default",
        help="Bittensor hotkey name"
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=34,
        help="Subnet UID"
    )
    parser.add_argument(
        "--chain-endpoint",
        default="",
        help="Subtensor network",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="Retry delay in seconds"
    )

    args = parser.parse_args()

    print()
    print(f"{Fore.CYAN}{Style.BRIGHT}=== Model Push Configuration ==={Style.RESET_ALL}")

    if args.onnx_dir:
        print(f"ONNX Directory: {args.onnx_dir}")
    else:
        print(f"Model Zip: {args.model_zip}")
    print(f"Wallet: {args.wallet_name}/{args.wallet_hotkey}")
    print(f"Subnet UID: {args.netuid}")
    print(f"Chain Endpoint: {args.chain_endpoint}")
    print()

    # Validate ONNX directory if provided
    if args.onnx_dir and not validate_onnx_directory(args.onnx_dir):
        print_error("ONNX directory validation failed")
        sys.exit(1)

    print()
    print(f"{Fore.CYAN}{Style.BRIGHT}=== Starting Model Push ==={Style.RESET_ALL}")

   # Initialize wallet
    print_info(f"Initializing wallet: {args.wallet_name}/{args.wallet_hotkey}")
    try:
        wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        print_info(f"Hotkey: {wallet.hotkey.ss58_address}")
    except Exception as e:
        print_error(f"Failed to initialize wallet: {e}")
        sys.exit(1)

    # Check if hotkey is registered
    netuid = args.netuid
    mg = bt.metagraph(netuid=netuid, network="finney" if netuid == 34 else "test")
    hotkey = wallet.hotkey.ss58_address
    if hotkey not in mg.hotkeys:
        print_warning(f"Hotkey {hotkey} not registered on netuid {netuid}")
        print_warning("This may cause issues with model registration")

    try:
        success = asyncio.run(
            push_model_zip(
                onnx_dir=args.onnx_dir,
                zip_path=args.model_zip,
                wallet=wallet,
                retry_delay_secs=args.retry_delay,
                netuid=netuid,
                chain_endpoint=args.chain_endpoint
            )
        )
        
        print()
        if success:
            print_success("🎉 Model push completed successfully!")
        else:
            print_error("💥 Model push failed!")
            sys.exit(1)

    except Exception as e:
        print()
        print_error(f"💥 Model push failed with error: {e}")
        if args.onnx_dir or args.model_zip:
            print_info("Full traceback:")
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
