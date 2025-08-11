#!/usr/bin/env python3
"""
Script to push a zip of multiple ONNX models to Hugging Face and register it on the Bittensor blockchain.

Usage:
    python push_model.py --onnx-dir "path/to/models/directory" [options]
    python push_model.py --model-zip "path/to/models.zip" [options]

Options:
    --wallet-name NAME         Bittensor wallet name (default: default)
    --wallet-hotkey KEY       Bittensor hotkey name (default: default)
    --netuid UID             Subnet UID (default: 379)
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

from gas.types import DiscriminatorModelId as ModelId
from gas.utils.chain_model_metadata_store import ChainModelMetadataStore
from gas.protocol.model_uploads import upload_model_zip_presigned
from gas.utils.model_zips import validate_onnx_directory, create_model_zip


MODEL_UPLOAD_ENDPOINT = "https://onnx-models-worker.bitmind.workers.dev/upload"


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

    if zip_path is not None:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Model zip file not found: {zip_path}")
        print(f"Using provided model zip: {zip_path}")

    elif onnx_dir is not None:
        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

        zip_path = create_model_zip(onnx_dir)
        print(f"Created model zip from directory: {onnx_dir}")
    else:
        raise ValueError("Either --onnx-dir or --model-zip must be provided")

    ### Upload to cloud inference system
    bt.logging.debug("Started uploading model zip...")
    response = upload_model_zip_presigned(wallet, zip_path, MODEL_UPLOAD_ENDPOINT)

    if not response["success"]:
        print(f"Failed to upload model to {MODEL_UPLOAD_ENDPOINT}")
        print(f"Failed at step: {response.get('step', 'unknown')}")
        print(f"Error: {response.get('error', 'Unknown error')}")
        if 'model_id' in response:
            print(f"Model ID: {response['model_id']}")
        print(f"Response: {response.get('response', 'Unknown')}")
        return False
    else:
        print(
            "✅ (Step 1/2) Model zip pushed successfully! Writing metadata to blockchain"
        )

    ### Register on Blockchain
    if chain_endpoint is None:
        chain_endpoint = "finney" if netuid == 34 else "test"

    print(f"Instantiating ChainModelMetadataStore on netuid {netuid} with endpoint {chain_endpoint}")
    subtensor = bt.subtensor(network=chain_endpoint)
    metadata_store = ChainModelMetadataStore(subtensor, netuid)

    model_key = response.get("r2_key", "")
    if not model_key:
        print(f"Key not provided in response")
        return False

    hash = get_hash_of_two_strings(
        compute_zip_hash(zip_path), wallet.hotkey.ss58_address
    )

    model_id = ModelId(key=model_key, hash=hash)
    print(f"Committing to chain with model_id: {model_id}")

    while True:
        try:
            await metadata_store.store_model_metadata(wallet, model_id)
            print("Wrote model metadata to chain. Verifying...")

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
                print(
                    f"Metadata verification failed. Expected: {model_id}, got: {model_metadata}"
                )
                raise ValueError("Metadata verification failed")

            print("✅ (Step 2/2) Committed model metadata to chain.")
            break
        except Exception as e:
            print(f"Failed to commit to chain: {e}")
            print(f"Retrying in {retry_delay_secs} seconds...")
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
        default=379,
        help="Subnet UID"
    )
    parser.add_argument(
        "--chain-endpoint",
        default="wss://test.finney.opentensor.ai:443/",
        help="Subtensor network",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="Retry delay in seconds"
    )

    args = parser.parse_args()
    print(args)

    # Validate ONNX directory if provided
    if args.onnx_dir and not validate_onnx_directory(args.onnx_dir):
        sys.exit(1)

    # Initialize wallet
    print(f"Initializing wallet: {args.wallet_name}/{args.wallet_hotkey}")
    try:
        wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        print(f"Hotkey: {wallet.hotkey.ss58_address}")
    except Exception as e:
        print(f"Error initializing wallet: {e}")
        sys.exit(1)

    netuid = args.netuid
    mg = bt.metagraph(netuid=netuid, network="finney" if netuid == 34 else "test")
    hotkey = wallet.hotkey.ss58_address
    if hotkey not in mg.hotkeys:
        bt.logging.error(f"❌ Hotkey {hotkey} not registered on netuid {netuid}")

    try:
        asyncio.run(
            push_model_zip(
                onnx_dir=args.onnx_dir,
                zip_path=args.model_zip,
                wallet=wallet,
                retry_delay_secs=args.retry_delay,
                netuid=netuid,
                chain_endpoint=args.chain_endpoint
            )
        )

    except Exception as e:
        print(f"Error pushing model zip: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
