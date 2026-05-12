__version__ = "4.7.12"

import bittensor as bt

# Bittensor v10 exposes class names (Wallet/Subtensor/Axon/Metagraph) where
# earlier SDKs exposed lowercase constructors. Keep the existing codepath
# working while the subnet migrates to v10 for mechid support and security
# fixes.
if not hasattr(bt, "wallet") and hasattr(bt, "Wallet"):
    bt.wallet = bt.Wallet
if not hasattr(bt, "subtensor") and hasattr(bt, "Subtensor"):
    bt.subtensor = bt.Subtensor
if not hasattr(bt, "axon") and hasattr(bt, "Axon"):
    bt.axon = bt.Axon
if not hasattr(bt, "metagraph") and hasattr(bt, "Metagraph"):
    bt.metagraph = bt.Metagraph

version_split = __version__.split(".")
__spec_version__ = (
    (100000 * int(version_split[0]))
    + (1000 * int(version_split[1]))
    + (10 * int(version_split[2]))
)
