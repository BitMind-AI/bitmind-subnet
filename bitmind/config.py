import os
import bittensor as bt

MAINNET_UID = 34


def validate_config_and_neuron_path(config):
    r"""Checks/validates the config namespace object."""
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    bt.logging.info(f"Logging path: {full_path}")
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)
    return config


def add_args(parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=34)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Neuron Name",
        default="bitmind",
    )

    parser.add_argument(
        "--epoch-length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=360,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode",
        default=False,
    )

    parser.add_argument(
        "--autoupdate-off",
        action="store_false",
        dest="autoupdate",
        help="Disable automatic updates on latest version on Main.",
        default=True,
    )

    parser.add_argument("--wandb.entity", type=str, default="bitmindai")

    parser.add_argument("--wandb.off", action="store_true", default=False)


def add_miner_args(parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--no-force-validator-permit",
        action="store_true",
        help="If set, we will not force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for detection models (cuda/cpu)",
    )


def add_validator_args(parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--vpermit-tao-limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=20000,
    )

    parser.add_argument(
        "--compressed-cache-update-interval",
        type=int,
        help="How often to download new zip/parquet files, measured in 12 second blocks",
        default=720,
    )

    parser.add_argument(
        "--media-cache-update-interval",
        type=int,
        help="How often to unpack random media files, measured in 12 second blocks",
        default=60,
    )

    parser.add_argument(
        "--challenge-interval",
        type=int,
        help="How often we set challenge miners, measured in 12 second blocks.",
        default=5,
    )

    parser.add_argument(
        "--wandb-restart-interval",
        type=int,
        help="How often we restart wandb run to avoid log truncation",
        default=2000,
    )

    parser.add_argument(
        "--cache.base-dir",
        type=str,
        default=os.path.expanduser("~/.cache/sn34"),
        help="Base directory for cache storage",
    )

    parser.add_argument(
        "--cache.max-compressed-gb",
        type=float,
        default=50.0,
        help="Maximum size in GB for compressed cache",
    )

    parser.add_argument(
        "--cache.max-media-gb",
        type=float,
        default=5.0,
        help="Maximum size in GB for media cache",
    )

    parser.add_argument(
        "--cache.media-files-per-source",
        type=int,
        default=50,
        help="Number of media files to keep per source",
    )

    parser.add_argument(
        "--neuron.max-state-backup-hours",
        type=float,
        help="The oldest backup of validator state to load in the case of a failure to load most recent",
        default=1,
    )

    parser.add_argument(
        "--neuron.miner-total-timeout",
        type=float,
        help="Total timeout for miner requests in seconds",
        default=11.0,
    )

    parser.add_argument(
        "--neuron.miner-connect-timeout",
        type=float,
        help="TCP connection timeout for miner requests in seconds",
        default=4.0,
    )

    parser.add_argument(
        "--neuron.miner-sock-connect-timeout",
        type=float,
        help="Socket connection timeout for miner requests in seconds",
        default=3.0,
    )

    parser.add_argument(
        "--neuron.heartbeat",
        action="store_true",
        help="Run validator heartbeat thread",
        default=False,
    )

    parser.add_argument(
        "--neuron.heartbeat-interval-seconds",
        type=float,
        help="Interval between heartbeat checks in seconds",
        default=60.0,
    )

    parser.add_argument(
        "--neuron.lock-sleep-seconds",
        type=float,
        help="Sleep duration when lock is held in seconds",
        default=5.0,
    )

    parser.add_argument(
        "--neuron.max-stuck-count",
        type=int,
        help="Number of consecutive heartbeats with no progress before restart",
        default=5,
    )

    parser.add_argument(
        "--neuron.sample-size",
        type=int,
        help="Number of miners to query per challenge",
        default=50,
    )

    parser.add_argument(
        "--scoring.moving-average-alpha",
        type=float,
        help="Alpha for miner score EMA",
        default=0.05,
    )

    parser.add_argument(
        "--scoring.image-weight",
        type=float,
        help="Weight for image modality scoring",
        default=0.6,
    )

    parser.add_argument(
        "--scoring.video-weight",
        type=float,
        help="Weight for video modality scoring",
        default=0.4,
    )

    parser.add_argument(
        "--scoring.binary-weight",
        type=float,
        help="Weight for binary classification scoring",
        default=0.75,
    )

    parser.add_argument(
        "--scoring.multiclass-weight",
        type=float,
        help="Weight for multiclass classification scoring",
        default=0.25,
    )

    parser.add_argument(
        "--challenge.image-prob",
        type=float,
        help="Probability of selecting image modality for challenges",
        default=0.5,
    )

    parser.add_argument(
        "--challenge.video-prob",
        type=float,
        help="Probability of selecting video modality for challenges",
        default=0.5,
    )

    parser.add_argument(
        "--challenge.real-prob",
        type=float,
        help="Probability of selecting real media for challenges",
        default=0.5,
    )

    parser.add_argument(
        "--challenge.synthetic-prob",
        type=float,
        help="Probability of selecting synthetic media for challenges",
        default=0.3,
    )

    parser.add_argument(
        "--challenge.semisynthetic-prob",
        type=float,
        help="Probability of selecting semisynthetic media for challenges",
        default=0.2,
    )

    parser.add_argument(
        "--challenge.multi-video-prob",
        type=float,
        help="Probability of stitching together two videos of the same media type",
        default=0.2,
    )

    parser.add_argument(
        "--challenge.min-clip-duration",
        type=float,
        help="Minimum video clip duration in seconds",
        default=1.0,
    )

    parser.add_argument(
        "--challenge.max-clip-duration",
        type=float,
        help="Maximum video clip duration in seconds",
        default=6.0,
    )

    parser.add_argument(
        "--challenge.max-frames",
        type=int,
        help="Maximum number of video frames to sample for a challenge",
        default=144,
    )


def add_data_generator_args(parser):
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/sn34"),
        help="Directory for caching data",
    )

    parser.add_argument(
        "--batch-size", type=int, default=3, help="Batch size for generation"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["t2v", "t2i", "i2i", "i2v"],
        default=["t2v", "t2i", "i2i", "i2v"],
        help="List of tasks to run (t2v, t2i, i2i, i2v). Defaults to all.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for generation (cuda/cpu)",
    )

    parser.add_argument(
        "--wandb.num-batches-per-run",
        type=int,
        default=50,
        help="Number of batches to generate before starting new W&B run (avoids log truncation)",
    )

    parser.add_argument("--wandb.process-name", type=str, default="generator")


def add_proxy_args(parser):
    parser.add_argument(
        "--proxy.sample-size",
        type=int,
        default=50,
        help="Number of miners to query for organics",
    )

    parser.add_argument(
        "--proxy.client-url",
        type=str,
        default="https://subnet-api.bitmindlabs.ai",
        help="URL for the proxy client authentication service",
    )

    parser.add_argument(
        "--proxy.host",
        type=str,
        default="0.0.0.0",
        help="Network interface to listen on",
    )

    parser.add_argument(
        "--proxy.port",
        type=int,
        default=10913,
        help="Port for the proxy server",
    )

    parser.add_argument(
        "--proxy.external_port",
        type=int,
        default=10913,
        help="Port for the proxy server",
    )
