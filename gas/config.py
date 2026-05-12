import os

import bittensor as bt

MAINNET_UID = 34
TESTNET_UID = 379


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


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
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=TESTNET_UID)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Neuron Name",
        default="bitmind",
    )

    parser.add_argument(
        "--neuron.callback_port",
        type=int,
        help="Port for receiving webhook callbacks from miners",
        default=10525,
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

    parser.add_argument(
        "--cache.base-dir",
        type=str,
        default=os.path.expanduser("~/.cache/sn34"),
        help="Base directory for cache storage",
    )

    parser.add_argument("--wandb.entity", type=str, default="bitmindai")

    parser.add_argument("--wandb.off", action="store_true", default=False)


# Shared source-limit/demand-loading args


def add_source_limit_args(parser):
    parser.add_argument(
        "--max-per-source",
        type=int,
        help="Maximum number of media items per source (dataset/scraper/model) (default: 1000)",
        default=1000,
    )

    parser.add_argument(
        "--enable-source-limits",
        action="store_true",
        help="Enable per-source maximum count limits",
        default=True,
    )

    parser.add_argument(
        "--prune-strategy",
        type=str,
        choices=["oldest", "least_used", "random"],
        help="Strategy for pruning media when limits are exceeded (default: oldest)",
        default="oldest",
    )

    parser.add_argument(
        "--min-source-threshold",
        type=float,
        help="Minimum items per source as fraction of max-per-source (default: 0.8 = 80%)",
        default=0.8,
    )

    parser.add_argument(
        "--remove-on-sample",
        action="store_true",
        help="Remove media items when sampled (instead of pruning on add)",
        default=False,
    )


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

    # === GENERATIVE MINER SPECIFIC ARGS ===
    parser.add_argument(
        "--miner.output-dir",
        type=str,
        default="generated_content",
        help="Directory to store generated content",
    )

    parser.add_argument(
        "--miner.max-task-age-hours",
        type=int,
        default=24,
        help="Maximum age (hours) before tasks are cleaned up",
    )

    parser.add_argument(
        "--miner.max-concurrent-tasks",
        type=int,
        default=5,
        help="Maximum number of tasks to process concurrently",
    )

    parser.add_argument(
        "--miner.task-timeout",
        type=float,
        default=300.0,
        help=(
            "Legacy CLI default; generation duration is bounded by each service "
            "(e.g. Runway poll timeout). Queue wait does not count against this."
        ),
    )

    parser.add_argument(
        "--miner.cleanup-interval",
        type=int,
        default=3600,
        help="Interval (seconds) between task cleanup runs",
    )

    parser.add_argument(
        "--miner.worker-threads",
        type=int,
        default=2,
        help="Number of worker threads for task processing",
    )

    parser.add_argument(
        "--miner.type",
        type=str,
        choices=["GENERATOR", "ENCODER", "CAPTIONER"],
        default=os.environ.get("MINER_TYPE", "GENERATOR"),
        help="Miner role advertised to validators",
    )

    parser.add_argument(
        "--dps-artifact.processor-command",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_PROCESSOR_COMMAND"),
        help="Optional command run for encoder/captioner artifact tasks; receives task metadata through environment variables",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-endpoint-url",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_ENDPOINT_URL"),
        help="Miner output R2 endpoint URL for DPS artifacts",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-bucket",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_BUCKET"),
        help="Miner output R2 bucket for DPS artifacts",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-region",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_REGION", "auto"),
        help="Miner output R2/S3 region hint; R2 commonly accepts auto",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-prefix",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_PREFIX", "dps-artifacts/"),
        help="Miner output R2 prefix for DPS artifacts",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-read-access-key-id",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_READ_ACCESS_KEY_ID"),
        help="Validator-readable access key id for miner DPS artifacts",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-read-secret-access-key",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_READ_SECRET_ACCESS_KEY"),
        help="Validator-readable secret access key for miner DPS artifacts",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-read-session-token",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_READ_SESSION_TOKEN"),
        help="Optional validator-readable session token for miner DPS artifacts",
    )

    parser.add_argument(
        "--dps-artifact.output-format",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_FORMAT", "npz"),
        help="Published DPS artifact output format, e.g. npz, parquet, jsonl",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-manifest-url",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_MANIFEST_URL"),
        help="Optional public or validator-readable manifest URL for miner DPS artifact outputs",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-manifest-key",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_MANIFEST_KEY"),
        help="Optional R2 manifest object key for miner DPS artifact outputs",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-write-access-key-id",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_WRITE_ACCESS_KEY_ID"),
        help="Miner write access key id for uploading DPS artifacts to R2",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-write-secret-access-key",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_WRITE_SECRET_ACCESS_KEY"),
        help="Miner write secret access key for uploading DPS artifacts to R2",
    )

    parser.add_argument(
        "--dps-artifact.output-r2-write-session-token",
        type=str,
        default=os.environ.get("DPS_ARTIFACT_OUTPUT_R2_WRITE_SESSION_TOKEN"),
        help="Optional miner write session token for uploading DPS artifacts to R2",
    )
    
    # Webhook configuration
    parser.add_argument(
        "--miner.webhook-max-retries",
        type=int,
        default=3,
        help="Maximum number of webhook retry attempts",
    )
    parser.add_argument(
        "--miner.webhook-retry-delay",
        type=float,
        default=2.0,
        help="Base delay between webhook retries (exponential backoff)",
    )
    parser.add_argument(
        "--miner.webhook-timeout",
        type=float,
        default=30.0,
        help="Timeout for webhook requests in seconds",
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
        "--generator-challenge-interval",
        type=int,
        help="How often we send challenges to generative miners, measured in 12 second blocks.",
        default=110,
    )

    parser.add_argument(
        "--dps-artifact-task-interval",
        type=int,
        help="How often validators assign R2 artifact tasks to encoder/captioner miners, measured in 12 second blocks.",
        default=int(os.environ.get("DPS_ARTIFACT_TASK_INTERVAL", 360)),
    )

    parser.add_argument(
        "--dataset-download-interval",
        type=int,
        help="How often to download and extract datasets, measured in 12 second blocks (default: 720 = 2.4 hours)",
        default=1800,
    )

    parser.add_argument(
        "--wandb-restart-interval",
        type=int,
        help="How often we restart wandb run to avoid log truncation",
        default=2000,
    )

    parser.add_argument(
        "--neuron.callback-port",
        type=int,
        help="Port for generative challenge callbacks (internal binding)",
        default=10525,
    )

    parser.add_argument(
        "--neuron.external-callback-port",
        type=int,
        help="External port to advertise to miners for callbacks (defaults to callback-port)",
        default=None,
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
        default=240.0,
    )

    parser.add_argument(
        "--neuron.miner-connect-timeout",
        type=float,
        help="TCP connection timeout for miner requests in seconds",
        default=6.0,
    )

    parser.add_argument(
        "--neuron.miner-sock-connect-timeout",
        type=float,
        help="Socket connection timeout for miner requests in seconds",
        default=5.0,
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
        "--store-failed-media",
        action="store_true",
        help="Save media that fails verification (tampering, duplicates, C2PA) to disk for inspection",
        default=False,
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
        "--scoring.window",
        type=int,
        help="Number of recent predictions to consider in evaluation",
        default=200,
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
        "--benchmark-api-url",
        type=str,
        help="Base URL for the benchmark API",
        default=os.environ.get("BENCHMARK_API_URL", "https://gas.bitmind.ai"),
    )

    parser.add_argument(
        "--dps-artifact-rewards-path",
        type=str,
        help="Optional JSON file containing DPS mechanism-1 artifact verification stats",
        default=os.environ.get("DPS_ARTIFACT_REWARDS_PATH") or os.environ.get("ENCODER_REWARDS_PATH"),
    )

    parser.add_argument(
        "--enable-dps-artifact-mechanism",
        action="store_true",
        help="Submit a second weight vector for DPS artifact mechanism 1",
        default=_env_bool("ENABLE_DPS_ARTIFACT_MECHANISM"),
    )

    parser.add_argument(
        "--dps-artifact.publish-input-to-chain",
        action="store_true",
        help="Publish validator DPS input R2 location and scoped read credentials to the subnet commitment",
        default=_env_bool("DPS_ARTIFACT_PUBLISH_INPUT_TO_CHAIN"),
    )

    parser.add_argument(
        "--dps-artifact-mechanism-id",
        type=int,
        help="Bittensor mechanism id for DPS artifact rewards",
        default=int(os.environ.get("DPS_ARTIFACT_MECHANISM_ID", 1)),
    )

    parser.add_argument(
        "--dps-artifact.encoder-weight",
        type=float,
        help="Mechanism-1 budget share for encoder rewards",
        default=float(os.environ.get("DPS_ARTIFACT_ENCODER_WEIGHT", 0.8)),
    )

    parser.add_argument(
        "--dps-artifact.captioner-weight",
        type=float,
        help="Mechanism-1 budget share reserved for captioner rewards",
        default=float(os.environ.get("DPS_ARTIFACT_CAPTIONER_WEIGHT", 0.2)),
    )

    parser.add_argument(
        "--dps-artifact.sample-size",
        type=int,
        help="Maximum encoder/captioner miners per role to receive each R2 artifact assignment round; 0 means all active miners.",
        default=int(os.environ.get("DPS_ARTIFACT_SAMPLE_SIZE", 0)),
    )

    parser.add_argument(
        "--dps-artifact.resolution",
        type=str,
        help="Required artifact input/output resolution for DPS encoder/captioner tasks, e.g. 512x512 or 720p",
        default=os.environ.get("DPS_ARTIFACT_RESOLUTION"),
    )

    parser.add_argument(
        "--dps-artifact.max-frames",
        type=int,
        help="Maximum frames miners should use per DPS artifact item",
        default=(
            int(os.environ["DPS_ARTIFACT_MAX_FRAMES"])
            if os.environ.get("DPS_ARTIFACT_MAX_FRAMES")
            else None
        ),
    )

    parser.add_argument(
        "--dps-artifact.encoding-model",
        type=str,
        help="VAE encoder or encoding model miners must use for DPS artifact tasks",
        default=os.environ.get("DPS_ARTIFACT_ENCODING_MODEL")
        or os.environ.get("DPS_ARTIFACT_VAE_ENCODER"),
    )

    parser.add_argument(
        "--dps-artifact.r2-endpoint-url",
        type=str,
        help="Cloudflare R2 endpoint URL for DPS artifact source data",
        default=os.environ.get("DPS_ARTIFACT_R2_ENDPOINT_URL"),
    )

    parser.add_argument(
        "--dps-artifact.r2-bucket",
        type=str,
        help="Cloudflare R2 bucket containing DPS artifact source data",
        default=os.environ.get("DPS_ARTIFACT_R2_BUCKET"),
    )

    parser.add_argument(
        "--dps-artifact.r2-prefix",
        type=str,
        help="Cloudflare R2 key prefix for the current DPS artifact dataset shard",
        default=os.environ.get("DPS_ARTIFACT_R2_PREFIX"),
    )

    parser.add_argument(
        "--dps-artifact.r2-manifest-url",
        type=str,
        help="Public or presigned R2 manifest URL describing the data shard miners should pull",
        default=os.environ.get("DPS_ARTIFACT_R2_MANIFEST_URL"),
    )

    parser.add_argument(
        "--dps-artifact.r2-manifest-key",
        type=str,
        help="R2 object key for the DPS artifact data manifest",
        default=os.environ.get("DPS_ARTIFACT_R2_MANIFEST_KEY"),
    )

    parser.add_argument(
        "--dps-artifact.r2-region",
        type=str,
        help="R2/S3 region hint for clients; R2 commonly accepts auto",
        default=os.environ.get("DPS_ARTIFACT_R2_REGION", "auto"),
    )

    parser.add_argument(
        "--dps-artifact.r2-read-access-key-id",
        type=str,
        help="Scoped read-only R2 access key id for miners; chain commitments are public",
        default=os.environ.get("DPS_ARTIFACT_R2_READ_ACCESS_KEY_ID"),
    )

    parser.add_argument(
        "--dps-artifact.r2-read-secret-access-key",
        type=str,
        help="Scoped read-only R2 secret access key for miners; chain commitments are public",
        default=os.environ.get("DPS_ARTIFACT_R2_READ_SECRET_ACCESS_KEY"),
    )

    parser.add_argument(
        "--dps-artifact.r2-read-session-token",
        type=str,
        help="Optional scoped R2 read session token for miners; chain commitments are public",
        default=os.environ.get("DPS_ARTIFACT_R2_READ_SESSION_TOKEN"),
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
        default=0.4,
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
        default=0.3,
    )

    parser.add_argument(
        "--challenge.generator-prob",
        type=float,
        help="Probability of selecting generator content for synthetic and semisynthetic challenges",
        default=0.5,
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
        default=24,
    )

    parser.add_argument(
        "--verification.temporal-phash-jump-threshold",
        type=int,
        help="Hamming distance threshold for detecting frame jumps in tampered videos (default: 30)",
        default=30,
    )

    parser.add_argument(
        "--verification.temporal-tamper-ratio",
        type=float,
        help="Fraction of frames that must be jumps to flag as tampered (default: 0.10)",
        default=0.10,
    )

    parser.add_argument(
        "--verification.embedding-dup-threshold",
        type=float,
        help="Cosine similarity threshold for CLIP embedding duplicate detection (default: 0.96)",
        default=0.96,
    )

    parser.add_argument(
        "--verification.global-dup-hamming-threshold",
        type=int,
        help="Hamming distance threshold for global (cross-prompt) perceptual hash duplicate detection (default: 4)",
        default=4,
    )

    parser.add_argument(
        "--prompt-modalities",
        type=str,
        help="Comma-separated modalities to generate prompts for (image,video). Default: video only",
        default="video",
    )


def add_generation_service_args(parser):
    """Add generation service specific arguments to the parser."""

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for generation models (cuda/cpu)",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for generation",
        default=1,
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="List of generation tasks to run",
        default=["prompt", "t2i", "i2i", "t2v", "i2v"],
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retries for generation",
        default=3,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for generation operations in seconds",
        default=300,
    )

    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        help="Number of prompts to generate per cycle",
        default=100,
    )

    parser.add_argument(
        "--prompt-modalities",
        type=str,
        help="Comma-separated modalities to generate prompts for (image,video). Default: video only",
        default="video",
    )

    parser.add_argument(
        "--local-batch-size",
        type=int,
        help="Number of local generations to run per cycle",
        default=1,
    )

    parser.add_argument(
        "--tps-batch-size",
        type=int,
        help="Number of third-party service generations to run per cycle",
        default=10,
    )

    # Shared source-limit args
    add_source_limit_args(parser)


def add_data_service_args(parser):
    """Add data service specific arguments to the parser."""

    parser.add_argument(
        "--benchmark-api-url",
        type=str,
        help="Base URL for the benchmark API (used for verification upload)",
        default=os.environ.get("BENCHMARK_API_URL", "https://gas.bitmind.ai"),
    )

    parser.add_argument(
        "--upload-check-interval",
        type=int,
        help="Upload interval in blocks (default: 300 = 1 hour)",
        default=300,
    )

    parser.add_argument(
        "--dataset-interval",
        type=int,
        help="Dataset download interval in blocks (default: 1800 = 6 hours)",
        default=1800,
    )

    parser.add_argument(
        "--hf-image-repo",
        type=str,
        help="HuggingFace dataset repo for uploaded images",
        default="gasstation/gs-images-v4",
    )

    parser.add_argument(
        "--hf-video-repo",
        type=str,
        help="HuggingFace dataset repo for uploaded videos",
        default="gasstation/gs-videos-v4",
    )

    parser.add_argument(
        "--dataset-images-per-parquet",
        type=int,
        help="Number of images to extract per parquet file (default: 100)",
        default=100,
    )

    parser.add_argument(
        "--dataset-videos-per-zip",
        type=int,
        help="Number of videos to extract per zip file (default: 50)",
        default=200,
    )

    parser.add_argument(
        "--dataset-parquet-per-dataset",
        type=int,
        help="Number of parquet files to download per dataset (default: 5)",
        default=2,
    )

    parser.add_argument(
        "--dataset-zips-per-dataset",
        type=int,
        help="Number of zip files to download per dataset (default: 2)",
        default=2,
    )

    parser.add_argument(
        "--upload-batch-size",
        type=int,
        help="Maximum number of media files to upload to HuggingFace per batch per modality",
        default=1000,
    )

    parser.add_argument(
        "--upload-image-threshold",
        type=int,
        help="Minimum number of unuploaded image files to trigger an image upload cycle",
        default=1000,
    )

    parser.add_argument(
        "--upload-video-threshold",
        type=int,
        help="Minimum number of unuploaded video files to trigger a video upload cycle",
        default=10,
    )

    parser.add_argument(
        "--upload-max-batches",
        type=int,
        help="Maximum number of upload batches to process per cycle",
        default=5,
    )

    parser.add_argument(
        "--upload-max-duration-hours",
        type=float,
        help="Safety measure:Maximum hours an uploader thread may run before being treated as hung (default: 2)",
        default=2.0,
    )

    parser.add_argument(
        "--images-per-archive",
        type=int,
        help="Number of images per tar archive (target ~100-200MB per archive)",
        default=500,
    )

    parser.add_argument(
        "--videos-per-archive",
        type=int,
        help="Number of videos per tar archive (target ~650MB per archive)",
        default=200,
    )

    parser.add_argument(
        "--cleanup-interval",
        type=int,
        help="Cleanup interval in blocks for removing uploaded media (default: 7200 = ~24 hours)",
        default=7200,
    )

    parser.add_argument(
        "--cleanup-min-age-hours",
        type=float,
        help="Minimum age in hours before uploaded media can be cleaned up (default: 48)",
        default=48.0,
    )

    parser.add_argument(
        "--cleanup-batch-size",
        type=int,
        help="Number of media entries to clean up per batch (default: 1000)",
        default=1000,
    )

    # Shared source-limit args
    add_source_limit_args(parser)
