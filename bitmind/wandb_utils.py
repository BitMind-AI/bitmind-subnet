import glob
import json
import os
import shutil
import time
import uuid

import bittensor as bt
import wandb


class WandbLogger:
    def __init__(self, config, validator_uid, validator_hotkey):
        """
        Initialize the WandB logger using a single project for both media and results.

        Args:
            config: Bittensor config object
            validator_uid: Validator UID
            validator_hotkey: Validator hotkey for signing
        """
        self.config = config
        self.wandb_dir = config.neuron.full_path

        self.uid = validator_uid
        self.hotkey = validator_hotkey
        self.run = None

        self.session_artifacts = set()

        clean_wandb_cache(self.wandb_dir)

    def start_new_run(self):
        """
        Ensure validator run is active and return it.

        Returns:
            wandb.Run: The active wandb run
        """
        clean_wandb_cache(self.wandb_dir)
        if self.run is None or not wandb.run:
            self.run = init_wandb(
                self.config, "validator", self.uid, self.hotkey, self.wandb_dir
            )
        else:
            self.run.finish()
            self.run = init_wandb(
                self.config, "validator", self.uid, self.hotkey, self.wandb_dir
            )
        return self.run

    def _ensure_run(self):
        """
        Ensure validator run is active and return it.

        Returns:
            wandb.Run: The active wandb run
        """
        if self.run is None or not wandb.run:
            self.run = init_wandb(
                self.config, "validator", self.uid, self.hotkey, self.wandb_dir
            )
        return self.run

    def _check_media_exists(self, filepath):
        """
        Check if a media file has already been logged to WandB using only UUID lookup.

        Args:
            filepath: Path to the media file to check

        Returns:
            tuple: (exists (bool), media_uuid (str or None))
        """
        metadata_path = os.path.splitext(filepath)[0] + ".json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                media_uuid = metadata.get("media_uuid")
                if media_uuid:
                    api = wandb.Api()
                    project = f"subnet-{self.config.netuid}-validator"
                    artifact_path = f"{self.config.wandb.entity}/{project}/media-{media_uuid}:latest"
                    try:
                        artifact = api.artifact(artifact_path)
                        return True, media_uuid
                    except wandb.errors.CommError:
                        pass
            except (json.JSONDecodeError, IOError) as e:
                bt.logging.warning(f"Error reading metadata file: {e}")

        return False, None

    def _maybe_log_media(self, media_path, metadata_path):
        """
        Log media as a WandB Artifact, with simple UUID-based deduplication.
        Only logs media that hasn't been logged yet.
        Only logs synthetic, locally generated media.

        Args:
            media_path: Path to the media file
            metadata_path: Path to the metadata JSON file

        Returns:
            str or None: UUID assigned to the media if logged, None if not logged
        """
        exists, existing_uuid = self._check_media_exists(media_path)
        if exists:
            bt.logging.info(f"Media already exists in WandB with UUID: {existing_uuid}")
            return existing_uuid

        run = self._ensure_run()

        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                bt.logging.warning(f"Error parsing metadata file: {metadata_path}")

        # Only create uuids for and log locally generated synthetic media.
        # All other media are already stored on Huggingface
        if not metadata.get("model_name"):
            return None

        if not metadata.get("media_uuid"):
            metadata["media_uuid"] = str(uuid.uuid4())
            try:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            except IOError as e:
                bt.logging.warning(f"Error writing metadata file: {e}")

        media_uuid = metadata["media_uuid"]
        media_artifact = wandb.Artifact(
            name=f"media-{media_uuid}", type="media", metadata=metadata
        )

        extension = os.path.splitext(media_path)[1]
        media_artifact.add_file(media_path, f"media{extension}")

        run.log_artifact(media_artifact)

        if "media_uuids" not in list(run.summary.keys()):
            run.summary["media_uuids"] = []

        media_uuids = run.summary.get("media_uuids", [])
        if media_uuid not in media_uuids:
            run.summary["media_uuids"] = media_uuids + [media_uuid]
            # run.summary.update()

        bt.logging.info(f"Logged media file to WandB with UUID: {media_uuid}")
        return media_uuid

    def _log_challenge_results(self, challenge_results, media_uuids):
        """
        Log challenge results with reference to media artifact.

        Args:
            challenge_results: Dictionary of challenge results
            media_uuids: List of UUIDs of the associated media
        """
        run = self._ensure_run()
        log_data = {
            "results": challenge_results,
            "media_uuids": media_uuids,
        }

        run.log(log_data)
        bt.logging.info(f"Logged challenge results with media UUIDs: {media_uuids}")

    def log(self, media_sample, challenge_results):
        """
        Combined method to log both media and challenge results.

        Args:
            media_sample: Dictionary containing media paths and metadata paths
            challenge_results: Dictionary of challenge results

        Returns:
            list: List of UUIDs assigned to the logged media
        """

        # Step 1: Log media if applicable
        # Only locally generated synthetic media are logged
        media_path = media_sample.get("path")
        metadata_path = media_sample.get("metadata_path")
        if media_path and metadata_path:
            media_uuids = [self._maybe_log_media(media_path, metadata_path)]
        else:
            media_uuids = []
            for i in range(1):
                media_path = media_sample.get(f"sample_{i}", {}).get("path")
                metadata_path = media_sample.get(f"sample_{i}", {}).get("metadata_path")
                if media_path and metadata_path:
                    media_uuids.append(self._maybe_log_media(media_path, metadata_path))

        # Step 2: Log challenge results with reference to logged media uuid if available
        self._log_challenge_results(challenge_results, media_uuids)
        return media_uuids

    def finish(self):
        """Finish the current run if it exists."""
        if self.run and wandb.run:
            self.run.finish()
            self.run = None


def init_wandb(
    config: bt.config, process: str, uid: int, hotkey: bt.Keypair, wandb_dir: str = None
) -> wandb.run:
    """
    Initialize a Weights & Biases run.

    Args:
        config: Bittensor config object
        process: Valid options are 'validator', 'data-generator', 'media-store'
        uid: Validator uid
        hotkey: Bittensor keypair for signing the run
        wandb_dir: Optional directory for wandb files

    Returns:
        wandb.run: The initialized wandb run, or None if initialization fails
    """
    from bitmind import __version__

    project = f"subnet-{config.netuid}-{process}"
    run_name = f"{process}-{uid}-{__version__}"
    config.run_name = run_name
    config.uid = uid
    config.hotkey = hotkey.ss58_address
    config.version = __version__

    bt.logging.info(f"Initializing wandb run in '{config.wandb.entity}/{project}'")

    try:
        run = wandb.init(
            name=run_name,
            project=project,
            entity=config.wandb.entity,
            config=config,
            dir=wandb_dir if wandb_dir else config.full_path,
            reinit=True,
        )
    except wandb.UsageError as e:
        bt.logging.warning(e)
        bt.logging.warning("Did you run wandb login?")
        return

    # sign the run to prove it's from this  hotkey
    signature = hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run {run_name}")
    return run


def clean_wandb_cache(wandb_dir, hours=1):
    """
    Cleans wandb runs except recent ones and latest-run.

    Args:
        wandb_dir: Directory containing wandb run files
        hours: Number of hours to keep runs for (default: 1)
    """
    if not wandb_dir.endswith("wandb"):
        wandb_dir = os.path.join(wandb_dir, "wandb")

    if not os.path.exists(wandb_dir):
        bt.logging.warning(f"W&B directory not found: {wandb_dir}")
        return

    bt.logging.info(f"Attempting to clean wandb cache at {wandb_dir}")
    run_dirs = [
        d for d in glob.glob(os.path.join(wandb_dir, "run-*")) if os.path.isdir(d)
    ]

    if not run_dirs:
        bt.logging.info("No W&B runs found.")
        return

    # Keep recent runs
    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    recent_runs = [d for d in run_dirs if os.path.getmtime(d) > cutoff_time]

    # Preserve latest-run target
    latest_run_link = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_run_link) and os.path.isdir(latest_run_link):
        try:
            latest_run_target = os.path.realpath(latest_run_link)
            if latest_run_target not in recent_runs and latest_run_target in run_dirs:
                recent_runs.append(latest_run_target)
                bt.logging.debug(
                    f"Preserving latest-run: {os.path.basename(latest_run_target)}"
                )
        except Exception as e:
            bt.logging.warning(f"Error with latest-run: {e}")

    bt.logging.info(
        f"Keeping {len(recent_runs)} runs (modified in last {hours} hours):"
    )
    for run in recent_runs:
        bt.logging.info(f"  - {os.path.basename(run)}")

    runs_removed = 0
    space_freed = 0
    for run_dir in run_dirs:
        if run_dir not in recent_runs:
            try:
                dir_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(run_dir)
                    for filename in filenames
                )
                space_freed += dir_size
                shutil.rmtree(run_dir)
                runs_removed += 1
            except Exception as e:
                bt.logging.warning(f"Error removing {run_dir}: {e}")

    bt.logging.info(
        f"Cleaned {runs_removed} runs, freed {space_freed / (1024*1024):.2f} MB"
    )
