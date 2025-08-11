#!/usr/bin/env python3
"""
GAS CLI - Simple Service Manager
Hierarchical CLI tool for managing GAS miners and validators
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/sn34")


class AliasedGroup(click.Group):
    """A Click Group that supports command aliases and prefix matching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track aliases for help display
        self._aliases = {}
        # Track which commands are aliases (so we can exclude them from help)
        self._alias_names = set()

    def add_command(self, cmd, name=None):
        """Override to track aliases"""
        super().add_command(cmd, name)
        if name and hasattr(cmd, "name") and name != cmd.name:
            # This is an alias
            if cmd.name not in self._aliases:
                self._aliases[cmd.name] = []
            self._aliases[cmd.name].append(name)
            self._alias_names.add(name)

    def list_commands(self, ctx):
        """Return only main commands, not aliases"""
        all_commands = super().list_commands(ctx)
        # Filter out alias names, only return main command names
        return [cmd for cmd in all_commands if cmd not in self._alias_names]

    def get_command(self, ctx, cmd_name):
        # First try to get the exact command
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv

        # Find matches that start with the given prefix (including aliases)
        all_commands = super().list_commands(ctx)
        matches = [x for x in all_commands if x.startswith(cmd_name)]

        if not matches:
            return None

        if len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])

        ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")

    def resolve_command(self, ctx, args):
        # Always return the full command name
        _, cmd, args = super().resolve_command(ctx, args)
        return cmd.name, cmd, args

    def format_commands(self, ctx, formatter):
        """Format the commands list with aliases shown"""
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue
            if cmd.hidden:
                continue

            # Build command name with aliases
            cmd_name = subcommand
            if subcommand in self._aliases:
                aliases = " | ".join(sorted(self._aliases[subcommand]))
                cmd_name = f"{subcommand} ({aliases})"

            commands.append((cmd_name, cmd.get_short_help_str()))

        if commands:
            with formatter.section("Commands"):
                formatter.write_dl(commands)


# Service names
VALIDATOR = "sn34-validator"
GENERATOR = "sn34-generator"
DATA = "sn34-data"
ALL_SERVICES = [VALIDATOR, GENERATOR, DATA]


def get_python_interpreter():
    """Get the appropriate Python interpreter path"""
    python_interpreter = "python3"
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up one level from gas/ to project root
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_interpreter = str(venv_python)
    click.echo(f"Using python interpreter: {python_interpreter}")
    return python_interpreter


def load_env():
    """Load environment variables from .env.validator"""
    # Get the absolute path to the .env.validator file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up one level from gas/ to project root
    env_file = project_root / ".env.validator"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value


def get_network_settings():
    """Determine network settings based on chain endpoint"""
    chain_endpoint = os.environ.get("CHAIN_ENDPOINT", "")
    click.echo(f"Debug: CHAIN_ENDPOINT = '{chain_endpoint}'")

    if "test" in chain_endpoint:
        return 379
    elif "finney" in chain_endpoint:
        return 34
    return None


def get_log_param():
    """Get logging parameter based on LOGLEVEL"""
    loglevel = os.environ.get("LOGLEVEL", "info")

    if loglevel == "trace":
        return "--logging.trace"
    elif loglevel == "debug":
        return "--logging.debug"
    else:
        return "--logging.info"


def get_auto_update_param():
    """Get auto-update parameter"""
    auto_update = os.environ.get("AUTO_UPDATE", "false").lower()
    return "" if auto_update == "true" else "--autoupdate-off"


def get_heartbeat_param():
    """Get heartbeat parameter"""
    heartbeat = os.environ.get("HEARTBEAT", "false").lower()
    return "--heartbeat" if heartbeat == "true" else ""


def run_pm2_command(cmd: str, service: Optional[str] = None):
    """Run PM2 command and handle output"""
    try:
        if service:
            result = subprocess.run(
                ["pm2", cmd, service], capture_output=True, text=True, check=True
            )
        else:
            result = subprocess.run(
                ["pm2", cmd], capture_output=True, text=True, check=True
            )
        return result.stdout
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running PM2 command: {e}", err=True)
        return None


def clean_existing_services(services: List[str]):
    """Clean up existing services"""
    for service in services:
        result = subprocess.run(["pm2", "list"], capture_output=True, text=True)
        if service in result.stdout:
            click.echo(f"Deleting existing {service}...")
            subprocess.run(["pm2", "delete", service])
            import time

            time.sleep(1)


def start_validator():
    """Start the validator service"""
    click.echo("Starting validator...")

    # Login to W&B if API key is provided (only needed for validator)
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        subprocess.run(["wandb", "login", wandb_key])

    netuid = get_network_settings()
    if not netuid:
        click.echo("Error: Could not determine NETUID from CHAIN_ENDPOINT", err=True)
        return False

    log_param = get_log_param()
    auto_update_param = get_auto_update_param()
    heartbeat_param = get_heartbeat_param()

    # Use virtual environment Python if available
    python_interpreter = get_python_interpreter()

    # Get the absolute path to the validator script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up one level from gas/ to project root
    validator_script = project_root / "neurons" / "validator" / "validator.py"

    validator_cmd = [
        "pm2",
        "start",
        str(validator_script),
        "--interpreter",
        python_interpreter,
        "--name",
        VALIDATOR,
        "--",
        "--wallet.name",
        os.environ.get("WALLET_NAME", ""),
        "--wallet.hotkey",
        os.environ.get("WALLET_HOTKEY", ""),
        "--netuid",
        str(netuid),
        "--subtensor.chain_endpoint",
        os.environ.get("CHAIN_ENDPOINT", ""),
        "--proxy.port",
        os.environ.get("PROXY_PORT", "10913"),
        "--neuron.sample-size",
        "256",
        "--cache.base-dir",
        os.environ.get("SN34_CACHE_DIR", DEFAULT_CACHE_DIR),
        "--cache.retain_previous_epoch",
        "--gen.off",
        "--scraper.off",
        log_param,
        auto_update_param,
        heartbeat_param,
    ]

    result = subprocess.run(validator_cmd)
    return result.returncode == 0


def start_generator():
    """Start the generator service"""
    click.echo("Starting generator service...")

    # Use virtual environment Python if available
    python_interpreter = get_python_interpreter()

    # Get the absolute path to the service script
    generator_script = Path(__file__).parent / "services" / "generator_service.py"

    generator_cmd = [
        "pm2",
        "start",
        str(generator_script),
        "--interpreter",
        python_interpreter,
        "--name",
        GENERATOR,
        "--",
        "--cache-dir",
        os.environ.get("SN34_CACHE_DIR", DEFAULT_CACHE_DIR),
        "--device",
        os.environ.get("DEVICE", "cuda"),
        "--batch-size",
        "1",
        "--log-level",
        os.environ.get("LOGLEVEL", "info"),
    ]

    result = subprocess.run(generator_cmd)
    return result.returncode == 0


def start_data():
    """Start the data service"""
    click.echo("Starting data service...")

    # Use virtual environment Python if available
    python_interpreter = get_python_interpreter()

    # Get the absolute path to the service script
    data_script = Path(__file__).parent / "services" / "data_service.py"

    data_cmd = [
        "pm2",
        "start",
        str(data_script),
        "--interpreter",
        python_interpreter,
        "--name",
        DATA,
        "--",
        "--cache-dir",
        os.environ.get("SN34_CACHE_DIR", DEFAULT_CACHE_DIR),
        "--chain-endpoint",
        os.environ.get("CHAIN_ENDPOINT", ""),
        "--scraper-interval",
        "300",
        "--dataset-interval",
        "1800",
        "--log-level",
        os.environ.get("LOGLEVEL", "info"),
    ]

    result = subprocess.run(data_cmd)
    return result.returncode == 0


# =============================================================================
# MAIN CLI GROUP
# =============================================================================


@click.group(cls=AliasedGroup)
@click.pass_context
def cli(ctx):
    """GAS CLI - Simple Service Manager for Miners and Validators"""
    # Load environment variables
    load_env()

    # Set default values
    os.environ.setdefault("PROXY_PORT", "10913")
    os.environ.setdefault("DEVICE", "cuda")
    os.environ.setdefault("SN34_CACHE_DIR", DEFAULT_CACHE_DIR)

    # Suppress logs
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["DIFFUSERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["ACCELERATE_LOG_LEVEL"] = "error"


# =============================================================================
# VALIDATOR COMMANDS
# =============================================================================


@cli.group(cls=AliasedGroup, name="validator")
def validator():
    """Validator management commands"""
    pass


# Add aliases for validator group
cli.add_command(validator, name="vali")
cli.add_command(validator, name="v")


@validator.command()
@click.option("--no-generation", is_flag=True, help="Skip starting generator")
@click.option("--no-data-downloads", is_flag=True, help="Skip starting data service")
def start(no_generation, no_data_downloads):
    """Start validator services"""
    click.echo("Starting validator services...")

    # Load environment variables from .env.validator
    load_env()

    services_to_start = ["validator"]
    services_to_clean = [VALIDATOR]

    if not no_generation:
        services_to_start.append("generator")
        services_to_clean.append(GENERATOR)
    if not no_data_downloads:
        services_to_start.append("data")
        services_to_clean.append(DATA)

    # Clean up existing services
    clean_existing_services(services_to_clean)

    # Start services
    success = True
    for service in services_to_start:
        if service == "validator":
            success &= start_validator()
        elif service == "generator":
            success &= start_generator()
        elif service == "data":
            success &= start_data()

    if success:
        click.echo(f"Validator services started: {', '.join(services_to_start)}")
    else:
        click.echo("Some validator services failed to start", err=True)


@validator.command()
@click.option(
    "--service",
    default="all",
    help="Service to stop (validator, generator, data, or all)",
)
def stop(service):
    """Stop validator services"""
    click.echo("Stopping validator services...")

    if service == "all":
        for service_name in ALL_SERVICES:
            run_pm2_command("stop", service_name)
    else:
        service_map = {"validator": VALIDATOR, "generator": GENERATOR, "data": DATA}
        pm2_service = service_map.get(service, service)
        run_pm2_command("stop", pm2_service)


@validator.command()
@click.option(
    "--service",
    default="all",
    help="Service to delete (validator, generator, data, or all)",
)
def delete(service):
    """Delete validator services"""
    click.echo("Deleting validator services...")

    if service == "all":
        for service_name in ALL_SERVICES:
            run_pm2_command("delete", service_name)
    else:
        service_map = {"validator": VALIDATOR, "generator": GENERATOR, "data": DATA}
        pm2_service = service_map.get(service, service)
        run_pm2_command("delete", pm2_service)


@validator.command()
def status():
    """Show status of validator services"""
    result = subprocess.run(["pm2", "list"], capture_output=True, text=True)
    if result.stdout:
        click.echo(result.stdout)
    else:
        click.echo("No validator services found")


@validator.command()
@click.option(
    "--service",
    default="all",
    help="Service to show logs for (validator, generator, data, or all)",
)
def logs(service):
    """Show logs for validator services"""
    if service == "all":
        subprocess.run(["pm2", "logs"])
    else:
        service_map = {"validator": VALIDATOR, "generator": GENERATOR, "data": DATA}
        pm2_service = service_map.get(service, service)
        subprocess.run(["pm2", "logs", pm2_service])


@validator.command()
@click.option("--db-path", default=None, help="Path to the prompt database")
@click.option("--base-dir", default=None, help="Base directory for cache system")
@click.option("--detailed", is_flag=True, help="Show detailed breakdown of model names and dataset names")
def db_stats(db_path, base_dir, detailed):
    """Show database statistics for prompts, search queries, and media"""
    # Use DEFAULT_CACHE_DIR if not specified
    if base_dir is None:
        base_dir = DEFAULT_CACHE_DIR
    if db_path is None:
        db_path = os.path.join(base_dir, "prompts.db")

    # Get the absolute path to the db_stats script
    script_dir = Path(__file__).parent
    db_stats_script = script_dir / "cache" / "util" / "db_stats.py"

    # Use virtual environment Python if available
    python_interpreter = get_python_interpreter()

    # Build command
    cmd = [
        python_interpreter,
        str(db_stats_script),
        "--db-path",
        db_path,
        "--base-dir",
        base_dir,
    ]
    
    # Add detailed flag if requested
    if detailed:
        cmd.append("--detailed")

    # Execute the db_stats script
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            click.echo("✅ Database statistics completed!")
    except subprocess.CalledProcessError as e:
        click.echo(
            f"❌ Database statistics failed with exit code {e.returncode}", err=True
        )
        sys.exit(e.returncode)
    except Exception as e:
        click.echo(f"❌ Error running db_stats script: {e}", err=True)
        sys.exit(1)


@validator.command(name="db-rows")
@click.option("--db-path", default=None, help="Path to the prompt database")
@click.option(
    "--table",
    type=click.Choice(["prompts", "media"]),
    required=True,
    help="Table to display (prompts or media)",
)
@click.option(
    "--rows", default=10, type=int, help="Number of rows to display (default: 10)"
)
@click.option(
    "--source-type",
    type=click.Choice(["scraper", "dataset", "generated"]),
    help="Filter media table by source type (only applies to media table)",
)
def db_rows(db_path, table, rows, source_type):
    """Show the first N rows of either the prompts or media table"""
    # Validate source-type is only used with media table
    if source_type and table != "media":
        click.echo("❌ --source-type can only be used with --table media", err=True)
        sys.exit(1)

    # Use DEFAULT_CACHE_DIR if not specified
    if db_path is None:
        db_path = os.path.join(DEFAULT_CACHE_DIR, "prompts.db")

    # Get the absolute path to the db_rows script
    script_dir = Path(__file__).parent
    db_rows_script = script_dir / "cache" / "util" / "db_rows.py"

    # Use virtual environment Python if available
    python_interpreter = get_python_interpreter()

    # Build command
    cmd = [
        python_interpreter,
        str(db_rows_script),
        "--db-path",
        db_path,
        "--table",
        table,
        "--rows",
        str(rows),
    ]

    # Add source-type filter if provided
    if source_type:
        cmd.extend(["--source-type", source_type])

    # Execute the db_rows script
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            click.echo("✅ Database rows display completed!")
    except subprocess.CalledProcessError as e:
        click.echo(
            f"❌ Database rows display failed with exit code {e.returncode}", err=True
        )
        sys.exit(e.returncode)
    except Exception as e:
        click.echo(f"❌ Error running db_rows script: {e}", err=True)
        sys.exit(1)


# =============================================================================
# MINER COMMANDS
# =============================================================================


@cli.group(cls=AliasedGroup, name="miner")
def miner():
    """Miner management commands"""
    pass


# Add alias for miner group
cli.add_command(miner, name="m")


@miner.command()
@click.option("--onnx-dir", help="Path to directory containing ONNX files")
@click.option("--model-zip", help="Path to pre-existing model zip file")
@click.option("--wallet-name", default="default", help="Bittensor wallet name")
@click.option("--wallet-hotkey", default="default", help="Bittensor hotkey name")
@click.option("--netuid", default=379, help="Subnet UID")
@click.option("--chain-endpoint", help="Subtensor network endpoint")
@click.option("--retry-delay", default=60, help="Retry delay in seconds")
def push_discriminator(
    onnx_dir, model_zip, wallet_name, wallet_hotkey, netuid, chain_endpoint, retry_delay
):
    """Push discriminator model to Hugging Face and register on blockchain"""
    # Build command arguments for the push_model script
    cmd = [sys.executable, "-m", "neurons.discriminator.push_model"]

    # Add required arguments
    if onnx_dir:
        cmd.extend(["--onnx-dir", onnx_dir])
    elif model_zip:
        cmd.extend(["--model-zip", model_zip])
    else:
        click.echo("Error: Either --onnx-dir or --model-zip must be provided", err=True)
        return

    # Add optional arguments
    cmd.extend(["--wallet-name", wallet_name])
    cmd.extend(["--wallet-hotkey", wallet_hotkey])
    cmd.extend(["--netuid", str(netuid)])

    if chain_endpoint:
        cmd.extend(["--chain-endpoint", chain_endpoint])

    cmd.extend(["--retry-delay", str(retry_delay)])

    # Execute the push_model script
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            click.echo("✅ Model push completed successfully!")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Model push failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)
    except Exception as e:
        click.echo(f"❌ Error running push_model script: {e}", err=True)
        sys.exit(1)


# =============================================================================
# GLOBAL COMMANDS (for backward compatibility and convenience)
# =============================================================================


@cli.command()
def status():
    """Show status of all services (global command)"""
    result = subprocess.run(["pm2", "list"], capture_output=True, text=True)
    if result.stdout:
        click.echo(result.stdout)
    else:
        click.echo("No services found")


if __name__ == "__main__":
    # Check if we're in a virtual environment, if not, try to use the project's venv
    if not hasattr(sys, "real_prefix") and not (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        # Not in a virtual environment, try to use the project's venv
        venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
        if venv_python.exists():
            # Re-execute with the venv Python interpreter
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

    cli()
