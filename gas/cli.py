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


# Path constants
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Service names
VALIDATOR = "sn34-validator"
GENERATOR = "sn34-generator" 
DATA = "sn34-data"
GENERATIVE_MINER = "bitmind-generative-miner"
ALL_SERVICES = [VALIDATOR, GENERATOR, DATA]

# Config file paths
VALIDATOR_CONFIG = "validator.config.js"
MINER_CONFIG = "gen_miner.config.js"
ENV_VALIDATOR = ".env.validator"
ENV_MINER = ".env.gen_miner"


def get_python_interpreter():
    """Get the appropriate Python interpreter path"""
    python_interpreter = "python3"
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_interpreter = str(venv_python)
    click.echo(f"Using python interpreter: {python_interpreter}")
    return python_interpreter


def load_env():
    """Load environment variables from .env.validator"""
    # Get the absolute path to the .env.validator file
    env_file = PROJECT_ROOT / ENV_VALIDATOR
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value


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


def start_validator_services(no_generation=False, no_data_downloads=False):
    """Start validator services using ecosystem config"""
    click.echo("Starting validator services...")

    # Login to W&B if API key is provided (only needed for validator)
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        subprocess.run(["wandb", "login", wandb_key])

    # Use ecosystem config to start services
    ecosystem_path = PROJECT_ROOT / VALIDATOR_CONFIG
    if not ecosystem_path.exists():
        click.echo(f"Error: {VALIDATOR_CONFIG} not found in project root.", err=True)
        return False

    # Set environment variables for service selection
    os.environ["START_VALIDATOR"] = "true"
    os.environ["START_GENERATOR"] = "false" if no_generation else "true"
    os.environ["START_DATA"] = "false" if no_data_downloads else "true"

    # Clean up existing services
    services_to_clean = []
    if not no_generation:
        services_to_clean.append(GENERATOR)
    if not no_data_downloads:
        services_to_clean.append(DATA)
    services_to_clean.append(VALIDATOR)
    clean_existing_services(services_to_clean)

    # Start services using ecosystem config
    result = subprocess.run(["pm2", "start", str(ecosystem_path)])
    
    if result.returncode == 0:
        services_started = ["validator"]
        if not no_generation:
            services_started.append("generator")
        if not no_data_downloads:
            services_started.append("data")
        click.echo(f"Validator services started: {', '.join(services_started)}")
    else:
        click.echo("Some validator services failed to start", err=True)
    
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
    # Load environment variables from .env.validator
    load_env()
    
    # Start services using the unified function
    return start_validator_services(no_generation, no_data_downloads)


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
    db_stats_script = SCRIPT_DIR / "cache" / "util" / "db_stats.py"

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
    db_rows_script = SCRIPT_DIR / "cache" / "util" / "db_rows.py"

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


@cli.group(cls=AliasedGroup, name="discriminator")
def discriminator():
    """Discriminator miner management commands"""
    pass


# Add alias for discriminator group
cli.add_command(discriminator, name="d")


@discriminator.command()
@click.option("--onnx-dir", help="Path to directory containing ONNX files")
@click.option("--model-zip", help="Path to pre-existing model zip file")
@click.option("--wallet-name", default="default", help="Bittensor wallet name")
@click.option("--wallet-hotkey", default="default", help="Bittensor hotkey name")
@click.option("--netuid", default=34, help="Subnet UID")
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
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except Exception as e:
        sys.exit(1)


discriminator.add_command(push_discriminator, name="push")


# Benchmark discriminator models against BitMind datasets
@discriminator.command(name="benchmark")
@click.option("--image-model", help="Path to image detector ONNX model")
@click.option("--video-model", help="Path to video detector ONNX model")
@click.option("-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("--no-stream-images", is_flag=True, help="Disable streaming images; download locally")
@click.option("--hf-home", help="Override Hugging Face cache root (sets HF_HOME)")
@click.option("--temp-dir", help="Directory for temporary video extraction (overrides TMPDIR)")
@click.option("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate per modality (use all if omitted)")
def benchmark(image_model, video_model, v, no_stream_images, hf_home, temp_dir, max_samples):
    """Run image/video benchmarks for provided ONNX models"""
    if not image_model and not video_model:
        click.echo("Error: Either --image-model or --video-model must be provided", err=True)
        return

    # Build command to invoke the benchmark runner
    cmd = [sys.executable, "-m", "neurons.discriminator.local_eval.run_benchmark"]
    if image_model:
        cmd.extend(["--image_model", image_model])
    if video_model:
        cmd.extend(["--video_model", video_model])
    if v:
        cmd.extend(["-" + "v" * v])
    # Stream by default; pass disable flag if requested
    if no_stream_images:
        cmd.append("--no-stream-images")
    if hf_home:
        cmd.extend(["--hf-home", hf_home])
    if temp_dir:
        cmd.extend(["--temp-dir", temp_dir])
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except Exception as e:
        sys.exit(1)

# =============================================================================
# GENERATOR COMMANDS
# =============================================================================


@cli.group(cls=AliasedGroup, name="generator")
def generator():
    """Generative miner management commands"""
    pass


# Add aliases for generator group
cli.add_command(generator, name="gen")
cli.add_command(generator, name="g")


def load_miner_env():
    """Load environment variables from .env.gen_miner"""
    # Get the absolute path to the .env.gen_miner file
    env_file = PROJECT_ROOT / ENV_MINER
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value


def start_miner_services():
    """Start generative miner using ecosystem config"""
    click.echo("Starting generative miner...")

    # Load miner environment variables
    load_miner_env()

    # Use ecosystem config to start miner
    ecosystem_path = PROJECT_ROOT / MINER_CONFIG
    if not ecosystem_path.exists():
        click.echo(f"Error: {MINER_CONFIG} not found in project root.", err=True)
        return False

    # Clean up existing miner service
    result = subprocess.run(["pm2", "list"], capture_output=True, text=True)
    if GENERATIVE_MINER in result.stdout:
        click.echo(f"Deleting existing {GENERATIVE_MINER}...")
        subprocess.run(["pm2", "delete", GENERATIVE_MINER])
        import time
        time.sleep(1)

    # Start miner using ecosystem config
    result = subprocess.run(["pm2", "start", str(ecosystem_path)])
    
    if result.returncode == 0:
        click.echo("✅ Generative miner started successfully!")
        # Show status
        subprocess.run(["pm2", "show", GENERATIVE_MINER])
    else:
        click.echo("❌ Generative miner failed to start", err=True)
    
    return result.returncode == 0


@generator.command()
def start():
    """Start the generative miner"""
    load_miner_env()
    return start_miner_services()


@generator.command()
def stop():
    """Stop the generative miner"""
    click.echo("Stopping generative miner...")
    run_pm2_command("stop", GENERATIVE_MINER)


@generator.command()
def delete():
    """Delete the generative miner"""
    click.echo("Deleting generative miner...")
    run_pm2_command("delete", GENERATIVE_MINER)


@generator.command()
def restart():
    """Restart the generative miner"""
    click.echo("Restarting generative miner...")
    run_pm2_command("restart", GENERATIVE_MINER)


@generator.command()
def status():
    """Show status of the generative miner"""
    result = subprocess.run(["pm2", "show", GENERATIVE_MINER], capture_output=True, text=True)
    if result.stdout:
        click.echo(result.stdout)
    else:
        click.echo("Generative miner not found")


@generator.command()
@click.option("--lines", "-n", default=50, help="Number of log lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def logs(lines, follow):
    """Show generative miner logs"""
    
    if follow:
        # Follow logs in real-time
        subprocess.run(["pm2", "logs", GENERATIVE_MINER, "--lines", str(lines)])
    else:
        # Show recent logs
        result = subprocess.run(["pm2", "logs", GENERATIVE_MINER, "--lines", str(lines), "--nostream"], 
                              capture_output=True, text=True)
        if result.stdout:
            click.echo(result.stdout)
        else:
            click.echo("No logs found for generative miner")


@generator.command()
def info():
    """Show generative miner information and configuration"""
    load_miner_env()
    
    click.echo("=== Generative Miner Configuration ===")
    click.echo(f"Wallet Name: {os.environ.get('BT_WALLET_NAME', 'miner1')}")
    click.echo(f"Wallet Hotkey: {os.environ.get('BT_WALLET_HOTKEY', 'default')}")
    click.echo(f"Network: {os.environ.get('BT_CHAIN_ENDPOINT', 'wss://test.finney.opentensor.ai:443')}")
    click.echo(f"NetUID: {os.environ.get('BT_NETUID', '379')}")
    click.echo(f"Axon Port: {os.environ.get('BT_AXON_PORT', '8093')}")
    click.echo(f"Device: {os.environ.get('MINER_DEVICE', 'auto')}")
    click.echo(f"Output Directory: {os.environ.get('MINER_OUTPUT_DIR', '/tmp/generated_content')}")
    click.echo(f"Max Concurrent Tasks: {os.environ.get('MINER_MAX_CONCURRENT_TASKS', '5')}")
    
    click.echo("\n=== API Keys Status ===")
    
    # Dynamically get API key requirements from all available services
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from neurons.generator.services.service_registry import ServiceRegistry
        
        registry = ServiceRegistry()
        api_keys = registry.get_all_api_key_requirements()
        
        if api_keys:
            for key, description in api_keys.items():
                status = "✅ Configured" if os.environ.get(key) else "❌ Not configured"
                click.echo(f"{description}: {status}")
        else:
            click.echo("No API key requirements found from services")
            
    except Exception as e:
        click.echo(f"❌ Could not load service API key requirements: {e}")
        # Fallback to basic message
        click.echo("Run the generator to see API key status")


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


@cli.command(name="install-py-deps")
@click.option("--clear-venv", is_flag=True, help="Delete existing .venv directory (default is to preserve)")
def install_py_deps(clear_venv):
    """Install Python dependencies via uv"""
    click.echo("Installing Python dependencies...")
    
    # Get the path to install.sh in the project root
    install_script = PROJECT_ROOT / "install.sh"
    
    if not install_script.exists():
        click.echo("Error: install.sh not found in project root.", err=True)
        return False
    
    # Build command args
    cmd_args = [str(install_script), "--py-deps-only"]
    if clear_venv:
        cmd_args.append("--clear-venv")
    
    # Run install.sh with appropriate flags
    try:
        result = subprocess.run(cmd_args, check=True)
        click.echo("✅ Python dependencies installation completed!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Python dependencies installation failed with exit code {e.returncode}", err=True)
        return False
    except Exception as e:
        click.echo(f"❌ Error running install script: {e}", err=True)
        return False


@cli.command(name="install-sys-deps")
@click.option("--clear-venv", is_flag=True, help="Delete existing .venv directory (default is to preserve)")
def install_sys_deps(clear_venv):
    """Install system dependencies"""
    click.echo("Installing system dependencies...")
    
    # Get the path to install.sh in the project root
    install_script = PROJECT_ROOT / "install.sh"
    
    if not install_script.exists():
        click.echo("Error: install.sh not found in project root.", err=True)
        return False
    
    # Build command args
    cmd_args = [str(install_script), "--sys-deps-only"]
    if clear_venv:
        cmd_args.append("--clear-venv")
    
    # Run install.sh with appropriate flags
    try:
        result = subprocess.run(cmd_args, check=True)
        click.echo("✅ System dependencies installation completed!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ System dependencies installation failed with exit code {e.returncode}", err=True)
        return False
    except Exception as e:
        click.echo(f"❌ Error running install script: {e}", err=True)
        return False


@cli.command()
@click.option("--clear-venv", is_flag=True, help="Delete existing .venv directory (default is to preserve)")
def install(clear_venv):
    """Run full installation (system + Python dependencies)"""
    click.echo("Running full installation...")
    
    # Get the path to install.sh in the project root
    install_script = PROJECT_ROOT / "install.sh"
    
    if not install_script.exists():
        click.echo("Error: install.sh not found in project root.", err=True)
        return False
    
    # Build command args
    cmd_args = [str(install_script)]
    if clear_venv:
        cmd_args.append("--clear-venv")
    
    # Run install.sh with appropriate flags
    try:
        result = subprocess.run(cmd_args, check=True)
        click.echo("✅ Full installation completed!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Full installation failed with exit code {e.returncode}", err=True)
        return False
    except Exception as e:
        click.echo(f"❌ Error running install script: {e}", err=True)
        return False


if __name__ == "__main__":
    # Check if we're in a virtual environment, if not, try to use the project's venv
    if not hasattr(sys, "real_prefix") and not (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        # Not in a virtual environment, try to use the project's venv
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            # Re-execute with the venv Python interpreter
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

    cli()
