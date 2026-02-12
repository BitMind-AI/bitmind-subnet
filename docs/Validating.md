# Validator Guide


## PM2 setup

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with validator setup.

> Once you've run the installation script, create a `.env.validator` file in the project root:

```bash
$ cp .env.validator.template .env.validator
```

See [.env.validator.template](../.env.validator.template) for all options. Then activate the virtual environment and start your validator processes:

```bash
$ source .venv/bin/activate
$ gascli validator start
```
The above command will create 3 pm2 processes:
```bash
┌────┬───────────────────┬─────────────┬─────────┬─────────┬──────────┬────────┬──────┬───────────┬──────────┬──────────┬──────────┬──────────┐
│ id │ name              │ namespace   │ version │ mode    │ pid      │ uptime │ ↺    │ status    │ cpu      │ mem      │ user     │ watching │
├────┼───────────────────┼─────────────┼─────────┼─────────┼──────────┼────────┼──────┼───────────┼──────────┼──────────┼──────────┼──────────┤
│ 2  │ sn34-data         │ default     │ N/A     │ fork    │ 4032914  │ 2s     │ 72   │ online    │ 100%     │ 529.0mb  │ user     │ disabled │
│ 1  │ sn34-generator    │ default     │ N/A     │ fork    │ 4032936  │ 2s     │ 72   │ online    │ 100%     │ 448.5mb  │ user     │ disabled │
│ 0  │ sn34-validator    │ default     │ N/A     │ fork    │ 4032918  │ 2s     │ 72   │ online    │ 100%     │ 504.0mb  │ user     │ disabled │
└────┴───────────────────┴─────────────┴─────────┴─────────┴──────────┴────────┴──────┴───────────┴──────────┴──────────┴──────────┴──────────┘
```
- **sn34-data**: Handles data downloads
- **sn34-generator**: Responsible for generating prompts, synthetic media, and validating miner-generated data
- **sn34-validator**: Core validator logic. Challenges, scoring, weight setting.


### Validator Operations

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Then run validator commands:
```bash
# Start validator services
gascli validator start
gascli v start                   # Using alias
gascli v stop
gascli v status
gascli v logs
gascli v --help
```


## Docker Deployment

As an alternative to the PM2-based setup above, you can run the validator entirely in Docker. The Docker setup runs all 3 services (`sn34-validator`, `sn34-generator`, `sn34-data`) inside a single container managed by supervisord.

### Prerequisites

- **Docker** (with Docker Compose v2)
- **NVIDIA Container Toolkit** for GPU passthrough. Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- **Bittensor wallet** files already created on the host (typically at `~/.bittensor/wallets/`)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd bitmind-subnet
   ```

2. **Create your `.env.validator`** file:
   ```bash
   cp .env.validator.template .env.validator
   ```
   Fill in your wallet name, hotkey, API keys, and network settings. See [.env.validator.template](../.env.validator.template) for all available options.

3. **Build and start** (use `--env-file .env.validator` so the same file drives both container env and Compose options like `WALLET_PATH` and `CALLBACK_PORT`):
   ```bash
   docker compose --env-file .env.validator up -d --build
   ```

4. **View logs**:
   ```bash
   docker compose logs -f validator
   ```


### Updating

Please enable autoupdate by adding the following crontab entry:

```bash
*/5 * * * * /path/to/bitmind-subnet/docker/autoupdate.sh >> /var/log/bitmind-docker-update.log 2>&1
```

To manually rebuild and recreate the container:

```bash
git pull
docker compose --env-file .env.validator build
docker compose --env-file .env.validator up -d
```

### Configuration

| Variable | Docker behavior |
|---|---|
| `SN34_CACHE_DIR` | Overridden to `/root/.cache/sn34` inside the container (persisted via Docker volume) |
| `HF_HOME` | Overridden to `/root/.cache/huggingface` inside the container (persisted via Docker volume) |
| `AUTO_UPDATE` | In-container autoupdate is disabled. Use manual rebuild or the cron-based `docker/autoupdate.sh` (see Updating) |
| `WALLET_PATH` | Host path for wallet bind-mount. Set in `.env.validator`; used by Compose when you pass `--env-file .env.validator` |
| `NETUID` | Auto-derived from `CHAIN_ENDPOINT`. Set explicitly if using a custom endpoint |
| `START_VALIDATOR` | Set to `false` to disable the validator process |
| `START_GENERATOR` | Set to `false` to disable the generator process |
| `START_DATA` | Set to `false` to disable the data process |

### Wallet Files

Wallet files are bind-mounted from the host. Set `WALLET_PATH` in your `.env.validator` file (see [.env.validator.template](../.env.validator.template)); the default is `~/.bittensor/wallets`. When you run with `--env-file .env.validator`, Compose uses that value for the volume mount. To override at runtime you can still set it in your shell before running `docker compose`.

### Persistent Storage

The Docker setup uses named volumes to persist data across container restarts:

- **`sn34-cache`** -- SQLite database, media files, state files (`/root/.cache/sn34/`)
- **`hf-cache`** -- HuggingFace model downloads (`/root/.cache/huggingface/`)

> **Note**: The first startup will download 100+ GB of ML models into the `hf-cache` volume. Subsequent restarts reuse the cached models.

### Common Operations

Use `--env-file .env.validator` so Compose reads `WALLET_PATH`, `CALLBACK_PORT`, and service toggles from your config:

```bash
# Start the validator
docker compose --env-file .env.validator up -d

# Stop the validator
docker compose down

# View live logs
docker compose logs -f validator

# Rebuild after code changes
docker compose --env-file .env.validator build && docker compose --env-file .env.validator up -d

# Restart the container
docker compose restart validator

# Check container status
docker compose ps

# Shell into the running container
docker compose exec validator bash
```
