# Validator Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with validator setup.


## Setup Instructions

> Once you've run the installation script, create a `.env.validator` file in the project root. 

```bash
$ cp .env.validator.template .env.validator
```

```bash
# .env.validator.template contents 

# Wallet configuration
WALLET_NAME=your_wallet_name
WALLET_HOTKEY=your_hotkey_name

# Network configuration
CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443/
NETUID=379

# Service configuration
PROXY_PORT=10913
DEVICE=cuda
SN34_CACHE_DIR=~/.cache/sn34_cache

# Logging
LOGLEVEL=info

# Optional features
AUTO_UPDATE=false
HEARTBEAT=false
WANDB_API_KEY=your_wandb_key
```
> Once you've populated `.env.validator`, activate the virtual environment and start your validator processes
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
- **sn34-generator**: Responsible for generating prompts and synthetic/semisynthetic media
- **sn34-validator**: Core validator logic. Challenges, scoring, weight setting.


## Validator Operations

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

