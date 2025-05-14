# Validator Guide

## Before you proceed ⚠️

If you are new to Bittensor (you're probably not if you're reading the validator guide 😎), we recommend familiarizing yourself with the basics in the [Bittensor Docs](https://docs.bittensor.com/) before proceeding.

**Run your own local subtensor** to avoid rate limits set on public endpoints. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary) for setup instructions.

**Understand the minimum compute requirements to run a validator**. Validator neurons on SN34 run a suite of generative (text-to-image, text-to-video, etc.) models that require an **80GB VRAM GPU**. They also maintain a large cache of real and synthetic media to ensure diverse, locally available data for challenging miners. We recommend **1TB of storage**. For more details, please see our [minimum compute documentation](../min_compute.yml)

## Required Hugging Face Model Access

To properly validate, you must gain access to several Hugging Face models used by the subnet. This requires logging in to your Hugging Face account and accepting the terms for each model below:

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [DeepFloyd IF-II-L-v1.0](https://huggingface.co/DeepFloyd/IF-II-L-v1.0)
- [DeepFloyd IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)

> **Note:** Accepting the terms for any one of the DeepFloyd IF models (e.g., IF-II-L or IF-I-XL) will grant you access to all DeepFloyd IF models.
>
> **If you've been validating with us for a while (prior to V3), you've likely already gotten access to these models and can disregard this step.**

To do this:
1. Log in to your Hugging Face account.
2. Visit each model page above.
3. Click the "Access repository" or "Agree and access repository" button to accept the terms.

## Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet
```

We recommend using a Conda virtual environment to install the necessary Python packages.
- You can set up Conda with this [quick command-line install](https://www.anaconda.com/docs/getting-started/miniconda/install#linux). 
- Note that after you run the last commands in the miniconda setup process, you'll be prompted to start a new shell session to complete the initialization. 

With miniconda installed, you can create your virtual environment with this command:

```bash
conda create -y -n bitmind python=3.10
```

- Activating your virtual environment: `conda activate bitmind`
- Deactivating your virtual environment `conda deactivate`

Install the remaining necessary requirements with the following chained command. 
```bash
conda activate bitmind
export PIP_NO_CACHE_DIR=1
chmod +x setup.sh
./setup.sh
```

Before you register, you should first fill out all the necessary fields in `.env.validator`. Make a copy of the template, and fill in your wallet information. 

```
cp .env.validator.template .env.validator
```

## Registration

To validate on our subnet, you must have a registered hotkey.

#### Mainnet

```bash
btcli s register --netuid 34 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Validating

Before starting your validator, please ensure you've populated the empty fields in `.env.validator`, including `WANDB_API_KEY` and `HUGGING_FACE_TOKEN`.

If you haven't already, you can start by copying the template,
```
cp .env.validator.template .env.validator
```

If you don't have a W&B API key, please reach out to the BitMind team via Discord and we can provide one. 

Now you're ready to run your validator!

```bash
conda activate bitmind
./start_validator.sh
```

- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are enabled by default (every 6 hours). To disable, run with `--no-self-heal`.


The above command will kick off 3 `pm2` processes
```
┌────┬───────────────────┬─────────────┬─────────┬─────────┬──────────┬────────┬──────┬───────────┬──────────┬──────────┬──────────┬──────────┐
│ id │ name              │ namespace   │ version │ mode    │ pid      │ uptime │ ↺    │ status    │ cpu      │ mem      │ user     │ watching │
├────┼───────────────────┼─────────────┼─────────┼─────────┼──────────┼────────┼──────┼───────────┼──────────┼──────────┼──────────┼──────────┤
│ 0  │ sn34-generator    │ default     │ N/A     │ fork    │ 2397505  │ 38m    │ 2    │ online    │ 100%     │ 3.0gb    │ user     │ disabled │
│ 2  │ sn34-proxy        │ default     │ N/A     │ fork    │ 2398000  │ 27m    │ 1    │ online    │ 0%       │ 695.2mb  │ user     │ disabled │
│ 1  │ sn34-validator    │ default     │ N/A     │ fork    │ 2394939  │ 108m   │ 0    │ online    │ 0%       │ 5.8gb    │ user     │ disabled │
└────┴───────────────────┴─────────────┴─────────┴─────────┴──────────┴────────┴──────┴───────────┴──────────┴──────────┴──────────┴──────────┘
```
- `sn34-validator` is the validator process
- `sn34-generator` runs our data generation pipeline to produce **synthetic images and videos** (stored in `~/.cache/sn34`)
- `sn34-proxy`routes organic traffic from our applications to miners. 
