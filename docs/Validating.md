# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Data 📊](#data)
   - [Registration ✍️](#registration)
2. [Validating ✅](#validating)

### Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is recommended for training, although not required for basic inference.

### Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet
```

To install system dependencies like `pm2`, run our install script:

```bash
chmod +x install_system_deps.sh
./install_system_deps.sh
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install), and create a virtual environment with this command:

```bash
conda create -y -n bitmind python=3.10 ipython jupyter ipykernel
```

To activate your virtual environment, run `conda activate bitmind`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command.

```bash
conda activate bitmind
export PIP_NO_CACHE_DIR=1
pip install -e .
```

### Data

You can download the necessary datasets by running:

```bash
python bitmind/download_data.py
```

- For **validators**, we recommend you do this prior to registering and running your validator. The download can take up to a few hours. Please note the minimum storage requirements specified in `min_compute.yml`.

- For **miners**, this is only necessary when training a new model. Deployed miner instances do not need access to these datasets.

### Registration

To validate on our subnet, must have a registered hotkey.

#### Mainnet

```bash
btcli s register --netuid 34 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

---

## Validating

You can launch your validator with `run_neuron.py`.

First, make sure to update `start_mainnet_validator.sh` or `start_testnet_validator.sh` with your **wallet name**, **hotkey name**, and your **validator port**. 

- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are enabled by default (every 6 hours). To disable, run with `--no-self-heal`.

```bash
conda activate bitmind
pm2 start run_neuron.py -- --validator --network finney
```

**Testnet Example**:

```bash
conda activate bitmind
pm2 start run_neuron.py -- --validator --network test
```

---
