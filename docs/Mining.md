# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Data 📊](#data)
   - [Registration ✍️](#registration)
2. [Mining ⛏️](#mining)
3. [Training 🚂](#training)

## Before you proceed ⚠️

**IMPORTANT**: If you are new to Bittensor, we recommend familiarizing yourself with the basics on the [Bittensor Website](https://bittensor.com/) before proceeding.

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is required for training (unless you want to wait weeks for training to complete), but is not required for inference while running a miner.

## Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install). Note that after you run the last commands in the miniconda setup process, you'll be prompted to start a new shell session to complete the initialization. 

With miniconda installed, you can create a virtual environment with this command:

```bash
conda create -y -n bitmind python=3.10 ipython jupyter ipykernel
```

To activate your virtual environment, run `conda activate bitmind`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command. This may take a few minutes to complete.

```bash
conda activate bitmind
export PIP_NO_CACHE_DIR=1
chmod +x setup_env.sh 
./setup_env.sh
```

### Data

*Only for training -- deployed miner instances do not require access to these datasets.*

You can optionally pre-download the training datasets by running:

```bash
python base_miner/datasets/download_data.py
```

Feel free to skip this step - datasets will be downloaded automatically when you run the training scripts.

The default list of datasets and default download location are defined in `base_miner/config.py` 


## Registration

To mine on our subnet, you must have a registered hotkey.

*Note: For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)*

To reduce the risk of deregistration due to technical issues or a poor performing model, we recommend the following:
1. Test your miner on testnet before you start mining on mainnet.
2. Before registering your hotkey on mainnet, make sure your port is open by running `curl your_ip:your_port`
3. If you've trained a custom model, test it's performance by deploying to testnet. You can use this [notebook](https://github.com/BitMind-AI/bitmind-utils/blob/main/wandb_data/wandb_miner_performance.ipynb) to query our tesnet Weights and Biases logs and compute your model's accuracy. Our testnet validator is running 24/7.


#### Mainnet

```bash
btcli s register --netuid 34 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## Mining

You can launch your validator with `run_neuron.py`.

First, make sure to update `validator.env` with your **wallet**, **hotkey**, and **miner port**. This file was created for you during setup, and is not tracked by git.


```bash
IMAGE_DETECTOR=CAMO                            # Options: CAMO, UCF, NPR, None
IMAGE_DETECTOR_CONFIG=camo.yaml                # Configs live in base_miner/deepfake_detectors/configs
                                               # Supply a filename or relative path

VIDEO_DETECTOR=TALL                            # Options: TALL, None
VIDEO_DETECTOR_CONFIG=tall.yaml                # Configs live in base_miner/deepfake_detectors/configs
                                               # Supply a filename or relative path

IMAGE_DETECTOR_DEVICE=cpu                         # Options: cpu, cuda
VIDEO_DETECTOR_DEVICE=cpu

# Subtensor Network Configuration:
NETUID=34                                      # Network User ID options: 34, 168
SUBTENSOR_NETWORK=finney                       # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                               # Endpoints:
                                               # - wss://entrypoint-finney.opentensor.ai:443
                                               # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Miner Settings:
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True          # Default setting to force validator permit for blacklisting
```

Now you're ready to run your miner!

```bash
conda activate bitmind
pm2 start run_neuron.py -- --miner 
```

- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are enabled by default (every 6 hours). To disable, run with `--no-self-heal`.

If you want to outperform the base model, you'll need to train on more data or try experiment with different hyperparameters and model architectures. See our [training](#train) section below for more details.


## Training

To see performance improvements over the base models, you'll need to train on more data, modify hyperparameters, or try a different modeling strategy altogether. Happy experimenting!

*We are working on a unified interface for training models, but for now, each model has its own training script and logging systems that are functionality independent.*

### NPR
```python
cd base_miner/NPR/ && python train_detector.py
```
The model with the lowest validation accuracy will be saved to `base_miner/NPR/checkpoints/<experiment_name>/model_epoch_best.pth`.<br>

### UCF
```python
cd base_miner/DFB/ && python train_detector.py --detector [UCF, TALL] --modality [image, video]
```
The model with the lowest validation accuracy will be saved to `base_miner/UCF/logs/training/<experiment_name>/`.<br>

In this directory, you will find your model weights (`ckpt_best.pth`) and training configuration (`config.yaml`). Note that
the training config, e.g. `config.yaml`, is different from the detector config, e.g. `ucf.yaml`.


## Deploy Your Model

Whether you have trained your own model, designed your own ``DeepfakeDetector`` subclass, or want to deploy a base miner using provided detectors in ``base_miner/deepfake_detectors/``, you can simply update the `miner.env` file to point to the desired detector class and config.

We recommend consulting the `README` in `base_miner/` to learn about the extensibility and modular design of our base miner detectors.

- The detector type (e.g. `UCF`) corresponds to the module name of the ``DeepfakeDetector`` subclass registered in ``base_miner/registry.py``'s ``DETECTOR_REGISTRY``.
- The associated detector config file (e.g., `ucf.yaml`) lives in `base_miner/deepfake_detectors/configs/`.
  - *For UCF only:* You will need to set the `train_config` field in the detector configuration file (`base_miner/deepfake_detectors/configs/ucf.yaml`) to point to the training configuration file. This allows the instantiation of `UCFDetector` to use the settings from training time to reconstruct the correct model architecture. After training a model, the training config can be found in `base_miner/UCF/logs/<your_training_run>/config.yaml`. Feel free to move this to a different location, as long as the `train_config` field in `configs/ucf.yaml` reflects this. 
- The model weights file (e.g., `ckpt_best.pth`) should be placed in `base_miner/<detector_type>/weights`.
  - If the weights specified in the config file do not exist, the miner will attempt to automatically download them from Hugging Face as specified by the `hf_repo` field in the config file. Feel free to use your own Hugging Face repository for hosting your model weights, and update the config file accordingly.



## Tensorboard 

Training metrics are logged with TensorboardX. You can view interactive graphs of these metrics by starting a tensorboard server with the following command, and navigating to `localhost:6006`.

```bash
tensorboard --logdir=./base_miner/checkpoints/<experiment_name>
```

If you're using remote compute for training, you can set up port forwarding by ssh'ing onto your machine with the following flags:

```bash
ssh -L 7007:localhost:6006 your_username@your_ip
```

with port forwarding enabled, you can start your tensorboard server on your remote machine with the following command, and view the tensorboard UI at `localhost:7007` in your local browser.

```bash
tensorboard --logdir=./base_miner/checkpoints/<experiment_name> --host 0.0.0.0 --port 6006
```
