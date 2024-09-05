# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Data 📊](#data)
   - [Registration ✍️](#registration)
2. [Mining ⛏️](#mining)
3. [Train 🚂](#train)
   - [Tensorboard 📈](#tensorboard)
   - [Update Miner Detector Model 🔄](#update-miner-detector-model)
4. [Predict 🔮](#predict)

### Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is recommended for training, although not required for basic inference.

### Installation

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
chmod +x setup_miner_env.sh 
./setup_miner_env.sh
```

### Data

If you intend on training a miner, you can download the our open source datasets by running:

```bash
python bitmind/download_data.py
```

- For **miners**, this is only necessary when training a new model. Deployed miner instances do not need access to these datasets.

### Registration

To mine on our subnet, you must have a registered hotkey.

*Note: For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)*


#### Mainnet

```bash
btcli s register --netuid 34 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

---

## Mining

You can launch your miner with `run_neuron.py`.

First, make sure to update `miner.env` with your **wallet**, **hotkey**, and **miner port**. This file was created for you during setup, and is not tracked by git.

```bash
MODEL_PATH=./mining_models/base.pth
NETUID=34 # or 168 
SUBTENSOR_NETWORK=finney # or test
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443 # or wss://test.finney.opentensor.ai:443/
WALLET_NAME=default
WALLET_HOTKEY=default
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True
```

Now you're ready to run your miner!

```bash
conda activate bitmind
pm2 start run_neuron.py -- --miner 
```

- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are enabled by default (every 6 hours). To disable, run with `--no-self-heal`.


### Bring Your Own Model
If you want to outperform the base model, you'll need to train on more data or try experiment with different model architectures. 

- If you want to deploy a model you trained with your base miner code, you can simply update `MODEL_PATH` in `miner.env` to point to your new `.pth` file
- If you try a different model architecture (which we encourage!), you'll also need to make the appropriate updates to `neurons/miner.py` and `bitmind/miner/predict.py` so that your miner can properly load and predict with your model.
---

## Train

To train a model, you can start with our base training script. If you prefer a notebook environment, you can use `base_miner/NPR/train_detector.ipynb`

```python
cd base_miner/NPR/ && python train_detector.py
```

- The model with the lowest validation accuracy will be saved to `base_miner/checkpoints/<experiment_name>/model_epoch_best.pth`.<br>
- Once you've trained your model, you can evaluate its performance and inspect its predictions in `base_miner/eval_detector.ipynb`.<br>
- To see performance improvements, you'll need to train on more data, modify hyperparameters, or try a different modeling strategy altogether. Happy experimenting!

### Tensorboard 

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

## Predict

- Prediction logic specific to the trained model your miner is hosting resides in `bitmind/miner/predict.py`
- If you train a custom model, or change the `base_transforms` used in training (defined in `bitmind.image_transforms`) you may need to update `predict.py` accordingly.
- Miners return a single float between 0 and 1, where a value above 0.5 represents a prediction that the image is fake.
- Rewards are based on accuracy. The reward from each challenge is binary.
