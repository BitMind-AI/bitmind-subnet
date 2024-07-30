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

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](./min_compute.yml). A GPU is recommended for training, although not required for basic inference.

### Installation

To install system dependencies like `pm2`, run our install script:

```bash
chmod +x install_system_deps.sh
./install_system_deps.sh
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install), and create a virtual environment with this command:

```bash
conda create -y -n bitmind python=3.10 ipython
```

To activate your virtual environment, run `conda activate bitmind`. To deactivate, `conda deactivate`.

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet
conda activate bitmind
export PIP_NO_CACHE_DIR=1
pip install -e .
```

### Data

If you intend on training a miner, you can download the our open source datasets by running:

```bash
python bitmind/download_data.py
```

- For **miners**, this is only necessary when training a new model. Deployed miner instances do not need access to these datasets.

### Registration

We are currently on testnet. To mine or validate on our subnet, must have a registered hotkey on subnet 168 on testnet.

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

---

## Mining

You can launch your miners via pm2 using the following command. To stop your miner, you can run `pm2 delete miner`.

```bash
pm2 start ./neurons/miner.py --name miner --interpreter $CONDA_PREFIX/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>
```

**Testnet Example**:

```bash
pm2 start ./neurons/miner.py --name miner --interpreter $CONDA_PREFIX/bin/python3 -- --neuron.model_path ./mining_models/miner.pth --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default --axon.port 8091
```

---

## Train

To train a model, you can start with our base training script. If you prefer a notebook environment, you can use `base_miner/train_detector.ipynb`

```python
cd base_miner && python train_detector.py
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

#### Update Miner Detector Model

The most straightforward way to deploy a new miner is by stopping its associated pm2 process (`pm2 delete miner`) and starting it again, setting the `--neuron.model_path` argument appropriately.

Another approach, which avoids miner downtime, is to replace the model file.

1. Optionally make a backup of the currently active model:

   ```bash
   cp mining_models/miner.pth mining_models/miner_backup.pth
   ```

2. Replace the currently active model with your newly trained one. The next forward pass of your miner will load the new model without a restart.

   ```bash
   cp path/to/your/trained/model_epoch_best.pth mining_models/miner.pth
   ```

## Predict

- Prediction logic specific to the trained model your miner is hosting resides in `bitmind/miner/predict.py`
- If you train a custom model, or change the `base_transforms` used in training (defined in `bitmind.image_transforms`) you may need to update `predict.py` accordingly.
- Miners return a single float between 0 and 1, where a value above 0.5 represents a prediction that the image is fake.
- Rewards are based on accuracy. The reward from each challenge is binary.
