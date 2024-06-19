<center>
    <img src="static/Bitmind-Logo.png" alt="BitMind Logo" width="200"/>
</center>

# Bitmind Subnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

- [Introduction](#introduction)
- [Setup](#setup)
- [Mining](#mining)
- [Validating](#validating)
- [License](#license)

## Introduction

**IMPORTANT**: If you are new to Bittensor, we suggest you get comfortable with the basics at the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section.

Introducing Bittensor Subnet X (Bitmind Subnet): A Platform for Identifying AI Generated Media.

The recent proliferation of generative models capable of creating high qualitiy, convincingly realistic images has brought with it an urgent need for reliable mechanisms for distinguishing between real and fake images. This is a Bittensor subnet that incentivizes innovation in such mechanisms. The quality and reliability of the Bitmind Subnet are inherently tied to the incentivized, decentralized nature of Bittensor - decentralization mitigates the centralization risk of single-model approaches in this problem space, and incentivazation facilitates innovation within a rich, competitive ecosystem. Our easy-to-use API and frontend (COMING SOON) democratize access to the collective intelligence of our miner pool, realizing a powerful tool that will help alleviate the issues of misinformation and deception that now pervade modern media and threaten deomocracy at a global scale.

The Bitmind Subnet comprises a suite of state-of-the-art generative and discriminative AI models, and will continually evolve to cover more generative algorithms.

- **Miners** are tasked with running a binary classifier capable of discriminating between real and AI generated images
    - Our base miner is from the 2024 CVPR Paper [*Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*](https://arxiv.org/abs/2312.10461), which introduces a novel metric called Neighborhood Pixel Relationships to guide the training of popular Convolutional Neural Networks (CNNs) to learn features specific to artifacts present in generated images.
    - In the interest of helping our miners experiment with diverse cutting edge strategies, will continue implementing detection solutions based on recent papers, and release training code and base weights.
- **Validators** are tasked with sending images to miners for classification, with each challenge having a 50/50 chance of containing a real or fake image. Validators run a prompt generation LLM and several image generation models, and sample real images from a pool composed of over 10 million images from several open source datasets.
    - We will iteratively expand the generative capabilities of validators, as well as the real image sample pool, to increase miner competition and, in turn, the utility of the subnet as a consumer-facing service.


<center>
    <img src="static/Subnet-Arch.png" alt="Subnet Architecture"/>
</center>


## Status

We are on testnet, uid 168!

[Join our Discord!](https://discord.gg/SaFbkGkU)

---

## Setup

### Before you proceed

Before running a validator or miner, note the following:

**IMPORTANT**: We **strongly recommend** before proceeding that you ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary)

**IMPORTANT:** Make sure you are aware of the minimum compute requirements for our subnet. See the [Minimum compute YAML configuration](./min_compute.yml). Our base miner model does not require a GPU for inference, but we strongly recommend using a GPU for training.

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

You can download the necessary datasets by running:

```bash
python download_data.py
```

- For **validators**, we recommend you do this prior to registering and running your validator. The download can take up to a few hours. Please note the minimum storage requirements specified in `min_compute.yml`. 

- For **miners**, this is only necessary when training a new model. Deployed miner instances do not need access to these datasets. 


### Registration

We are currenlty on testnet. To mine or validate on our subnet, must have a registered hotkey on subnet 168 on testnet.

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

---

## Mine

You can launch your miners via pm2 using the following command. To stop your miner, you can run `pm2 delete miner`.

```bash
pm2 start ./neurons/miner.py --name miner --interpreter $CONDA_PREFIX/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>
```

**Testnet Example**:

```bash
pm2 start ./neurons/miner.py --name miner --interpreter $CONDA_PREFIX/bin/python3 -- --neuron.model_path ./mining_models/miner.pth --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default --axon.port 8091
```


## Validate

You can launch your validator via pm2 using the following command. To stop your validator, you can run `pm2 delete validator`.

```bash
pm2 start ./neurons/validator.py --name validator --interpreter $CONDA_PREFIX/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

**Testnet Example**:

```bash
pm2 start ./neurons/validator.py --name validator --interpreter  $CONDA_PREFIX/bin/python3 -- --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default
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


#### Tensorboard
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

Another approach, which avoids avoids miner downtime, is to replace the model file.

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



---

## License

This repository is licensed under the MIT License.

```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
