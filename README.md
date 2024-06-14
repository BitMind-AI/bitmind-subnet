<center>
    <img src="static/Bitmind-Logo.png" alt="BitMind Logo" width="200"/>
</center>

# AI Image Detection Subnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

- [Introduction](#introduction)
- [Setup](#setup)
- [Mining](#mining)
- [Validating](#validating)
- [License](#license)

## Introduction

**IMPORTANT**: If you are new to Bittensor, we suggest you get comfortable with the basics at the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section.

Introducing Bittensor Subnet X (AI Image Detection Subnet): A Platform for Identifying AI Generated Media.

The recent proliferation of generative models capable of creating high qualitiy, convincingly realistic images has brought with it an urgent need for reliable mechanisms for distinguishing between real and fake images. This is a Bittensor subnet that incentivizes innovation in such mechanisms. The quality and reliability of the AI Image Detection Subnet are inherently tied to the incentivized, decentralized nature of Bittensor - decentralization mitigates the centralization risk of single-model approaches in this problem space, and incentivazation facilitates innovation within a rich, competitive ecosystem. Our easy-to-use API and frontend (COMING SOON) democratize access to the collective intelligence of our miner pool, realizing a powerful tool that will help alleviate the issues of misinformation and deception that now pervade modern media and threaten deomocracy at a global scale. 


<center>
    <img src="static/Subnet-Arch.png" alt="Subnet Architecture"/>
</center>


The AI Generated Image Detection Subnet is composed of a suite of state-of-the-art generative and discriminative AI models, and will continuously evolve to cover more generative algorithms. 

- **Miners** are tasked with running a binary classifier capable of discriminating between real and AI generated images
    - Our base miner is from the 2024 CVPR Paper [*Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*](https://arxiv.org/abs/2312.10461), which introduces a novel metric called Neighborhood Pixel Relationships to guide the training of popular Convolutional Neural Networks (CNNs) to learn features specific to artifacts present in generated images.
    - In the interest of helping our miners experiment with diverse cutting edge strategies, will continue implementing more detection solutions based on recent papers, and release training code and base weights.
- **Validators** are tasked with sending images to miners for classification, with each challenge having a 50/50 chance of containing a real or fake image. Validators run a prompt generation LLM and several image generation models, and sample real images from a pool composed of over 10 million images from several open source datasets.
    - We will iteratively expand the generative capabilities of validators, as well as the real image sample pool, to increase miner competition and, in turn, the utility of the subnet as a consumer-facing service.

Join us at the AI Image Detection Subnet, your partner in maintaining digital authenticity and leading the fight against misinformation. Be part of the solution and stay at the forefront of innovation with our cutting-edge detection tools – Defending Truth, Empowering the Future!

## Status

We are on testnet, uid 168!


## Setup

### Before you proceed

Before running a validator or miner, note the following:

**IMPORTANT**: We **strongly recommend** before proceeding that you ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary)

**IMPORTANT:** Make sure you are aware of the minimum compute requirements for our subnet. See the [Minimum compute YAML configuration](./min_compute.yml).

### Installation

If you don't have them, install `git`, `pip` and a text editor like `nano` or `emacs` if you don't like `vi`
```bash
sudo apt update -y && sudo apt-get install git -y && sudo apt install python3-pip -y && sudo apt install nano
```

Install `pm2` to use our scripts for running miners and validators.
```bash
sudo apt install npm -y && sudo npm install pm2@latest -g 
```

We recommend you use a virtual environment to install the necessary requirements.<br>
We like conda. You can get it with this [super quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install), and use it to create a virtual environment like this:
```
conda create -n fakedet python=3.10 ipython
conda activate fakedet
```

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet
conda activate fakedet
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

To mine or validate on our subnet, ensure you have a registered hotkey on subnet 168 on testnet. If not, run the command

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Train

To see performance improvements, you'll need to train on more data, modify hyperparameters, or try a different modeling strategy altogether.

To train a model, run the following command. The model with the lowest validation accuracy will be saved to `base_miner/checkpoints/<experiment_name>/model_epoch_best.pth`. 
```python
cd base_miner && python train_detector.py
```
If you prefer a notebook environment, you can instead train a model with `base_miner/train_detector.ipynb`

Once you've trained your model, you can evaluate its performance on the test dataset in `base_miner/eval_detector.ipynb`.


#### Update Miner Detector Model
1. Optionally make a backup of the currently active model:
   ```bash
   cp mining_models/miner.pth mining_models/miner_<version>.pth
   ```
2. Replace the currently active model with your newly trained one. The next forward pass of your miner will load the new model.
   ```bash
   cp path/to/your/trained/model_epoch_best.pth mining_models/miner.pth
   ```

## Predict
- Prediction logic specific to the trained model your miner is hosting resides in `bitmind/miner/predict.py`
- If you train a custom model, or change the `base_transforms` used in training (defined in `bitmind.image_transforms`) you may need to update `predict.py` accordingly.
- Miners return a single float between 0 and 1, where a value above 0.5 represents a prediction that the image is fake.
- Rewards are based on accuracy. The reward from each challenge is binary.
  

## Mine

You can launch your miners via pm2 using the following command.

```bash
pm2 start ./neurons/miner.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>
```

### Testnet Example

```bash
pm2 start ./neurons/miner.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default --axon.port 8091
```


## Validate

You can launch your validator via pm2 using the following command.

```bash
pm2 start ./neurons/validator.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

### Testnet Example

```bash
pm2 start ./neurons/validator.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default
```

---

### Other Resources
[Join our Discord!](https://discord.gg/SaFbkGkU)


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
