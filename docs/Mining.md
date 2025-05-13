# Miner Setup Guide

## Before you proceed ⚠️

If you are new to Bittensor, we recommend familiarizing yourself with the basics in the [Bittensor Docs](https://docs.bittensor.com/) before proceeding.

**Run your own local subtensor** to avoid rate limits set on public endpoints. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary) for setup instructions.

**Understand your minimum compute requirements** for model training and miner deployment, which varies depending on your choice of model. You will likely need at least a consumer grade GPU for training. Many models can be deploying in CPU-only environments for mining. 


## Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet
```

We recommend using a Conda virtual environment to install the necessary Python packages.
- You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install). 
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

Before you register a miner on testnet or mainnet, you must first fill out all the necessary fields in `.env.miner`. Make a copy of the template, and fill in your wallet and axon information. 

```
cp .env.miner.template .env.miner
```


## Miner Task

### Expected Miner Outputs

> Miners respond to validator queries with a probability vector [$p_{real}$, $p_{synthetic}$, $p_{semisynthetic}$]

Your task as a SN34 miner is to classify images and videos as real, synthetic, or semisynthetic.
- **Real**: Authentic meida, not touched in any way by AI
- **Synthetic**: Fully AI-generated media
- **Semisynthetic**: AI-modified (spatially, not temporally) media. E.g. faceswaps, inpainting, etc.

Minor details:
- You are scored only on correctness, so rounding these probabilities will not give you extra incentive.
- To maximize incentive, you must respond with the multiclass vector described above. 
  - If your classifier returns a binary response (e.g. a float in $[0., 1.]$ or a vector [$p_{real}$, $p_{synthetic}$]), you will earn partial credit (as defined by our incentive mechanism)


### Training your Detector

> [!IMPORTANT]
> The default video and image detection models provided in `neurons/miner.py` serve only to exemplify the desired behavior of the miner neuron, and will not provide competitive performance on mainnet.

#### Model

#### Data


## Registration

To run a miner, you must have a registered hotkey. 

> [!IMPORTANT]
> Registering on a Bittensor subnet burns TAO. To reduce the risk of deregistration due to technical issues or a poor performing model, we recommend the following:
> 1. Test your miner on testnet before you start mining on mainnet.
> 2. Before registering your hotkey on mainnet, make sure your axon port is accepting incoming traffic by running `curl your_ip:your_port`


#### Mainnet

```bash
btcli s register --netuid 34 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

> For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)


```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

#### Mining

You can now launch your miner with `start_miner.sh`, which will use the configuration you provided in `.env.miner` (see the last step of the [Installation](#installation) section). 
