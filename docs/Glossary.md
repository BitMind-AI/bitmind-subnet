# Project Structure and Terminology

## Table of Contents

1. [Overview and Terminology 📖](#overview-and-terminology)
2. [Notable Directories 📁](#notable-directories)
3. [Key Files and Descriptions 🗂️](#key-files-and-descriptions)
      - [bitmind/base/ 🔧](#bitmindbase)
      - [bitmind/validator/ 🛡️](#bitmindvalidator)
      - [bitmind/miner/ ⛏️](#bitmindminer)
4. [Datasets 📊](#datasets)
5. [Additional Tools 🧰](#additional-tools)

### Overview and Terminology

Before diving into the specifics of the directory structure and key components, let's familiarize ourselves with the essential terms used throughout this project. Understanding these terms is important for navigating and contributing to the BitMind Subnet effectively. For a more detailed explanation of the terminology, please refer to [Bittensor Building Blocks](https://docs.bittensor.com/learn/bittensor-building-blocks).

- **Synapse**: Acts as a communication bridge between axons (servers) and dendrites (clients), facilitating data flow and processing.
- **Neuron**: A fundamental unit that includes both an axon and a dendrite, enabling full participation in the network operations.

### Notable Directories

- **bitmind/**: This directory contains the specific implementations of Bittensor operations, which include the key components such as miners, validators, and neurons. This code is used both by validators/miners as well as the base_miner training/eval code.
  - **base/**: Houses base classes for miner, validator, and neuron functionalities, each inheriting from the broader Bittensor framework. 

### Key Files and Descriptions

#### bitmind/base/
- **miner.py**: Responsible for loading models and weights, and handling predictions on images.
- **validator.py**: Implements core functionality for generating challenges for miners, scoring responses, and setting weights.
- **neuron.py**: A class that inherits from the base miner class provided by Bittensor, incorporating both axon and dendrite functionalities.

#### bitmind/validator/
- **forward.py**: Manages image processing and synapse operations using `ImageSynapse` for 256x256 images. Includes logic for challenge issuance and reward updates based on performance.
- **proxy.py**: Temporarily unused; intended for handling frontend requests.

#### bitmind/miner/
- **predict.py**: Handles image transformation and the execution of model inference.

### Datasets

- **real_fake_dataset**: Utilized by the base miner for training, distinguishing between real and fake images.

### Additional Tools

- **random_image_generator.py**: A class that uses a prompt generation model and a suite of diffusion models to produce synthetic images. Supports caching of image/prompt pairs to a local directory.