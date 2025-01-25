<p align="center">
  <img src="static/Bitmind-Logo.png" alt="BitMind Logo" width="150"/>
</p>
<h1 align="center">BitMind Subnet<br><small>Bittensor Subnet 34 | Deepfake Detection</small></h1>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


The BitMind Subnet is **the world's first decentralized AI-generated content detection network**. Our incentive mechanism rewards the most accurate detection algorithms, creating an adaptive defense against synthetic media.

## Quick Links
<table>
<tr>
<td width="50%" valign="top" style="border: none !important;">

**Docs**

🧠 [Learn About Bittensor](https://docs.bittensor.com/learn/bittensor-building-blocks)

⛏️ [Mining Guide](docs/Mining.md)

🔧 [Validator Guide](docs/Validating.md)

🏗️ [Architecture Diagrams](#Subnet-Architecture)

📈 [Incentive Mechanism](docs/Incentive.md)


</td>

<td width="50%" valign="top" style="border: none !important;">

**Resources**

🚀 [Applications powered by Subnet 34](https://www.bitmind.ai/apps)

🤗 [BitMind Huggingface](https://huggingface.co/bitmind)

📊 [Mainnet 34 W&B](https://wandb.ai/bitmindai/bitmind-subnet) |  [Testnet 168 W&B](https://wandb.ai/bitmindai/bitmind)

📖 [Project Structure and Terminology](docs/Glossary.md)

🤝 [Contributor Guide](docs/Contributor_Guide.md)


</td>
</tr>
</table>


## Decentralized Detection of AI Generated Content
The explosive growth of generative AI technology has unleashed an unprecedented wave of synthetic media creation. These AI-generated images, videos, and other content have become remarkably sophisticated, making them virtually indistinguishable from authentic media. This development presents a critical challenge to information integrity and societal trust in the digital age, as the line between real and synthetic content continues to blur.

To address this growing challenge, SN34 aims to create the most accurate fully-generalized detection system. Here, fully-generalized means that the system is capable of detecting both synthetic and semi-synthetic media with high degrees of accuracy regardless of their content or what model generated them. Our incentive mechanism evolves alongside state-of-the-art generative AI, rewarding miners whose detection algorithms best adapt to new forms of synthetic content.


## Core Components

> This documentation assumes basic familiarity with Bittensor concepts. For an introduction, please check out the docs: https://docs.bittensor.com/learn/bittensor-building-blocks. 

**Miners** 
- Miners are tasked with running binary classifiers that discern between genuine and AI-generated content, and are rewarded based on their accuracy. 
- Miners predict a float value in [0., 1.], with values greater than 0.5 indicating the image or video is AI generated. 
- Miners are rewarded based on the accuracy of their predictions. For a more detailed breakdown of rewards, weights and incentive, please see our [incentive mechanism docs](docs/Incentive.md)


**Validators** 
- Validators challenge miners with a balanced mix of real and synthetic media drawn from a diverse pool of sources.
- We continually add new datasets and generative models to our validators in order to maximize coverage of the types of diverse data.


## Subnet Architecture


![Subnet Architecture](static/Subnet-Arch.png)

<details>
<summary align=center><i>Figure 1 (above): Ecosystem Overview</i></summary>
<br>

> This diagram provides an overview of the validator neuron, miner neuron, and other components external to the subnet.

- The green arrows show how applications interact with the subnet to provide AI-generated image and video detection functionality.
- The blue arrows show how validators generate synthetic data, challenge miners and score their responses.

</details>

<br>


![Subnet Architecture](static/Vali-Arch.png)

<details>
<summary align=center><i>Figure 2 (above): Validator Components</i></summary>
<br>

> This diagram presents the same architecture as figure 1, but with organic traffic ommitted and with a more detailed look at scored challenges and the associated validator neuron components.


**Challenge Generation and Scoring (Blue Arrows)**

For each challenge, the validator performs the following steps:
1. Randomly samples a real or synthetic image/video from the cache
2. Applies random augmentations to the sampled media
3. Distributes the augmented data to 50 randomly selected miners for classification
4. Updates its score vector based on each miner's historical performance and computed rewards for the current challenge
5. Logs comprehensive challenge results to [Weights and Biases](https://wandb.ai/bitmindai/bitmind-subnet), including the generated media, original prompt, miner responses and rewards, and other challenge metadata

**Synthetic Data Generation (Pink Arrows)**:

The synthetic data generator coordinates a VLM and LLM to generate prompts for our suite of text-to-image, image-to-image, and text-to-video models. Each image or video is written to the cache along with the prompt, generation parameters, and other metadata.

**Dataset Downloads (Green Arrows)**:

The real data fetcher performs partial dataset downloads, fetching random compressed chunks of datasets from HuggingFace and unpacking random portions of these chunks into the cache along with their metadata. Partial downloads avoid requiring TBs of space for large video datasets like OpenVid1M.

</details>



## Community

<p align="left">
  <a href="https://discord.gg/kKQR98CrUn">
    <img src="static/Join-BitMind-Discord.png" alt="Join us on Discord" width="60%">
  </a>
</p>

For real-time discussions, community support, and regular updates, <a href="https://discord.gg/kKQR98CrUn">join our Discord server</a>. Connect with developers, researchers, and users to get the most out of BitMind Subnet.

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
