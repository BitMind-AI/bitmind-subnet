# Bitmind Subnet Incentive Mechanism

This document covers the current state of SN34's incentive mechanism. 
1. [Overview](#overview)
2. [Miner Types](#miner-types)
3. [Rewards](#rewards)
4. [Scores](#scores)
5. [Weights](#weights)
6. [Incentive](#incentives)

## TLDR

SN34 supports two miner types with different reward mechanisms:

- **DETECTOR miners**: Rewards based on classification performance using Matthews Correlation Coefficient (MCC) across image and video challenges
- **SEGMENTER miners**: Rewards based on segmentation performance using Intersection over Union (IoU) on image challenges

Validators keep track of miner performance using a score vector, which is updated using an exponential moving average. These scores are used by validators to set weights for miners, which determine their reward distribution, incentivizing high-quality predictions and consistent performance.

## Miner Types

SN34 operates with two distinct miner types, each with specialized scoring mechanisms:

### DETECTOR Miners (Classification)
- **Task**: Classify media as real, synthetic, or semi-synthetic
- **Modalities**: Images and videos
- **Scoring**: Matthews Correlation Coefficient (MCC) for binary and multiclass classification
- **Reward Range**: 0.0 to 1.0 based on classification accuracy

### SEGMENTER Miners (Segmentation)
- **Task**: Identify AI-generated regions within images
- **Modalities**: Images only (video segmentation not yet supported)
- **Scoring**: Intersection over Union (IoU) against ground truth masks
- **Reward Range**: 0.0 to 1.0 based on segmentation precision

## Rewards

### DETECTOR Miner Rewards
>A DETECTOR miner's total reward $C$ combines their performance across both image and video challenges, weighted by configurable parameters $p$ that controls the emphasis placed on each modality.

$$
C = \sum_{m \in \{image, video\}} p_m \sum_{k \in \{b,m\}} w_k MCC_k
$$

The reward for each modality $m$ is a weighted combination of binary and multiclass ($b$ and $m$) Matthews Correlation Coefficient (MCC) scores. The weights $w_k$ allow emphasis to be shifted as needed between the binary distinction between synthetic and authentic, and the more granular separation of fully- and semi-synthetic content.

### SEGMENTER Miner Rewards
>A SEGMENTER miner's reward is based on their segmentation performance using Intersection over Union (IoU):

$$
C = \text{IoU} = \frac{|\text{Prediction} \cap \text{Ground Truth}|}{|\text{Prediction} \cup \text{Ground Truth}|}
$$

Where:
- **Prediction**: Binary mask derived from the miner's confidence mask (thresholded at 0.5)
- **Ground Truth**: True AI-generated regions provided by validators

The final reward is computed as the average IoU over the miner's recent challenges (up to 200 most recent).

## Scores

>Validators set weights based on historical miner performances, tracked by their score vector. 

For each challenge *t*, a validator will randomly sample 50 miners of the appropriate type, send them media, and compute their rewards *C* as described above. These reward values are then used to update the validator's score vector *V* using an exponential moving average (EMA) with *&alpha;* = 0.02. 

$$
V_t = 0.02 \cdot C_t + 0.98 \cdot V_{t-1}
$$

A low *&alpha;* value places emphasis on a miner's historical performance, adding additional smoothing to avoid having a single prediction cause significant score fluctuations.

**Note**: Scores are normalized separately for each miner type to ensure fair competition between DETECTOR and SEGMENTER miners.

## Weights

> Validators set weights around once per tempo (360 blocks) by sending a normalized score vector to the Bittensor blockchain (in `UINT16` representation).

Weight normalization by L1 norm, applied separately for each miner type:

$$w = \frac{\text{V}}{\lVert\text{V}\rVert_1}$$

This ensures that DETECTOR and SEGMENTER miners compete within their respective categories while maintaining overall subnet balance.

## Incentives
> The [Yuma Consensus algorithm](https://docs.bittensor.com/yuma-consensus) translates the weight matrix *W* into incentives for the subnet miners and dividends for the subnet validators

Specifically, for each miner *j*, incentive is a function of rank *R*:

$$I_j = \frac{R_j}{\sum_k R_k}$$

where rank *R* is *W* (a matrix of validator weight vectors) weighted by validator stake vector *S*. 

$$R_k = \sum_i S_i \cdot W_{ik}$$

The dual miner type system allows for specialized incentives:
- **DETECTOR miners** are incentivized for accurate classification across multiple modalities
- **SEGMENTER miners** are incentivized for precise pixel-level segmentation
- Both contribute to the overall goal of detecting AI-generated content through different approaches




