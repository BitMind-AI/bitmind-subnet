# Bitmind Subnet Incentive Mechanism

This document covers the current state of SN34's incentive mechanism. 
1. [Overview](#overview)
2. [Rewards](#rewards)
3. [Scores](#scores)
4. [Weights](#weights)
5. [Incentive](#incentives)

## Overview

Miner rewards are a weighted combination of their performance on both video and image challenges.

Video and image rewards are computed separately, where each is its own weighted combination of the MCC of their last 100 predictions and the accuracy of their last 10. Validators keep track of miner performance using a score vector, which is updated using an exponential moving average. The weights assigned by validators determine the distribution of rewards among miners, incentivizing high-quality predictions and consistent performance.

Reward implementation can be found in `bitmind/validator/rewards.py`

<p align="center">
  <img src="../static/incentive.gif" alt="Incentive Mechanism">
</p>
<p align="center"><em>Simulation applying our latest iteration of our incentive mechanism on historical subnet data. Note that this graphic shows incentive changes at a much more granular timescale (one timestep per challenge) than that of actual weight setting (once per 360 blocks)<br><a href=https://github.com/BitMind-AI/incentive-simulator>incentive-simulator repository</a>
</em></p>



## Rewards

> Miners rewards are computed based on the [Matthews Correlation Coefficient (MCC)](https://en.wikipedia.org/wiki/Phi_coefficient) of (up to) their last 100 predictions, combined with the accuracy of their last 10 predictions. 

$$ 
0.5 \times MCC_{100} + 0.5 \times Accuracy_{10} 
$$


## Scores

>Validators set weights based on a score vector they use to keep track of miner performance. 

For each challenge *t*, a validator will randomly sample 50 miners, send them an image, and compute their rewards *C* based on their responses. These reward values are then used to update the validator's score vector *V* using an exponential moving average (EMA) with *&alpha;* = 0.02. 

$$
V_t = 0.02 \cdot C_t + 0.98 \cdot V_{t-1}
$$

A low *&alpha;* value places emphasis on a miner's historical performance, adding additional smoothing to avoid having a single prediction cause significant score fluctuations.


## Weights

> Validators set weights around once per tempo (360 blocks) by sending a normalized score vector to the Bittensor blockchain (in `UINT16` representation).

Weight normalization by L1 norm:

$$w = \frac{\text{V}}{\lVert\text{V}\rVert_1}$$


## Incentives
> The [Yuma Consensus algorithm](https://docs.bittensor.com/yuma-consensus) translates the weight matrix *W* into incentives for the subnet miners and dividends for the subnet validators

Specifically, for each miner *j*, incentive is a function of rank *R*:

$$I_j = \frac{R_j}{\sum_k R_k}$$

where rank *R* is *W* (a matrix of validator weight vectors) weighted by validator stake vector *S*. 

$$R_k = \sum_i S_i \cdot W_{ik}$$




