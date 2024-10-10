# Bitmind Subnet Incentive Mechanism

This document covers the current state of SN34's incentive mechanism. 
1. [Overview](#overview)
2. [Rewards](#rewards)
3. [Scores](#scores)
4. [Weights](#weights)
5. [Incentive](#incentives)

## Overview

Miners are rewarded based on their performance, which is measured by the squared ROC AUC of their last 100 predictions. Validators keep track of miner performance using a score vector, which is updated using an exponential moving average. The weights assigned by validators determine the distribution of rewards among miners, incentivizing high-quality predictions and consistent performance.

<p align="center">
  <img src="../static/incentive.gif" alt="Incentive Mechanism">
</p>
<p align="center"><em>Simulation applying our latest iteration of our incentive mechanism on historical subnet data. Note that this graphic shows incentive changes at a much more granular timescale (one timestep per challenge) than that of actual weight setting (once per 360 blocks)<br><a href=https://github.com/BitMind-AI/incentive-simulator>incentive-simulator repository</a>
</em></p>



## Rewards

> Miners rewards are computed as the squared [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) of (up to) their last 100 predictions. 


$$
r_i = \text{AUC}_i^2
$$

$$
\text{AUC} = \int_{0}^{1} \text{TPR}(t) \cdot \frac{d}{dt}\text{FPR}(t) dt
$$

where *t* is the classification threshold and

$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

and

$$\text{ FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$


ROC AUC, or Receiver Operating Characteristic Area Under the Curve, is a powerful metric for evaluating binary classification models. It was chosen for its following strenghts:

- Robustness to class imbalance: performs well even when one class is more common than the other (to avoid volatilitiy in cases where miners receieve a series of all fake or all real images).
- Interpretability: provides a single scalar value between 0 and 1, where 1 indicates perfect classification.
- Sensitivity and specificity balance: considers both true positive and false positive rates.

These characteristics make ROC AUC an ideal metric for evaluating miner performance in our subnet, where accurate classification of real and synthetic images is crucial.



## Scores

>Validators set weights based on a score vector they use to keep track of miner performance. 

For each challenge *t*, a validator will randomly sample 50 miners, send them an image, and compute their rewards *C* based on their responses. These reward values are then used to update the validator's score vector *V* using an exponential moving average (EMA) with *&alpha;* = 0.01. 

$$
V_t = 0.01 \cdot C_t + 0.99 \cdot V_{t-1}
$$

A low *&alpha;* value places emphasis on a miner's historical performance, adding additional smoothing to avoid having a single prediction cause significant score fluctuations.


## Weights

> Validators set weights around once per tempo (360 blocks) by sending a normalized score vector to the Bittensor blockchain (in `UINT16` representation).

Weight normalization by L1 norm:

$$w = \frac{\text{Score}}{\lVert\text{Score}\rVert_1}$$


## Incentives
> The [Yuma Consensus algorithm](https://docs.bittensor.com/yuma-consensus) translates the weight matrix *W* into incentives for the subnet miners and dividends for the subnet validators

Specifically, for each miner *j*, incentive is a function of rank *R*:

$$I_j = \frac{R_j}{\sum_k R_k}$$

where rank *R* is *W* (a matrix of validator weight vectors) weighted by validator stake vector *S*. 

$$R_k = \sum_i S_i \cdot W_{ik}$$




