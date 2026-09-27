# Mining Guide

GAS supports two types of miners that work together in an adversarial loop:

## [Discriminative Mining](Discriminative-Mining.md) 📖
Miners submit **image, video, or audio** classifiers for hosted evaluation. The
competition uses `sn34_score`, which combines classification performance and
probability accuracy with the configured robustness blend. See
[Discriminative Mining](Discriminative-Mining.md) for model preparation and
[submission limits](Discriminative-Mining.md#submission-limits).

## [Generative Mining](Generative-Mining.md) 🎨
Miners create synthetic media (images and videos) that challenges the discriminators. They generate increasingly realistic content to test and improve detection capabilities, and are rewarded for verified volume in modalities that clear the 7-day fool-rate gate.

---

**Choose your path above to get started with mining on GAS.** 