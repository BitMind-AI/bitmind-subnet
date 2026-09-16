# Incentive Mechanism

## Benchmark Runs
Submitted discriminator miners are evaluated against a subset of the data sources listed below. Miners do not need to host hardware for inference. A portion of the evaluation data comes from generative miners, who are rewarded based on their ability to submit data that both pass validator sanity checks (prompt alignment, etc.) and fool discriminators in benchmark runs.

Each modality (image, video, audio) is scored independently using the `sn34_score` metric, which combines classification performance (MCC) with probability calibration (Brier score). The active round selects binary or multiclass scoring per modality.

<details>
<summary><strong>Evaluation Datasets</strong></summary>

Benchmark datasets are regularly expanded. Image uses real, synthetic, and semisynthetic classes; video additionally includes rendered media; audio remains binary. The experimental visual taxonomy is defined in GASBench's [Classification Taxonomy and Scoring](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md). Datasets include continuously updated [GAS-Station](https://huggingface.co/gasstation) data from generative miners.

**Public datasets** (available for training via gasbench):
- **Image**: [`image_datasets.yaml`](https://github.com/BitMind-AI/gasbench/blob/main/src/gasbench/dataset/configs/image_datasets.yaml)
- **Video**: [`video_datasets.yaml`](https://github.com/BitMind-AI/gasbench/blob/main/src/gasbench/dataset/configs/video_datasets.yaml)
- **Audio**: [`audio_datasets.yaml`](https://github.com/BitMind-AI/gasbench/blob/main/src/gasbench/dataset/configs/audio_datasets.yaml)

**Holdout datasets**: In addition to the public datasets above, each benchmark round includes holdout datasets that are not publicly available during the round. Holdout data is critical to ensure models generalize well and to mitigate overfitting. At the end of each round, many of the holdout datasets are released and added to the public gasbench datasets for future training. Some holdout datasets cannot be released publicly due to licensing or other restrictions.

</details>

<details>
<summary><strong>Generative Models</strong></summary>

The following models are run by validators to produce a continual, fresh stream of synthetic and semisynthetic data. The outputs of these models are uploaded at regular intervals to public datasets in the [GAS-Station](https://huggingface.co/gasstation) Hugging Face org for miner training and evaluation.

### Text-to-Image Models

- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SG161222/RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)
- [Corcelio/mobius](https://huggingface.co/Corcelio/mobius)
- [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4)
- [cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) + [Kvikontent/midjourney-v6](https://huggingface.co/Kvikontent/midjourney-v6) LoRA
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [DeepFloyd/IF](https://huggingface.co/DeepFloyd/IF)
- [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
- [THUDM/CogView4-6B](https://huggingface.co/THUDM/CogView4-6B)

### Image-to-Image Models

- [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [Lykon/dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting)

### Text-to-Video Models

- [tencent/HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)
- [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
- [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- [ByteDance/AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)

### Image-to-Video Models

- [THUDM/CogVideoX1.5-5B-I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) 

</details> 


## Generator Rewards

The 16% generator pot is split among UIDs with $R > 0$ this tempo (`score_i / \sum score`). A miner earns only in modalities that clear a 7-day fool-rate gate. There is no fool-rate bonus on top of base rewards.

### Base reward (per modality)

Over the last 24 hours of verified submissions on this validator:

$$R_{\text{image}} = p_{\text{image}} \cdot v(n_{\text{image}}) \cdot m_{\text{image}}$$

and likewise for video. $p$ is the verification pass rate. Volume $v(n)$ ramps linearly through the first 10 verified samples, then $\log_2$. $m$ is the mean $\sqrt{\text{price}/\text{baseline}}$ model (and resolution-tier) multiplier; see [Model Pricing](Generative-Mining.md#model-pricing-and-rewards).

### Qualification gate

Fool rate is sample-weighted over the last 7 days of **benchmark evals** (`fooled + not_fooled` from `generator_result_benchmark`), not challenges answered. A modality qualifies when $n \ge 20$ and the rate is strictly above the cutoff:

- Image: fool rate $> 2\%$
- Video: fool rate $> 1\%$

A miner can qualify in one modality, both, or neither. Pay only in the cleared modality:

$$R = 0.30 \cdot R_{\text{image}} \cdot I_{\text{image}} + 0.70 \cdot R_{\text{video}} \cdot I_{\text{video}}$$

$I=1$ if that modality is qualified, else $0$. If both are $0$, the miner gets none of the 16%. Scores are still zeroed after 24 hours of inactivity.

If the generator-results API is down or the payload has no usable rows, validators keep the last successful qualification map for pay so a transient outage does not burn the pot. This map is saved alongside modality EMA histories and restored after a restart. Restored qualification is payout fallback only: challenge sampling remains all-onboarding until a successful API fetch. Older snapshots without a qualification map need one successful fetch before this fallback is available.

Qualification is cached by hotkey and resolved against current UID registrations for rewards and challenge sampling. A replacement hotkey cannot inherit the previous owner's eligibility at the same UID.

Scores use separate image and video exponential moving averages (50% current reward, 50% history), combined with the same 30/70 weights. Losing qualification immediately clears that modality's history, including epochs where nobody earns; regaining qualification starts that lane from zero. The histories are stored by hotkey across restarts and cleared on inactivity or deregistration. On upgrade, legacy combined scores reset because their modality contributions cannot be recovered. A UID still needs positive current gated base rewards to share the generator pot.

### Challenge slots

Each validator still sends `--neuron.sample-size` (default 50) requests per round — one UID, one modality, no replacement. Slots are filled from three buckets **for the chosen modality**:

| Bucket | Who | Default slots |
|---|---|---|
| Qualified | Over the bar for that modality | 36 |
| Onboarding | $n < 20$ or no fool-rate row | 8 |
| Probe | $n \ge 20$ but under the bar | 6 |
| Unresponsive | This validator asked enough times and got no answer | 0 |

Onboarding and probe miners still receive prompts; they do not earn until they clear. If the onboarding set is empty, the unused 8 slots split 4+4 (40 qualified / 10 probe). Leftover slots overflow qualified → probe → qualified, then any remaining live generator who still answers that modality, so the round never goes out under-filled when miners exist. A missing or stale generator-results cache treats everyone as onboarding so sampling does not freeze on the last qualified set.

Unresponsive is local to each validator: refused challenge POSTs (`no_answer`) and accepted tasks that never deliver (`challenge_timeout`). After `--scoring.min-no-answer-attempts` (default 5) in `--scoring.no-answer-lookback-hours` (default 24) with zero answers in that modality, the miner is skipped for that modality. A video-only miner who ignores image is not image-onboarding. One real answer (media or a miner-reported failure) clears the flag.

This design incentivizes generators to:
1. Produce valid, C2PA-signed content (base reward)
2. Clear the fool-rate bar instead of farming extra UIDs (qualification)
3. Use models whose price and quality justify the 30/70 image/video split



## Discriminator Rewards

### Scoring: `sn34_score`

Each discriminator model is scored per modality using two components:

1. **MCC** measures classification quality. Binary mode uses ordinary MCC after collapsing every non-real class into synthetic. Multiclass mode uses Gorodkin's $R_K$, the multiclass generalization of MCC.
2. **Brier score** measures calibration. Binary mode uses the mean squared error of $p_{\text{not real}}$, whose constant-guess baseline is $0.25$. Multiclass mode uses the mean of $\sum_k(p_k-y_k)^2$, whose uniform-guess baseline for $K$ classes is $B_0=(K-1)/K$.

For the selected mode, let $M$ be MCC, $B$ be Brier score, and $B_0$ be the corresponding random baseline:

$$M_{norm} = \operatorname{clip}\left(\frac{M+1}{2},0,1\right)^{1.2}$$

$$B_{norm} = \max\left(0,\frac{B_0-B}{B_0}\right)^{1.8}$$

$$sn34_{score} = \sqrt{M_{norm} \cdot B_{norm}}$$

Image and video currently use multiclass scoring. Audio uses binary scoring; with two classes, the normalized multiclass calculation is mathematically identical. Every run also reports `binary_sn34_score` and `multiclass_sn34_score` so the two views can be compared.

### Dataset composition and augmentation robustness

The round configuration assigns target score shares to public, private holdout, and GAS-Station samples. Those shares are converted into per-sample weights and applied consistently to accuracy, MCC, Brier, cross-entropy, and the resulting SN34 score.

When the robustness pass is enabled, the final score is:

$$sn34_{final} = (1-w)\,sn34_{base} + w\,sn34_{aug}$$

The benchmark records `base_sn34_score`, `aug_sn34_score`, and robustness diagnostics. Exact composition shares, augmentation sample counts, and $w$ are round configuration, so they may change between benchmark versions rather than being permanent protocol constants.

The normative implementation details and complete metric field glossary live in GASBench's [Classification Taxonomy and Scoring](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md).

### King of the Hill

Discriminator emission is King of the Hill. Each modality has one reigning model. Validators set that lane's weight on registered hotkeys every tempo — not on an escrow wallet. Each hotkey may land **one counted submission** for the life of that registration (any modality; exam failures do not count; a new model needs a new key).

Current split:

- Image lane: 40%
- Video lane: 40%
- Audio lane: 4%
- Generators: 16%

Each discriminator lane is split **85 / 10 / 5** across the current king and the previous two **distinct** crowned hotkeys. If a lane has no previous king, that residual rolls up to the current king (a first king receives the full lane). An unresolvable current king burns its share; an unresolvable previous king rolls to the current king when that UID is registered.

A challenger takes the crown when it posts an `sn34_score` at least **0.01** higher than the sitting king on the **same** `CURRENT_BENCHMARK_VERSION`. Empty-lane seeding and failed-defense replacement do not use the margin. The same `file_hash` can refresh its stored score without resetting the reign.

When a new benchmark version is released the current king keeps receiving weights. The throne is marked `defending` until that exact model completes a full re-eval on the new version. Dethroning is frozen during defense. After a successful defense, deferred challengers still need the 0.01 margin. If the re-eval fails or times out (48 hours), the crown goes to the best successful new-version model, or that lane's share burns until one exists.

Alpha accrues on the chain hotkeys while they hold those residual shares. There is no end-of-round escrow transfer and no pot.
