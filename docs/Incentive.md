# Incentive Mechanism

## Benchmark Runs

Submitted classifiers are evaluated separately for image, video, and audio.
Miners do not need to host inference hardware. The competition uses each run's
final `sn34_score`, including any configured augmentation blend.

Image and video use `[real, synthetic, semisynthetic]`; audio uses
`[real, synthetic]`. Non-AI rendered media belongs to `real`. See GASBench's
[classification and scoring guide](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md)
for the class definitions and scoring rules.

The [GASBench dataset registry](https://github.com/BitMind-AI/gasbench/tree/main/src/gasbench/dataset/configs)
lists public datasets. Full evaluations also use private holdouts to measure
generalization, alongside fresh [GAS-Station](https://huggingface.co/gasstation)
data from generative miners. Holdouts may be released for future training when
licensing permits.

For generation services and model choices, see the
[Generative Mining Guide](Generative-Mining.md#generation-services).

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

Each validator still sends `--neuron.sample-size` (default 50) requests per round — one UID, one modality, no replacement.

The default `--neuron.challenge-allocation random` draws those UIDs uniformly from every registered generator and assigns image or video at random. Qualification, onboarding, and unresponsive status still gate **pay**; they do not change who gets asked.

`--neuron.challenge-allocation buckets` is the older slot fill. Slots are filled from three buckets **for the chosen modality**:

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

Each evaluation pass combines MCC (classification performance) and Brier error
(probability accuracy). The round configuration selects binary or multiclass
scoring and target weights for public, holdout, and GAS-Station samples.

When an augmentation pass contributes to the score:

```text
sn34_score = (1 - aug_weight) * base_sn34_score + aug_weight * aug_sn34_score
```

Use the final `sn34_score` for competition comparisons. `benchmark_score` is
accuracy, while `binary_sn34_score` and `multiclass_sn34_score` describe the base
pass. The round configuration controls the scoring mode, dataset shares,
augmentation sample count, and blend weight.

GASBench's [classification and scoring guide](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md)
defines the score calculation and result fields.

### King of the Hill

Each modality has one reigning model. Validators assign that lane's emissions
to registered hotkeys every tempo. See [Submission Limits](Discriminative-Mining.md#submission-limits)
for the free submission allowance and the same-hotkey resubmission process.

Current split:

- Image lane: 40%
- Video lane: 40%
- Audio lane: 4%
- Generators: 16%

Each discriminator lane is split **85 / 10 / 5** across the current king and the previous two **distinct** crowned hotkeys. If a lane has no previous king, that residual rolls up to the current king (a first king receives the full lane). An unresolvable current king burns its share; an unresolvable previous king rolls to the current king when that UID is registered.

A challenger takes the crown when it posts an `sn34_score` at least **0.01** higher than the sitting king on the **same** `CURRENT_BENCHMARK_VERSION`. Empty-lane seeding and failed-defense replacement do not use the margin. The same `file_hash` can refresh its stored score without resetting the reign.

When a new benchmark version is released the current king keeps receiving weights. The throne is marked `defending` until that exact model completes a full re-eval on the new version. Dethroning is frozen during defense. After a successful defense, deferred challengers still need the 0.01 margin. If the re-eval fails or times out (48 hours), the crown goes to the best successful new-version model, or that lane's share burns until one exists.

Alpha accrues on the chain hotkeys while they hold those residual shares. There is no end-of-round escrow transfer and no pot.
