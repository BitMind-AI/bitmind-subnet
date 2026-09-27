# Discriminative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with mining operations.

## Discriminative Mining Overview

Submit an image, video, or audio classifier. Evaluation runs on subnet
infrastructure; miners do not need to host inference hardware.

Class order is part of the submission contract:

| Modality | `num_classes` | Logit indices |
| --- | ---: | --- |
| Image | 3 | `0=real`, `1=synthetic`, `2=semisynthetic` |
| Video | 3 | `0=real`, `1=synthetic`, `2=semisynthetic` |
| Audio | 2 | `0=real`, `1=synthetic` |

Non-AI rendering, such as CGI and game footage, belongs to `real` for both image
and video. Audio metadata labeled `semisynthetic` maps to `synthetic`.
See GASBench's [classification and scoring guide](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md)
for class definitions and how probabilities contribute to the score.

## Model Preparation

> **⚠️ Important**: Competition submissions now require **safetensors format**. ONNX is no longer accepted.

Discriminative miners must submit models in **safetensors format**:
- Directory containing: `model_config.yaml`, `model.py`, `*.safetensors`
- ZIP archive of the directory for upload

**📖 [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md)** - Requirements for model submission

Choose one modality per upload:
- `image_detector.zip` - Image classification model
- `video_detector.zip` - Video classification model
- `audio_detector.zip` - Audio classification model

Submissions share the hotkey's allowance across modalities; see [Submission Limits](#submission-limits) for repeat submissions.

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Upload a model using the `push` command:

```bash
gascli d push \
  --image-model image_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Use --video-model or --audio-model for other modalities:
# gascli d push --video-model video_detector.zip --wallet-hotkey video_key
# gascli d push --audio-model audio_detector.zip --wallet-hotkey audio_key
```

### Command Options

The `push` command accepts several parameters:

```bash
gascli d push \
  --image-model image_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name \
  --netuid 34 \
  --chain-endpoint wss://entrypoint-finney.opentensor.ai:443/ \
  --retry-delay 60
```

**Parameters:**
- `--image-model`: Path to image detector zip file
- `--video-model`: Path to video detector zip file
- `--audio-model`: Path to audio detector zip file
- `--wallet-name`: Bittensor wallet name (default: "default")
- `--wallet-hotkey`: Bittensor hotkey name (default: "default") 
- `--netuid`: Subnet UID (default: 34)
- `--chain-endpoint`: Subtensor network endpoint (default: "wss://entrypoint-finney.opentensor.ai:443/")
- `--retry-delay`: Retry delay in seconds (default: 60)

Provide exactly one of `--image-model`, `--video-model`, or `--audio-model`.

## Submission Limits

Each registered hotkey gets **one free counted submission** (image, video, or audio — not one of each).

- Exam failures and incomplete uploads do not consume the slot. You can retry on the same key until a model is successfully uploaded and not later marked exam-failed.
- A confirmed or superseded model **does** consume the free slot for the life of that registration, for every modality.
- A new benchmark version does **not** refill the slot.
- To submit another model from the **same** hotkey, run `gascli d push` again. The CLI walks you through burning **0.5 TAO of SN34 alpha**: `burn_alpha` if this hotkey already has enough α, otherwise one `add_stake_burn` of 0.5 TAO. Recycle does not count. There is no second counted model without that burn.
- The on-chain burn is irreversible. Its **submission credit** is consumed only when the new model counts; unused credits are reused on upload/exam retries.
- Before confirmation, the CLI shows the exact alpha amount and quoted TAO value, including up to **2% price padding** for `burn_alpha` (about 0.51 TAO worth). `add_stake_burn` spends 0.5 TAO. Transaction fees are additional in both cases.
- Before broadcasting, the CLI atomically saves the signed transaction, hash, chain identity, amount, and intended submission in `$GAS_HOME/resubmit_burns` (default `~/.gas/resubmit_burns`). No private keys are stored. A local lock prevents overlapping pushes for the same hotkey/subnet using that state directory.
- After a crash or timeout, rerun the same command on the same machine with the same `GAS_HOME`. The CLI checks finalized chain history and reuses a successful burn. Malformed state, unavailable history, or an unresolved saved transaction blocks a fresh burn; a saved transaction not found even after expiry requires investigation, not automatic repayment. A confirmed on-chain failure permits a newly confirmed attempt on the next run.
- Preserve the journal and receipt. Deleting state, changing `GAS_HOME`, or using another machine bypasses local protection; this is not a cross-machine exactly-once guarantee. If recovery remains blocked, use the printed transaction hash and saved journal to investigate with the operator before retrying elsewhere. Historical burns made before journaling was introduced cannot be recovered automatically without their saved receipt.

---

## Competition Rules and Constraints

### Scoring

`sn34_score` is the competition score. It combines classification performance
(MCC) with probability accuracy (Brier error), using the active round's scoring
mode, dataset weights, and augmentation settings. `benchmark_score` is class
accuracy, which is used for the entrance exam.

See GASBench's [classification and scoring guide](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md)
for the formula and result fields, and [Incentive Mechanism](Incentive.md#discriminator-rewards)
for how the score affects rewards.

### Model Requirements

- **Format**: Safetensors only (ONNX is no longer accepted)
- **Submission allowance**: see [Submission Limits](#submission-limits) for the free allowance and repeat submissions

### Sandbox and Import Restrictions

Your `model.py` is checked by a static analyzer and executed in a sandboxed environment. Key allowed imports include `torch`, `torchvision`, `torchaudio`, `transformers`, `timm`, `einops`, `flash_attn`, `PIL`, `cv2`, `scipy`, `numpy`, and `safetensors`. Network access, system calls, serialization libraries, and dynamic code execution are all blocked.

For the complete list of allowed and blocked imports, see the [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md#allowed-imports).

### Evaluation

- Evaluation runs against a diverse dataset of image samples, video samples, and audio samples per benchmark cycle
- Datasets are refreshed weekly with new GAS-Station data alongside static benchmark datasets

---

## Model Format

For the full model specification including `model_config.yaml` structure, `model.py` requirements, input/output specs per modality, and complete examples, see:

**📖 [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md)**

In short, your submission ZIP must contain:

```
my_detector.zip
├── model_config.yaml    # Metadata and preprocessing config
├── config.json          # (optional) Include if using AutoModel.from_pretrained()
├── model.py             # Model architecture with load_model() function
└── model.safetensors    # Trained weights
```

Package and push:

```bash
cd my_model/
zip -r ../my_detector.zip model_config.yaml model.py model.safetensors
gascli d push --image-model my_detector.zip
```

---

### What Happens During Push

1. **Model Validation**: The system checks that the zip files are present and valid
2. **Model Upload**: Your model zip files are uploaded for evaluation
3. **Blockchain Registration**: Model metadata is registered on the Bittensor blockchain
4. **Verification**: The system verifies the registration was successful

---

## Evaluation Pipeline

After a successful push, your model goes through a two-stage evaluation process automatically.

### Stage 1: Entrance Exam (`--small` mode)

Before your model is ever scored on the network, it must pass an **entrance exam** — a fast sanity check run against a reduced sample of the benchmark datasets.

- The exam uses GASBench `small` mode on datasets selected for the submitted modality and vertical
- The submitted model must achieve **≥ 80% class accuracy** (`benchmark_score`) to pass
- The evaluator enforces a time budget; models that exhaust it fail the exam
- The exam runs in an **isolated sandbox** — your code has no network access and cannot interact with the host environment
- Submissions are statically analyzed and executed in an isolated sandbox; prohibited code or imports result in rejection

**Model status during the exam:**

| Status | Meaning |
|---|---|
| `examining` | Entrance exam is currently running |
| `confirmed` | Exam passed — model is eligible for full benchmarking |
| `exam_failed` | Accuracy below 80% — model will not be scored |
| `blocked` | Cheat pattern detected — model is permanently blocked |

Use `gasbench run --small` as a local preflight before pushing. The hosted exam
also applies its own dataset selection, sandbox, and resource limits:

```bash
gasbench run --image-model ./my_image_model/ --small
gasbench run --video-model ./my_video_model/ --small
gasbench run --audio-model ./my_audio_model/ --small
```

### Stage 2: Full Benchmark (`--full` mode)

Models that pass the entrance exam are evaluated on a larger sample of the
datasets selected for their modality and vertical, including:

- Public benchmark datasets
- **Private holdout datasets** — curated datasets not visible to miners, used to prevent overfitting to the public benchmark set
- Refreshed weekly with new data from the GAS-Station pipeline

The evaluator enforces a time budget for the full run. The resulting
`sn34_score`, including any configured robustness blend, is used in the King of
the Hill competition. See [Incentive Mechanism](Incentive.md#king-of-the-hill) for
challenge margins and emission shares.

For a local image run with multiclass scoring:

```bash
gasbench run --image-model ./my_image_model/ --full --multiclass-scoring
```

Local runs use the public datasets available to you. Their scores are not
directly comparable to a hosted round unless the datasets, sampling, scoring
mode, weights, and augmentation settings match.

### Checking Your Performance

Query your runs from the CLI. Requests are authenticated with your hotkey and
return your own results:

```bash
# View all your benchmark runs
gascli d perf

# Filter by modality or vertical
gascli d perf --modality image
gascli d perf --modality image --vertical human

# Use a specific wallet
gascli d perf --wallet-name miner1 --wallet-hotkey default
```

Each row shows the run ID, status (`queued`/`running`/`success`/`failed`), modality, vertical, SN34 score, MCC, and Brier score. The displayed MCC and Brier fields may be the binary compatibility metrics; `sn34_score` remains the authoritative competition score selected by the round configuration.

### Getting Help

```bash
gascli discriminator --help        # Miner help
gascli d push --help               # Push command help
gascli d perf --help               # Performance query help
```

**Note**: Remember to activate the virtual environment first with `source .venv/bin/activate` before running any `gascli` commands.
