# Discriminative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with mining operations.

## Discriminative Mining Overview

- Miners submit media-provenance classifiers across three modalities: **image**, **video**, and **audio**.
- Image models classify `[real, synthetic, semisynthetic]`; video models classify `[real, synthetic, semisynthetic, rendered]`; audio remains `[real, synthetic]`.
- The visual taxonomy is experimental. Semisynthetic media retains materially captured visual content alongside spatially localized generated or replaced content. Fully synthesized output remains synthetic even when captured media conditions generation.
- Models are evaluated on cloud infrastructure -- miners do not need to host hardware for inference.

Class order is part of the submission contract:

| Modality | `num_classes` | Logit indices |
| --- | ---: | --- |
| Image | 3 | `0=real`, `1=synthetic`, `2=semisynthetic` |
| Video | 4 | `0=real`, `1=synthetic`, `2=semisynthetic`, `3=rendered` |
| Audio | 2 | `0=real`, `1=synthetic` |

See GASBench's [Classification Taxonomy and Scoring](https://github.com/BitMind-AI/gasbench/blob/main/docs/Classification-and-Scoring.md) for the normative class definitions, metric formulas, and binary compatibility collapse.

## Model Preparation

> **⚠️ Important**: Competition submissions now require **safetensors format**. ONNX is no longer accepted.

Discriminative miners must submit models in **safetensors format**:
- Directory containing: `model_config.yaml`, `model.py`, `*.safetensors`
- ZIP archive of the directory for upload

**📖 [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md)** - Requirements for model submission

You can submit models for any combination of modalities:
- `image_detector.zip` - Image classification model
- `video_detector.zip` - Video classification model  
- `audio_detector.zip` - Audio classification model

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Push your models to the network using the `push` command:

```bash
# Upload all three models
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload individual models
gascli d push \
  --image-model image_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name
```

### Command Options

The `push` command accepts several parameters:

```bash
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
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

At least one model (image, video, or audio) must be provided.

## Submission Limits

Each registered hotkey gets **one counted submission** (image, video, or audio — not one of each).

- Exam failures and incomplete uploads do not consume the slot. You can retry on the same key until a model is successfully uploaded and not later marked exam-failed.
- A confirmed or superseded model **does** consume the slot for the life of that hotkey, for every modality.
- A new benchmark version does **not** refill the slot.
- To submit another model, register a new miner hotkey.

---

## Competition Rules and Constraints

### Scoring

Each model is scored per modality using `sn34_score`, a geometric mean of normalized MCC and Brier performance:

$$sn34_{score} = \sqrt{M_{norm} \cdot B_{norm}}$$

The normalized terms apply exponents $1.2$ to MCC performance and $1.8$ to Brier performance. Image and video currently use multiclass Gorodkin MCC and multiclass Brier, rewarding correct distinctions between provenance classes. Audio uses the equivalent two-class calculation. Binary real-versus-not-real metrics are still reported for compatibility and diagnosis. See [Incentive Mechanism](Incentive.md) for the full formula.

### Model Requirements

- **Format**: Safetensors only (ONNX is no longer accepted)
- **Submission cap**: one counted model per hotkey (see [Submission Limits](#submission-limits))

### Sandbox and Import Restrictions

Your `model.py` is checked by a static analyzer and executed in a sandboxed environment. Key allowed imports include `torch`, `torchvision`, `torchaudio`, `transformers`, `timm`, `einops`, `flash_attn`, `PIL`, `cv2`, `scipy`, `numpy`, and `safetensors`. Network access, system calls, serialization libraries, and dynamic code execution are all blocked.

For the complete list of allowed and blocked imports, see the [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md#allowed-imports).

### Evaluation

- Models are benchmarked on cloud infrastructure (not miner hardware)
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
2. **Model Upload**: Your model zip files are uploaded to the cloud inference system
3. **Blockchain Registration**: Model metadata is registered on the Bittensor blockchain
4. **Verification**: The system verifies the registration was successful

---

## Evaluation Pipeline

After a successful push, your model goes through a two-stage evaluation process automatically.

### Stage 1: Entrance Exam (`--small` mode)

Before your model is ever scored on the network, it must pass an **entrance exam** — a fast sanity check run against a reduced sample of the benchmark datasets.

- Internally this runs `gasbench run --small`, which downloads one archive per dataset and evaluates roughly 100 samples per dataset
- Your model must achieve **≥ 80% accuracy** averaged across all submitted modalities to pass
- The exam has a **maximum wall-clock timeout of 1 hour 25 minutes** (5,100 seconds); models that exceed this are treated as failed
- The exam runs in an **isolated cloud sandbox** — your code has no network access and cannot interact with the host environment
- Submissions are statically analyzed and executed in an isolated sandbox; prohibited code or imports result in rejection

**Model status during the exam:**

| Status | Meaning |
|---|---|
| `examining` | Entrance exam is currently running |
| `confirmed` | Exam passed — model is eligible for full benchmarking |
| `exam_failed` | Accuracy below 80% — model will not be scored |
| `blocked` | Cheat pattern detected — model is permanently blocked |

You can use `gasbench run --small` locally to replicate exam conditions before pushing:

```bash
gasbench run --image-model ./my_image_model/ --small
gasbench run --video-model ./my_video_model/ --small
gasbench run --audio-model ./my_audio_model/ --small
```

### Stage 2: Full Benchmark (`--full` mode)

Models that pass the entrance exam are benchmarked against the **complete dataset suite**, which includes:

- All public benchmark datasets across image, video, and audio modalities
- **Private holdout datasets** — curated datasets not visible to miners, used to prevent overfitting to the public benchmark set
- Refreshed weekly with new data from the GAS-Station pipeline

The full benchmark has a **maximum wall-clock timeout of 5 hours** (18,000 seconds) per modality. This `sn34_score` is what the King of the Hill competition uses: a high enough score can take or keep a lane, and emissions then follow the 85/10/5 split on the current king plus the previous two. The active round configuration selects provenance weighting, multiclass scoring, and augmentation robustness parameters; see [Incentive Mechanism](Incentive.md).

You can simulate a full benchmark run locally (without holdouts) to get a sense of your model's performance:

```bash
gasbench run --image-model ./my_image_model/ --full
```

> **Note**: Local full runs will not include the private holdout datasets used in the actual network evaluation.

### Checking Your Performance

Once your model has been benchmarked, you can query your scores directly from the CLI. This is authenticated with your hotkey so you can only see your own results — including the active round that isn't shown on the public leaderboard.

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
