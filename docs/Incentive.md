# Incentive Mechanism

## Benchmark Runs
Submitted discriminator miners are evaluated against a subset of the data sources listed below. Models are evaluated on cloud infrastructure -- miners do not need to host hardware for inference. A portion of the evaluation data comes from generative miners, who are rewarded based on their ability to submit data that both pass validator sanity checks (prompt alignment, etc.) and fool discriminators in benchmark runs.

Each modality (image, video, audio) is scored independently using the `sn34_score` metric, which combines discrimination accuracy (MCC) with calibration quality (Brier score).

<details>
<summary><strong>Evaluation Datasets</strong></summary>

Benchmark datasets are regularly expanded. The current evaluation set includes ~50 image datasets, ~40 video datasets, and ~25 audio datasets drawn from the sources below.

### Image Datasets (~50,000 sample benchmark)

**Real Images:**
- [gasstation/gs-images-v3](https://huggingface.co/datasets/gasstation/gs-images-v3) (GAS-Station generated - continuously updated)
- [drawthingsai/megalith-10m](https://huggingface.co/datasets/drawthingsai/megalith-10m)
- [bitmind/bm-eidon-image](https://huggingface.co/datasets/bitmind/bm-eidon-image)
- [bitmind/open-image-v7-256](https://huggingface.co/datasets/bitmind/open-image-v7-256)
- [bitmind/celeb-a-hq](https://huggingface.co/datasets/bitmind/celeb-a-hq)
- [bitmind/ffhq-256](https://huggingface.co/datasets/bitmind/ffhq-256)
- [bitmind/MS-COCO-unique-256](https://huggingface.co/datasets/bitmind/MS-COCO-unique-256)
- [bitmind/AFHQ](https://huggingface.co/datasets/bitmind/AFHQ)
- [bitmind/lfw](https://huggingface.co/datasets/bitmind/lfw)
- [bitmind/caltech-256](https://huggingface.co/datasets/bitmind/caltech-256)
- [bitmind/caltech-101](https://huggingface.co/datasets/bitmind/caltech-101)
- [bitmind/dtd](https://huggingface.co/datasets/bitmind/dtd)
- [bitmind/idoc-mugshots-images](https://huggingface.co/datasets/bitmind/idoc-mugshots-images)
- [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)
- [detection-datasets/fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia)
- [allenai/CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K)
- [CausalLM/Retrievatar](https://huggingface.co/datasets/CausalLM/Retrievatar)
- [YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision)
- [ethz/food101](https://huggingface.co/datasets/ethz/food101)
- [MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU)
- [FreedomIntelligence/PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)
- [MMMGBench/MMMG](https://huggingface.co/datasets/MMMGBench/MMMG)
- [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart)
- [OpenDriveLab/OpenScene](https://huggingface.co/datasets/OpenDriveLab/OpenScene)
- [bitmind/FakeClue](https://huggingface.co/datasets/bitmind/FakeClue) (real subsets: doc, ff++, genimage, satellite)

**Synthetic Images:**
- [bitmind/JourneyDB](https://huggingface.co/datasets/bitmind/JourneyDB)
- [bitmind/GenImage_MidJourney](https://huggingface.co/datasets/bitmind/GenImage_MidJourney)
- [bitmind/bm-aura-imagegen](https://huggingface.co/datasets/bitmind/bm-aura-imagegen)
- [bitmind/bm-imagine](https://huggingface.co/datasets/bitmind/bm-imagine)
- [bitmind/ideogram-27k](https://huggingface.co/datasets/bitmind/ideogram-27k)
- [bitmind/Dalle-3-1M](https://huggingface.co/datasets/bitmind/Dalle-3-1M)
- [bitmind/SyntheticFacesHQ](https://huggingface.co/datasets/bitmind/SyntheticFacesHQ) (parts 1-4)
- [bitmind/Deepfake-leonardo-stablecog](https://huggingface.co/datasets/bitmind/Deepfake-leonardo-stablecog)
- [bitmind/klingai-images](https://huggingface.co/datasets/bitmind/klingai-images)
- [bitmind/FakeClue](https://huggingface.co/datasets/bitmind/FakeClue) (fake subsets: chameleon, doc, ff++, genimage, satellite)
- [Andrew613/PICA-100K](https://huggingface.co/datasets/Andrew613/PICA-100K)
- [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M)
- [bitmind/Nano-banana-150k](https://huggingface.co/datasets/bitmind/Nano-banana-150k)
- [Rapidata/bananamark-dataset](https://huggingface.co/datasets/Rapidata/bananamark-dataset)
- [cactuslab/IDNet-2025](https://huggingface.co/datasets/cactuslab/IDNet-2025)

**Semi-synthetic Images:**
- [bitmind/face-swap](https://huggingface.co/datasets/bitmind/face-swap)

### Video Datasets (~20,000 sample benchmark)

**Real Videos:**
- [bitmind/bm-eidon-video](https://huggingface.co/datasets/bitmind/bm-eidon-video)
- [shangxd/imagenet-vidvrd](https://huggingface.co/datasets/shangxd/imagenet-vidvrd)
- [facebook/PE-Video](https://huggingface.co/datasets/facebook/PE-Video)
- [faridlab/deepaction_v1](https://huggingface.co/datasets/faridlab/deepaction_v1) (Pexels subset)
- [Hemgg/deep-fake-detection-dfd-entire-original-datasett](https://huggingface.co/datasets/Hemgg/deep-fake-detection-dfd-entire-original-datasett) (original sequences)
- [Hemgg/Physics-101-Video-dataset](https://huggingface.co/datasets/Hemgg/Physics-101-Video-dataset)
- [Pai3dot14/Moments_in_Time_Raw_50k](https://huggingface.co/datasets/Pai3dot14/Moments_in_Time_Raw_50k)
- [bitmind/UCF101Fullvideo](https://huggingface.co/datasets/bitmind/UCF101Fullvideo)
- [yeray142/first-impressions-v2](https://huggingface.co/datasets/yeray142/first-impressions-v2)
- [SushantGautam/SoccerNet-10s-5Class](https://huggingface.co/datasets/SushantGautam/SoccerNet-10s-5Class)
- [HopLeeTop/Sports-QA](https://huggingface.co/datasets/HopLeeTop/Sports-QA)
- [builddotai/Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K)
- [OpenDriveLab/FreeTacMan](https://huggingface.co/datasets/OpenDriveLab/FreeTacMan)
- [USC-GVL/humanoid-everyday](https://huggingface.co/datasets/USC-GVL/humanoid-everyday)
- [34data/workout-vids](https://huggingface.co/datasets/34data/workout-vids)
- [cz-5f/LoVoRA](https://huggingface.co/datasets/cz-5f/LoVoRA) (real subset)

**Synthetic Videos:**
- [gasstation/gs-videos-v3](https://huggingface.co/datasets/gasstation/gs-videos-v3) (GAS-Station generated - continuously updated)
- [Rapidata/text-2-video-human-preferences-veo3](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-veo3)
- [Rapidata/text-2-video-human-preferences-veo2](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-veo2)
- [Rapidata/text-2-video-human-preferences-wan2.1](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-wan2.1)
- [bitmind/aura-video](https://huggingface.co/datasets/bitmind/aura-video)
- [bitmind/aislop-videos](https://huggingface.co/datasets/bitmind/aislop-videos)
- [bitmind/klingai-videos](https://huggingface.co/datasets/bitmind/klingai-videos)
- [bitmind/VidProM](https://huggingface.co/datasets/bitmind/VidProM)
- [faridlab/deepaction_v1](https://huggingface.co/datasets/faridlab/deepaction_v1) (BDAnimateDiffLightning, CogVideoX5B, RunwayML, StableDiffusion, Veo, VideoPoet subsets)
- [hi-paris/FakeParts](https://huggingface.co/datasets/hi-paris/FakeParts) (T2V subset)
- [saiyan-world/Goku-MovieGenBench](https://huggingface.co/datasets/saiyan-world/Goku-MovieGenBench)
- [kevinzzz8866/ByteDance_Synthetic_Videos](https://huggingface.co/datasets/kevinzzz8866/ByteDance_Synthetic_Videos)
- [BianYx/VAP-Data](https://huggingface.co/datasets/BianYx/VAP-Data)
- [WenhaoWang/VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO)
- [chungimungi/VideoDPO-10k](https://huggingface.co/datasets/chungimungi/VideoDPO-10k)
- [aadityaubhat/synthetic-emotions](https://huggingface.co/datasets/aadityaubhat/synthetic-emotions)
- [sophiaa/Open-VFX](https://huggingface.co/datasets/sophiaa/Open-VFX)
- [cz-5f/LoVoRA](https://huggingface.co/datasets/cz-5f/LoVoRA) (synthetic subset)
- [SENORITADATASET/Senorita](https://huggingface.co/datasets/SENORITADATASET/Senorita) (controllable videos, inpainting, local style transfer, obj removal, obj swap, style transfer)

**Semi-synthetic Videos:**
- [bitmind/semisynthetic-video](https://huggingface.co/datasets/bitmind/semisynthetic-video)
- [hi-paris/FakeParts](https://huggingface.co/datasets/hi-paris/FakeParts) (Faceswap, Inpainting, Outpainting, Change of style subsets)
- [Hemgg/deep-fake-detection-dfd-entire-original-datasett](https://huggingface.co/datasets/Hemgg/deep-fake-detection-dfd-entire-original-datasett) (manipulated sequences)

### Audio Datasets (~30,000 sample benchmark)

**Real Audio:**
- [fixie-ai/common_voice_17_0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) (Mozilla Common Voice 17.0 - 100+ languages)
- [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) (EU Parliament speeches - 18 languages)
- [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) (AMI meeting corpus)
- [facebook/omnilingual-asr-corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) (1600+ languages)
- [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech) (10K hours English)
- [MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
- [parler-tts/mls_eng_10k](https://huggingface.co/datasets/parler-tts/mls_eng_10k) (MLS English 10K subset)
- [ylacombe/english_dialects](https://huggingface.co/datasets/ylacombe/english_dialects)
- [myleslinder/crema-d](https://huggingface.co/datasets/myleslinder/crema-d) (CREMA-D emotional speech)
- [CSALT/deepfake_detection_dataset_urdu](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu) (real subset)
- [language-and-voice-lab/samromur_children](https://huggingface.co/datasets/language-and-voice-lab/samromur_children)
- [language-and-voice-lab/raddromur_asr](https://huggingface.co/datasets/language-and-voice-lab/raddromur_asr)
- [qmeeus/slurp](https://huggingface.co/datasets/qmeeus/slurp)
- [ymoslem/MediaSpeech](https://huggingface.co/datasets/ymoslem/MediaSpeech)
- [simon3000/genshin-voice](https://huggingface.co/datasets/simon3000/genshin-voice)

**Synthetic Audio:**
- [DeepFake-Audio-Rangers/Arabic_Audio_Deepfake](https://huggingface.co/datasets/DeepFake-Audio-Rangers/Arabic_Audio_Deepfake)
- [CSALT/deepfake_detection_dataset_urdu](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu) (spoofed subset)
- [yhaha/EmoVoice-DB](https://huggingface.co/datasets/yhaha/EmoVoice-DB) (synthetic emotional voices)
- [tutu0604/UltraVoice](https://huggingface.co/datasets/tutu0604/UltraVoice) (CJK TTS voices)
- [unfake/fake_voices](https://huggingface.co/datasets/unfake/fake_voices)
- [skypro1111/elevenlabs_dataset](https://huggingface.co/datasets/skypro1111/elevenlabs_dataset)
- [velocity-engg/eleven_labs_dataset](https://huggingface.co/datasets/velocity-engg/eleven_labs_dataset)
- [Sh1man/elevenlabs](https://huggingface.co/datasets/Sh1man/elevenlabs)
- [NeoBoy/elevenlabsSpeechTest](https://huggingface.co/datasets/NeoBoy/elevenlabsSpeechTest)
- [velocity-engg/eleven_labs_datase_latin](https://huggingface.co/datasets/velocity-engg/eleven_labs_datase_latin)
- [Thorsten-Voice/TV-44kHz-Full](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full) (German TTS)
- [ash56/ShiftySpeech](https://huggingface.co/datasets/ash56/ShiftySpeech) (7 domains, 6 TTS systems, 12 vocoders, 3 languages)

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

The generator incentive mechanism combines two components: a base reward for passing data validation checks, and a multiplier based on adversarial performance against discriminators.

### Base Reward (Data Validation)

Generators receive a base reward based on their data verification pass rate:

$$R_{\text{base}} = p \cdot \min(n, 10)$$

Where:
- $p$ = pass rate (proportion of generated content that passes validation)
- $n$ = number of verified samples (`min(n, 10)` creates a rampup of incentive for the first 10 samples)

### Fool Rate Multiplier (Adversarial Performance)

Generators earn additional rewards by successfully fooling discriminators. The multiplier is calculated as:

$$M = \max(0, \min(2.0, f \cdot s))$$

Where:
- $f$ = fool rate = $\frac{N_{\text{fooled}}}{N_{\text{fooled}} + N_{\text{not fooled}}}$
- $s$ = sample size multiplier

The sample size multiplier encourages generators to be evaluated on more samples, similar to the sample size ramp used in the base reward.

$$s = \begin{cases}
\max(0.5, \frac{c}{20}) & \text{if } c < 20 \\
\min(2.0, 1.0 + \ln(\frac{c}{20})) & \text{if } c \geq 20
\end{cases}$$

Where:
- $c$ = total evaluation count (fooled + not fooled)
- Reference count of 20 gives multiplier of 1.0
- Sample sizes below 20 are penalized
- Sample sizes above 20 receive logarithmic bonus up to 2.0x

### Final Generator Reward

The total generator reward combines both components:

$$R_{\text{total}} = R_{\text{base}} \cdot M$$

This design incentivizes generators to:
1. Produce high-quality, valid content (base reward)
2. Create adversarially robust content that can fool discriminators (multiplier)
3. Participate in more evaluations for sample size bonuses



## Discriminator Rewards

### Scoring: `sn34_score`

Each discriminator model is scored per modality using the `sn34_score`, which combines two metrics:

1. **Binary MCC (Matthews Correlation Coefficient)** -- measures how well the model discriminates between real and synthetic content. Ranges from -1 (worst) to +1 (perfect).

2. **Brier Score** -- measures calibration quality (how well predicted probabilities match actual outcomes). Ranges from 0 (perfect) to 0.25 (random baseline).

These are combined as follows:

$$\text{mcc\_norm} = \left(\frac{\text{MCC} + 1}{2}\right)^{\alpha}$$

$$\text{brier\_score} = \left(\frac{0.25 - \text{Brier}}{0.25}\right)^{\beta}$$

$$\text{sn34\_score} = \sqrt{\text{mcc\_norm} \cdot \text{brier\_score}}$$

With default parameters $\alpha = 1.2$ and $\beta = 1.8$. The geometric mean penalizes models that are strong on one axis but weak on the other -- a model must be both accurate *and* well-calibrated to score highly.

### Competition Rounds

The discriminator competition is organized into **rounds**. Each round introduces new benchmark datasets and evaluates all submitted models. Winners are determined **per modality** (image, video, audio) independently.

#### How Rounds Work

1. **New round begins**: Benchmark datasets are updated (new GAS-Station data, potentially new static datasets). All modalities share the same benchmark version number.
2. **Models are benchmarked**: All submitted discriminator models are evaluated against the current round's datasets and scored using `sn34_score`.
3. **Winner determined per modality**: The highest-scoring model for each modality wins that round.
4. **Alpha reward**: The round winner for each modality receives an alpha reward.

#### Winner-Take-All Per Round

Each round is winner-take-all -- only the top-scoring discriminator for each modality receives the alpha reward for that round. This incentivizes miners to continuously improve their models and push the state of the art in AI-generated content detection.

Rounds progress as benchmark versions are incremented, ensuring that models are always evaluated against fresh, evolving data.
