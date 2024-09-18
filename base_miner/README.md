## Base Miners

The `base_miner/` directory facilitates the training, orchestration, and deployment of modular and highly customizable deepfake detectors.


## Directory Structure

### 1. Architectures and Training
- **UCF/** and **NPR/**

These folders contain model architectures and training loops for `UCF (ICCV 2023)` and `NPR (CVPR 2024)`, adapted to use curated and preprocessed training datasets on our [BitMind Huggingface](https://huggingface.co/bitmind).

### 2. deepfake_detectors/
The modular structure for detectors used in the miner neuron is defined here, through `DeepfakeDetector` abstract base class and subclass implementations.

- **deepfake_detectors/** contains:
  - **configs/**: YAML configuration files to load detector instance attributes, including any pretrained model weights.
  - **Abstract Base Class**: A foundational class that outlines the standard structure for implementing detectors.
  - **Detector Subclasses**: Specialized detector implementations that can be dynamically loaded and managed based on configuration.

The `DeepfakeDetector design` allows for high configurability and extension.

### 3. gating_mechanisms/
Similar to `deepfake_detectors/`, this folder contains abstract base classes and implementations of `Gate`s that are used to handle content-aware preprocessing and routing. This is especially useful for multi-agent detection systems, such as the `DeepfakeDetector` subclass `CAMODetector` in `deepfake_detectors/camo_detector.py`*.

- **Abstract Gate Class**: A base class for implementing image content gating.
- **Gate Subclasses**: These subclasses define specific gating mechanisms responsible for routing inputs to appropriate expert detectors or preprocessing steps based on content characteristics. This is useful for multi-detector or mixture-of-expert detector setups.

### 4. registry.py
The `registry.py` file is responsible for managing the creation of detectors and gates using a **Factory Method** design pattern. It auto-registers all `DeepfakeDetector` and `Gate` subclasses from their subfolders to respective `Registry` constants, making it simple to instantiate detectors and gates dynamically based on predefined constants.

- **Factory Pattern**: Ensures a clean, maintainable, and scalable method for creating instances of detectors and gating mechanisms.
- **Auto-Registration**: Automatically registers all available detector and gate subclasses, enabling a flexible and extensible system.

## Integration with `miner.py`

- **Modular Initialization**: The miner neuron in `bitmind-subnet/neurons/miner.py` leverages the registry system to dynamically initialize the detector used for the forward function, facilitating a highly modular design. The detector module used is determined by neuron config args, defaulting to `"CAMO"`.

## *CAMO: Content-Aware Model Orchestration

Read about [CAMO (Content Aware Model Orchestration)](https://bitmindlabs.notion.site/CAMO-Content-Aware-Model-Orchestration-CAMO-Framework-for-Deepfake-Detection-43ef46a0f9de403abec7a577a45cd075), our generalized framework for creating “hard mixture of expert” models for deepfake detection. The latest and most performant iteration of our CAMO miner neuron uses finetuned expert and generalist UCF models.

- **Latest Iteration**: The most performant iteration of `class CAMODetector(DeepfakeDetector)` incorporates a `GatingMechanism(Gate)` that routes to fine-tuned expert and generalist models from the `UCF` architecture, for more robust detection.