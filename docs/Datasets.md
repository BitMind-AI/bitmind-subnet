# Datasets README

This document provides an overview of the datasets used within the BitMind Subnet, including descriptions and links to their sources and access points on Hugging Face. Each dataset is under its own license, and users are encouraged to review these licenses before use. 

Details on decentralized data storage access points, which will serve as another way to facilitate the utilization of our datasets, will be provided soon. Stay tuned for updates on how to access these resources efficiently and securely.

## Dataset Categories

### Third-Party Real Image Datasets

These datasets consist of authentic images sourced from various real-world scenarios. They are necessary for training our models to recognize genuine content.

#### Dataset Name
**Description**: Brief description of what the dataset includes and its purpose.
**Hugging Face Link**: [Dataset on Hugging Face](#)
**Original Source**: [Dataset Source](#)

### Third-Party Synthetic Image Datasets

These datasets contain images that are artificially created to simulate different imaging conditions and scenarios, aiding in model training against synthetic manipulations.

#### Dataset Name
**Description**: Brief description of what the dataset includes and its purpose.
**Hugging Face Link**: [Dataset on Hugging Face](#)
**Original Source**: [Dataset Source](#)

### Synthetic Datasets Generated via Image-to-Text Annotation Models

These datasets are created by annotating a third-party real image dataset using an image-to-text annotation model like BLIP-2, followed by generating corresponding images through a diffusion model. This approach ensures that the synthetic data mirrors the distribution of our real image datasets 1 to 1, providing a balanced training ground for our models.

#### Dataset Name
**Description**: Brief description of how the dataset was created and its purpose.
**Hugging Face Link**: [Dataset on Hugging Face](#)
**Original Source**: [Dataset Source](#)

## Usage Guidelines

Please ensure to adhere to the licensing agreements specified for each dataset. These licenses dictate how the datasets can be used, shared, and modified. For detailed licensing information, refer to the respective dataset links provided.

For any further questions or clarifications, please contact the project maintainers or visit our [community Discord](https://discord.gg/bitmind).
