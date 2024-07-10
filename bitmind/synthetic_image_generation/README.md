
# Synthetic Image Generation

This folder contains files for the implementation of a joint vision-to-language and text-to-image model system that generates highly diverse and realistic images for deepfake detector training.

**data/:**

Directory contaning subdirectories of generated annotations (in /annotations_from_real/) and synthetic images (in /synthetics_from_annotations/).

**annotations/:**

Folder containing subdirectories for each real image dataset. Subdirectories contain JSONs with text annotations of images. The filename of the JSONs correspond to the image index in the associated dataset dictionary.

**image_annotation_experiments.ipynb :**

Notebook containing related code snippets for

-real image captioning using BLIP-2

-caption summarization using transformer LLM's


**real_image_to_text_annotation.ipynb :**

Pipeline for real image dataset to text caption dataset generation.

**synthetic_dataset_evaluation.ipynb :**

Evaluation metrics for quantifying realism and diversity of synthetic images.
