
# Synthetic Image Generation

This folder contains files for the implementation of a joint vision-to-language and text-to-image model system that generates highly diverse and realistic images for deepfake detector training.

**test_data/:**

Default output directory for real-image-to-annotation and annotation-to-synthetic-image pipelines in the associated notebooks.

**real_image_to_text_annotation.ipynb :**

Pipeline for real image dataset to text caption dataset generation. Contains function that generates subdirectories of annotations for each real image dataset. Annotations are formatted as JSONs with captions (Strings) of images. The filename of the JSONs correspond to the image index in the associated dataset dictionary.

**text_annotation_to_synthetic_image.ipynb :**

Pipeline for text annotation to synthetic image dataset generation.
