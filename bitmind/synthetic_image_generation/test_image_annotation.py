from bitmind.image_dataset import ImageDataset
from bitmind.constants import DATASET_META
from bitmind.constants import IMAGE_ANNOTATION_MODEL
from image_annotation_generator import ImageAnnotationGenerator
import numpy as np
import random

def test_dataset_annotation(generator, datasets):
    annotations, avg_latency = generator.generate_annotations(
        real_image_datasets=datasets,
        verbose=2,
        max_images=5,
        resize_images=True
    )

    print("Generated Annotations:", annotations)
    print("Average Latency:", avg_latency)


def test_single_image_annotation(generator, image_info, dataset_name, image_index=0, resize_images=False, verbose=0):
    print("Dataset name: ", dataset_name)
    print("Image info: ", image_info)
    annotation, time_elapsed = generator.process_image(
        image_info, dataset_name, image_index, resize_images, verbose)
    print("Annotation: ", annotation)
    print("Time Elapsed: ", time_elapsed)

def sample_with_index(dataset, k=1):
    """ Sample images along with their indices from the dataset. """
    indices = random.sample(range(len(dataset)), k)
    return [(dataset[i], i) for i in indices]


if __name__ == "__main__":
    generator = ImageAnnotationGenerator(model_name=IMAGE_ANNOTATION_MODEL)
    real_image_datasets = [
        ImageDataset(ds['path'], 'test', ds.get('name', None), ds['create_splits'])
        for ds in DATASET_META['real']
    ]
    dataset = real_image_datasets[np.random.randint(0, len(real_image_datasets))]
    dataset_name = dataset.huggingface_dataset_path
    sample, image_index = sample_with_index(dataset, k=1)[0]
    test_single_image_annotation(generator, sample, dataset_name, image_index=image_index)
    # test_dataset_annotation(generator, real_image_datasets)
