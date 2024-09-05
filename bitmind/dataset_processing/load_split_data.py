from typing import List, Tuple, Dict
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

from bitmind.real_fake_dataset import RealFakeDataset
from bitmind.image_dataset import ImageDataset
from bitmind.constants import DATASET_META

def load_and_split_datasets(dataset_meta: list) -> Dict[str, List[ImageDataset]]:
    """
    Helper function to load and split dataset into train, validation, and test sets.

    Args:
        dataset_meta: List containing metadata about the dataset to load.

    Returns:
        A dictionary with keys == "train", "validation", or "test" strings,
        and values == List[ImageDataset].

        Dict[str, List[ImageDataset]]

        e.g. given two dataset paths in dataset_meta,
        {'train': [<ImageDataset object>, <ImageDataset object>],
        'validation': [<ImageDataset object>, <ImageDataset object>],
        'test': [<ImageDataset object>, <ImageDataset object>]}
    """
    splits = ['train', 'validation', 'test']
    datasets = {split: [] for split in splits}

    for meta in dataset_meta:
        print(f"Loading {meta['path']} for all splits... ", end='')
        dataset = ImageDataset(
            meta['path'],
            meta.get('name', None),
            create_splits=True, # dataset.dataset == (train, val, test) splits from load_huggingface_dataset(...)
            download_mode=meta.get('download_mode', None)
        )

        train_ds, val_ds, test_ds = dataset.dataset

        for split, data in zip(splits, [train_ds, val_ds, test_ds]):
            # Create a new ImageDataset instance without calling __init__
            # This avoids calling load_huggingface_dataset(...) and redownloading
            split_dataset = ImageDataset.__new__(ImageDataset)

            # Assign the appropriate split data
            split_dataset.dataset = data

            # Copy other attributes from the initial dataset
            split_dataset.huggingface_dataset_path = dataset.huggingface_dataset_path
            split_dataset.huggingface_dataset_name = dataset.huggingface_dataset_name
            split_dataset.sampled_images_idx = dataset.sampled_images_idx

            # Append to the corresponding split list
            datasets[split].append(split_dataset)

        split_lengths = ', '.join([f"{split} len={len(datasets[split][0])}" for split in splits])
        print(f'done, {split_lengths}')

    return datasets

def load_and_split_datasets_with_transform_subsets(dataset_meta: list,
                                                   split_transforms: dict,
                                                   shuffle_before_split: bool = False) -> Dict[str, List[ImageDataset]]:
    """
    Load datasets from Hugging Face, apply specified transformation subsets for each split 
    (train, validation, test), and perform a stratified split by `original_index` to ensure 
    no data leakage between the splits.

    Parameters:
    -----------
    dataset_meta : list
        A list of dictionaries where each dictionary contains metadata for a dataset. 
        Metadata includes the Hugging Face dataset path, optional split name, and download mode.
    
    split_transforms : dict
        A dictionary where keys are split names ('train', 'validation', 'test'), and values 
        are dictionaries containing the subset names to be used for each split.

    Returns:
    --------
    Dict[str, List[ImageDataset]]
        A dictionary where keys are split names ('train', 'validation', 'test') and values 
        are lists of ImageDataset objects for each split.
    
    Algorithm:
    ----------
    1. Initialize an empty list for each split (train, validation, test).
    2. For each dataset in `dataset_meta`:
       a. Load all required subsets and store them.
       b. Group the indices by `original_index` to prepare for stratified splitting.
    3. Perform a stratified split on the `original_index` values into train, validation, and test sets.
    4. For each split (train, validation, test):
       a. Filter the data according to the assigned subset and the indices from the stratified split.
       b. Create and store the `ImageDataset` for the split.
    5. Return a datasets dictionary containing the splits.
    """
    splits = split_transforms.keys()
    subset_names = set(subset['name'] for subset in split_transforms.values())
    datasets = {split: [] for split in splits}
       
    for meta in dataset_meta:
        loaded_subsets = {}

        # Store subset indices grouped by original_index for stratified train-validation-test split
        all_data_indices = defaultdict(list)
        subset_original_index_dfs = []
        # Load all subsets and store in a dictionary
        for subset_name in subset_names:
            print(f"Loading {meta['path']} subset {subset_name} for all splits...")
            subset = ImageDataset(
                huggingface_dataset_path=meta['path'],
                huggingface_dataset_split=meta.get('name', None),
                huggingface_dataset_name=subset_name,
                create_splits=False,
                download_mode=meta.get('download_mode', None)
            )
            if shuffle_before_split:
                subset.dataset['train'] = subset.dataset['train'].shuffle(seed=42)
            # Save the loaded subset to the dictionary
            loaded_subsets[subset_name] = subset
            print(f"{subset_name} subset len: {len(subset.dataset['train'])}")

            print(f"Creating {subset_name} subset dataframe for stratified splits grouped by original index.")
            df = pd.DataFrame({'original_index': subset.dataset['train']['original_index'][:]})
            df['index'] = df.index
            subset_original_index_dfs.append(df)

        combined_df = pd.concat(subset_original_index_dfs, ignore_index=True)
        # Group by 'original_index' and aggregate indices into lists
        grouped = combined_df.groupby('original_index')['index'].apply(list).to_dict()
        # Get all unique original indices
        all_data_indices.update(grouped)
        all_original_indices = list(all_data_indices.keys())
        
        # Stratified split of indices
        train_indices, temp_indices = train_test_split(all_original_indices,
                                                       test_size=0.2,
                                                       random_state=42)

        val_indices, test_indices = train_test_split(temp_indices,
                                                     test_size=0.5,
                                                     random_state=42)
    
        def get_data_indices(indices_list):
            # Returns all subset indices for the given 'original_index' values, 
            # used to create non-overlapping train, validation, and test splits.
            return [index for orig_idx in indices_list for index in all_data_indices[orig_idx]]

        # Get all subset indices for each split based on the selected 'original_index' values.
        train_data_indices = get_data_indices(train_indices)
        val_data_indices = get_data_indices(val_indices)
        test_data_indices = get_data_indices(test_indices)
        
        if len(subset_names) == 1:
            assert set(train_data_indices).isdisjoint(set(val_data_indices))==True
            assert set(train_data_indices).isdisjoint(set(test_data_indices))==True
            assert set(val_data_indices).isdisjoint(set(test_data_indices))==True

        # Check split proportions
        print(f"train len: {len(train_data_indices)}, val len: {len(val_data_indices)}, test len: {len(test_data_indices)}")
        
        # Split data according to the specific transformation subset assigned for each split
        for split, original_indices in tqdm(zip(['train', 'validation', 'test'],
                                       [train_data_indices, val_data_indices, test_data_indices])):
            
            subset_name = split_transforms[split]['name']
            subset = loaded_subsets[subset_name]

            # Create a new ImageDataset instance for the split
            split_dataset = ImageDataset.__new__(ImageDataset)
            
            # Filter subset to include only indices corresponding to the current split's original_index values
            original_indices_set = set(original_indices)
            split_dataset.dataset = subset.dataset['train'].filter(
                lambda data_batch: [original_index in original_indices_set for original_index in data_batch['original_index']],
                batched=True,
                batch_size=5000
            )
            # Copy other attributes from the initial dataset
            split_dataset.huggingface_dataset_path = meta['path']
            split_dataset.huggingface_dataset_name = subset_name
            split_dataset.sampled_images_idx = []

            # Append to the corresponding split list
            datasets[split].append(split_dataset)

        split_lengths = ', '.join([f"{split} len={len(datasets[split][0].dataset)}" for split in splits])
        print(f'done, {split_lengths}')

    return datasets

def load_datasets(dataset_meta: dict = DATASET_META,
                  expert: bool = False,
                  split_transforms: dict = None) -> Tuple[Dict[str, List[ImageDataset]],
                                                        Dict[str, List[ImageDataset]]]:
    """
    Loads several ImageDatasets, each of which is an abstraction of a huggingface dataset.
    If loading a dataset tailored for training an expert/specialized model, perform a
    stratified split on transform subsets.

    Returns:
        (real_datasets: Dict[str, List[ImageDataset]], fake_datasets: Dict[str, List[ImageDataset]])

    """
    if expert:
        fake_datasets = load_and_split_datasets_with_transform_subsets(dataset_meta['fake'], split_transforms)
        real_datasets = load_and_split_datasets_with_transform_subsets(dataset_meta['real'], split_transforms)
    else:
        fake_datasets = load_and_split_datasets(dataset_meta['fake'])
        real_datasets = load_and_split_datasets(dataset_meta['real'])

    return real_datasets, fake_datasets

def create_source_label_mapping(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]]
    ) -> Dict:

    source_label_mapping = {}

    # Iterate through real datasets and set their source label to 0.0
    for split, dataset_list in real_datasets.items():
        for dataset in dataset_list:
            source = dataset.huggingface_dataset_path
            if source not in source_label_mapping.keys():
                source_label_mapping[source] = 0.0

    # Assign incremental labels to fake datasets
    fake_source_label = 1.0
    for split, dataset_list in fake_datasets.items():
        for dataset in dataset_list:
            source = dataset.huggingface_dataset_path
            if source not in source_label_mapping.keys():
                source_label_mapping[source] = fake_source_label
                fake_source_label += 1.0

    return source_label_mapping


def create_real_fake_datasets(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    train_transforms: transforms.Compose = None,
    val_transforms: transforms.Compose = None,
    test_transforms: transforms.Compose = None,
    source_labels: bool = False,
    normalize_config: dict = None) -> Tuple[RealFakeDataset, ...]:
    """
    Args:
        real_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        fake_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        train_transforms: transforms to apply to training dataset
        val_transforms: transforms to apply to val dataset
        test_transforms: transforms to apply to test dataset
    Returns:
        Train, val, and test RealFakeDatasets

    """
    source_label_mapping = \
    create_source_label_mapping(real_datasets, fake_datasets) if source_labels else None
    print(f"Source label mapping: {source_label_mapping}")
    
    train_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['train'],
        fake_image_datasets=fake_datasets['train'],
        transforms=train_transforms,
        source_label_mapping=source_label_mapping,
        normalize_config=normalize_config)

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        transforms=val_transforms,
        source_label_mapping=source_label_mapping,
        normalize_config=normalize_config)

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        transforms=test_transforms,
        source_label_mapping=source_label_mapping,
        normalize_config=normalize_config)

    return train_dataset, val_dataset, test_dataset