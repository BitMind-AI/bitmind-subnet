import numpy as np


class RealFakeDataset:

    def __init__(
        self,
        real_image_datasets,
        fake_image_datasets,
        image_generator=None,
        transforms=None,
        fake_prob=0.5,
    ):

        self.real_image_datasets = real_image_datasets
        self.fake_image_datasets = fake_image_datasets
        self.image_generator = image_generator
        self.transforms = transforms
        self.fake_prob = fake_prob

        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }

    def __getitem__(self, index):

        if len(self._history['index']) > index:
            self.reset()

        if np.random.rand() > self.fake_prob:
            source = self.fake_image_datasets[np.random.randint(0, len(self.fake_image_datasets))]
            image = source[index]['image']
            label = 1.
        else:
            source = self.real_image_datasets[np.random.randint(0, len(self.real_image_datasets))]
            #image = source.sample(1)[0]['image']
            imgs, idx = source.sample(1)
            image = imgs[0]['image']
            index = idx[0]
            label = 0.

        self._history['source'].append(source.huggingface_dataset_path)
        self._history['label'].append(label)
        self._history['index'].append(index)

        try:
            if self.transforms is not None:
                image = self.transforms(image)
        except Exception as e:
            print(e)
            print(source.huggingface_dataset_path, index)

        return image, label


    def __len__(self):
        """ This limits the number of images sampled each epoch to the length of the smallest dataset """
        real_dataset_min = min([len(ds) for ds in self.real_image_datasets])
        fake_dataset_min = min([len(ds) for ds in self.fake_image_datasets])
        return min(fake_dataset_min, real_dataset_min)

    def reset(self):
        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }
