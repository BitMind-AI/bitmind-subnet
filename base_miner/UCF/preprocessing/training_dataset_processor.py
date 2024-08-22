import os
import sys
from typing import List, Tuple, Dict
import torchvision.transforms as T
import torch
from tqdm import tqdm
import dlib
import numpy as np
import cv2
import time
from datasets import DatasetDict, Image, load_dataset, Dataset
from huggingface_hub import create_repo
from util.real_fake_dataset import RealFakeDataset
from bitmind.image_dataset import ImageDataset
from bitmind.constants import DATASET_META
from bitmind.utils.data import create_splits
from base_miner.UCF.preprocessing.preprocess import extract_aligned_face_dlib
from bitmind.image_transforms import random_aug_transforms
from base_miner.UCF.preprocessing.pil_image_dataset import PILImageDataset
from torch.utils.data import DataLoader
from skimage import transform as trans
from imutils import face_utils
from PIL import Image

class TrainingDatasetProcessor:
    def __init__(self,
                 config: dict,
                 transforms: dict = None,
                 hf_token: str = None,
                 dataset_meta: dict = DATASET_META):
        self.config = config
        self.transforms = transforms
        self.hf_token = hf_token
        self.dataset_meta = dataset_meta
        self.face_detector, self.face_predictor = self.load_face_models()

    def load_face_models(self):
        face_detector = dlib.get_frontal_face_detector()
        predictor_path = './preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
        if not os.path.exists(predictor_path):
            print(f"Predictor path does not exist: {predictor_path}")
            sys.exit()
        face_predictor = dlib.shape_predictor(predictor_path)
        print("Loaded face detector and predictor models.")
        return face_detector, face_predictor

    def get_keypts(self, image, face):
        # detect the facial landmarks for the selected face
        shape = self.face_predictor(image, face)
        
        # select the key points for the eyes, nose, and mouth
        leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
        reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
        nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
        lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
        rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
        
        pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

        return pts

    def align_and_crop_faces(self, rgb_image_arr, faces, res=256, mask=None):
        def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
            """ 
            align and crop the face according to the given bbox and landmarks
            landmark: 5 key points
            """
    
            M = None
            target_size = [112, 112]
            dst = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
    
            if target_size[1] == 112:
                dst[:, 0] += 8.0
    
            dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
            dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]
    
            target_size = outsize
    
            margin_rate = scale - 1
            x_margin = target_size[0] * margin_rate / 2.
            y_margin = target_size[1] * margin_rate / 2.
    
            # move
            dst[:, 0] += x_margin
            dst[:, 1] += y_margin
    
            # resize
            dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
            dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)
    
            src = landmark.astype(np.float32)
    
            # use skimage tranformation
            tform = trans.SimilarityTransform()
            tform.estimate(src, dst)
            M = tform.params[0:2, :]
    
            img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))
    
            if outsize is not None:
                img = cv2.resize(img, (outsize[1], outsize[0]))
            
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
                mask = cv2.resize(mask, (outsize[1], outsize[0]))
                return img, mask
            else:
                return img, None
    
        # Image size
        height, width = rgb_image_arr.shape[:2]
    
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = self.get_keypts(rgb_image_arr, face)
    
        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb_image_arr, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract all landmarks from the aligned face
        face_align = self.face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = self.face_predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        # Convert back to RGB
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        return cropped_face, landmark, mask_face

    def align_and_crop_faces_in_batch(self, batch_images, batch_faces, res=256, mask=None):
        cropped_faces = []
        landmarks = []
        mask_faces = []
        assert len(batch_images) == len(batch_faces)
        for idx in range(len(batch_images)):
            if len(batch_faces[idx]): # If one or more face was detected
                cropped_face, landmark, mask_face = \
                self.align_and_crop_faces(np.array(batch_images[idx]), batch_faces[idx], res=res, mask=None)
                if cropped_face is not None:
                    cropped_face = Image.fromarray(cropped_face).convert('RGB')
                cropped_faces.append(cropped_face)
                landmarks.append(landmark)
                mask_faces.append(mask_face)
        return cropped_faces, landmarks, mask_faces
        
    def detect_faces_in_batch(self, batch_images):
        batch_faces = []
        batch_faces_present = []
        for image in batch_images:
            faces = self.face_detector(np.array(image), 1)
            batch_faces.append(faces)
            batch_faces_present.append(len(faces) > 0)
        return batch_faces, batch_faces_present

    def filter_cropped_faces_with_no_landmark(self, faces_present, cropped_faces, landmarks, masks):
        assert (faces_present.count(True) == len(cropped_faces)) and (len(cropped_faces) == len(landmarks))
        assert len(masks) == len(landmarks)
        cropped_faces_present_with_landmarks = []
        cropped_faces_with_landmarks = []
        masks_with_landmarks = []
        valid_landmarks = []
        landmarks_idx = 0
        for i in range(len(faces_present)):
            if faces_present[i]:
                assert landmarks_idx < len(landmarks) 
                if landmarks[landmarks_idx] is not None:
                    cropped_faces_present_with_landmarks.append(True)
                    cropped_faces_with_landmarks.append(cropped_faces[landmarks_idx])
                    masks_with_landmarks.append(masks[landmarks_idx])
                    valid_landmarks.append(landmarks[landmarks_idx])
                    #print("appending landmark", landmarks_idx, landmarks[landmarks_idx] is not None)
                else:
                    cropped_faces_present_with_landmarks.append(False)
                    #print("skipping landmark", landmarks_idx, landmarks[landmarks_idx] is not None)
                landmarks_idx += 1
            else:
                cropped_faces_present_with_landmarks.append(False)
        return cropped_faces_present_with_landmarks, cropped_faces_with_landmarks, valid_landmarks, masks_with_landmarks
    
    def face_filter_and_crop_align(self, dataset_dict, transform, batch_size=32):
        for split in dataset_dict.keys():
            dataloader = DataLoader(dataset_dict[split].with_format("torch"), batch_size=batch_size, shuffle=False)
            indices_to_keep = []
            new_dataset = {"image": [],
                            "landmark": [],
                            "mask": []}
            total_valid_faces = 0
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                images = batch['image']
                # Convert tensors to PIL Images
                start_time = time.time()
                images = [T.ToPILImage()(transform(T.ToPILImage()(image))) for image in images]
    
                # Detect and process faces
                faces, faces_present = self.detect_faces_in_batch(images)
                cropped_faces, landmarks, masks = self.align_and_crop_faces_in_batch(images, faces)
                valid_faces_present, valid_faces, valid_landmarks, valid_masks = \
                self.filter_cropped_faces_with_no_landmark(faces_present, cropped_faces, landmarks, masks)
                assert valid_faces_present.count(True) == len(valid_faces)
                
                total_valid_faces += len(valid_faces)
                # Collect results where faces are present
                if any(valid_faces_present):
                    batch_indices_to_keep = torch.arange(batch_idx * batch_size,
                                                        batch_idx * batch_size + len(images))[valid_faces_present]
                    assert len(batch_indices_to_keep) == valid_faces_present.count(True)
                    assert len(batch_indices_to_keep) == len(valid_faces)
                    
                    indices_to_keep.extend(batch_indices_to_keep.tolist())
                    new_dataset["image"].extend(valid_faces)
                    new_dataset["landmark"].extend(valid_landmarks)
                    new_dataset["mask"].extend(valid_masks)
                    assert (len(new_dataset["image"]) == len(new_dataset["landmark"])) and \
                           (len(new_dataset["landmark"]) == len(new_dataset["mask"]))

            # Replace dataset with processed cropped and aligned face data
            filtered_split_dataset = Dataset.from_dict(new_dataset)
            assert total_valid_faces == len(filtered_split_dataset)
            dataset_dict[split] = filtered_split_dataset        
        return dataset_dict
    
    def load_process_split_datasets(self, dataset_meta: list, transform_info, split=True) -> Dict[str, List[ImageDataset]]:
        print("Processing datasets for " + transform_info[0])
        for meta in dataset_meta:
            hf_repo_path = meta['path']+'_'+transform_info[0]
            print(f"Loading {meta['path']} for all splits... ", end='')
            dataset = ImageDataset(meta['path'],
                                   meta.get('name', None),
                                   create_splits=False,
                                   download_mode=meta.get('download_mode', None))

            if self.config['face_crop_and_align']:
                dataset.dataset = self.face_filter_and_crop_align(dataset.dataset, transform_info[1]) 
            if split:
                splits = ['train', 'validation', 'test']
                datasets = {split: [] for split in splits}
                train_ds, val_ds, test_ds = create_splits(dataset.dataset)
                self.upload_datasets(hf_repo_path,
                                     DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds}))
            else:
                self.upload_datasets(hf_repo_path, dataset.dataset)
                #return datasets
            #return dataset

    def get_processed_datasets(self,
                               transform_info,
                               split=True) -> Tuple[Dict[str, List[ImageDataset]], Dict[str, List[ImageDataset]]]:
        fake_datasets = self.load_process_split_datasets(self.dataset_meta['fake'], transform_info, split)
        real_datasets = self.load_process_split_datasets(self.dataset_meta['real'], transform_info, split)
        return real_datasets, fake_datasets

    def create_source_label_mapping(self,
                                    real_datasets: Dict[str, List[ImageDataset]],
                                    fake_datasets: Dict[str, List[ImageDataset]]) -> Dict:
        source_label_mapping = {}
        for split, dataset_list in real_datasets.items():
            for dataset in dataset_list:
                source = dataset.huggingface_dataset_path
                if source not in source_label_mapping.keys():
                    source_label_mapping[source] = 0.0

        fake_source_label = 1.0
        for split, dataset_list in fake_datasets.items():
            for dataset in dataset_list:
                source = dataset.huggingface_dataset_path
                if source not in source_label_mapping.keys():
                    source_label_mapping[source] = fake_source_label
                    fake_source_label += 1.0

        return source_label_mapping

    def create_real_fake_datasets(self,
                                  real_datasets: Dict[str, List[ImageDataset]],
                                  fake_datasets: Dict[str, List[ImageDataset]]) -> Tuple[RealFakeDataset, ...]:
        source_label_mapping = self.create_source_label_mapping(real_datasets, fake_datasets)
        print(f"Source label mapping: {source_label_mapping}")
        train_dataset = RealFakeDataset(real_image_datasets=real_datasets['train'],
                                        fake_image_datasets=fake_datasets['train'],
                                        source_label_mapping=source_label_mapping)
        val_dataset = RealFakeDataset(real_image_datasets=real_datasets['validation'],
                                      fake_image_datasets=fake_datasets['validation'],
                                      source_label_mapping=source_label_mapping)
        test_dataset = RealFakeDataset(real_image_datasets=real_datasets['test'],
                                       fake_image_datasets=fake_datasets['test'],
                                       source_label_mapping=source_label_mapping)
        return train_dataset, val_dataset, test_dataset

    def load_or_generate_datasets(self,
                                  dataset_name=None,
                                  upload=False) -> Tuple[RealFakeDataset, RealFakeDataset, RealFakeDataset]:
        if dataset_name:
            try:
                dataset_dict = load_dataset(dataset_name)
                return dataset_dict['train'], dataset_dict['validation'], dataset_dict['test']
                print(f"Dataset {dataset_name} loaded from Hugging Face Hub.")
            except Exception as e:
                print(f"Dataset {dataset_name} not found on Hugging Face Hub. Error: {e}")
        else: # Generate dataset
            for t in self.transforms.keys():
                real_datasets, fake_datasets = self.get_processed_datasets((t, self.transforms[t]), split=True)

    def upload_datasets(self,
                        dataset_name: str,
                        datasets):
        print("Dataset to push:", datasets)
        print("Pushing "+dataset_name+"_face_training to hub.")
        repo_id = dataset_name+"_face_training"
        try:
            create_repo(repo_id, token=self.hf_token, exist_ok=False)
            datasets.push_to_hub(repo_id=repo_id, token=self.hf_token)
        except Exception as e:
            print(e)
        print("Uploaded "+dataset_name+"_face_training.")