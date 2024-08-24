import os
import sys
from typing import List, Tuple, Dict
import torchvision.transforms as T
import torch
from tqdm import tqdm
import dlib
import numpy as np
import cv2
from datasets import DatasetDict, Image, load_dataset, Dataset
from huggingface_hub import create_repo
from bitmind.image_dataset import ImageDataset
from bitmind.utils.data import create_splits
from base_miner.UCF.preprocessing.preprocess import extract_aligned_face_dlib
from bitmind.image_transforms import random_aug_transforms
from bitmind.constants import DATASET_META
from torch.utils.data import DataLoader
from skimage import transform as trans
from imutils import face_utils
from PIL import Image

class TrainingDatasetProcessor:
    def __init__(self,
                 dataset_meta: dict = DATASET_META,
                 faces_only: bool = False,
                 transforms: dict = None,
                 split: bool = False,
                 hf_token: str = None,
                ):
        self.dataset_meta = dataset_meta
        self.faces_only = faces_only
        self.transforms = transforms
        self.hf_token = hf_token
        self.split = split
        if self.faces_only:
            self.face_detector, self.face_predictor = self.load_face_models()

    def load_face_models(self):
        face_detector = dlib.get_frontal_face_detector()
        predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
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
    
    def preprocess_faces_only(self, dataset_dict, transform, batch_size=32):
        for split in dataset_dict.keys():
            dataloader = DataLoader(dataset_dict[split].with_format("torch"), batch_size=batch_size, shuffle=False)
            indices_to_keep = []
            preprocessed_dataset = {"image": [],
                                    "original_index": [],
                                    "landmark": [],
                                    "mask": []}
            total_valid_faces = 0
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                images = batch['image']
                # Convert tensors to PIL Images
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
                    preprocessed_dataset["image"].extend(valid_faces)
                    preprocessed_dataset["original_index"].extend(batch_indices_to_keep.tolist())
                    preprocessed_dataset["landmark"].extend(valid_landmarks)
                    preprocessed_dataset["mask"].extend(valid_masks)
                    assert (len(preprocessed_dataset["image"]) == len(preprocessed_dataset["original_index"])) and \
                           (len(preprocessed_dataset["landmark"]) == len(preprocessed_dataset["mask"])) and \
                           (len(preprocessed_dataset["image"]) == len(preprocessed_dataset["landmark"]))
            # Replace dataset with processed cropped and aligned face data
            filtered_split_dataset = Dataset.from_dict(preprocessed_dataset)
            assert total_valid_faces == len(filtered_split_dataset)
            dataset_dict[split] = filtered_split_dataset        
        return dataset_dict

    def preprocess(self, dataset_dict, transform, batch_size=32):
        for split in dataset_dict.keys():
            dataloader = DataLoader(dataset_dict[split].with_format("torch"), batch_size=batch_size, shuffle=False)
            indices_to_keep = []
            preprocessed_dataset = {"image": []}
            total_valid_faces = 0
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                images = batch['image']
                # Transform images and ensure PIl format
                images = [T.ToPILImage()(transform(T.ToPILImage()(image))) for image in images]
                preprocessed_dataset["image"].extend(images)
            # Replace dataset with processed cropped and aligned face data
            transformed_dataset = Dataset.from_dict(preprocessed_dataset)
            dataset_dict[split] = transformed_dataset
        return dataset_dict

    def upload_dataset(self, repo_id: str, dataset, transform_name):
        print(f"Pushing {dataset} to {repo_id} in hub.")
        try:
            dataset.push_to_hub(repo_id=repo_id, token=self.hf_token, config_name=transform_name)
            print(f"Uploaded {repo_id} config {transform_name}.")
        except Exception as e:
            print(f"Failed to upload {repo_id} config {transform_name}:", e)
    
    def load_preprocess_upload_datasets(self, dataset_meta: list, transform_info) -> Dict[str, List[ImageDataset]]:
        for meta in dataset_meta:
            print(f"Loading {meta['path']}...")
            dataset = ImageDataset(meta['path'],
                                   meta.get('name', None),
                                   create_splits=False,
                                   download_mode=meta.get('download_mode', None))

            hf_repo_path = meta['path'] + '_training'
            if self.faces_only:
                print(f"Preprocessing {meta['path']} faces only...")
                hf_repo_path += '_faces'
                dataset.dataset = self.preprocess_faces_only(dataset.dataset, transform_info[1]) 
            else:
                print(f"Preprocessing {meta['path']}...")
                dataset.dataset = self.preprocess(dataset.dataset, transform_info[1]) 
            print(f"Uploading preprocessed {meta['path']}...")
            if self.split:
                train_ds, val_ds, test_ds = create_splits(dataset.dataset)
                self.upload_dataset(repo_id=hf_repo_path+'_splits',
                                    dataset=DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds}),
                                    transform_name=transform_info[0])
            else:
                self.upload_dataset(repo_id=hf_repo_path,
                                    dataset=dataset.dataset,
                                    transform_name=transform_info[0])
    
    def process_and_upload_all_datasets(self):
        for t in self.transforms.keys():
            print("Transform:", t)
            self.load_preprocess_upload_datasets(self.dataset_meta['fake'], (t, self.transforms[t]))
            self.load_preprocess_upload_datasets(self.dataset_meta['real'], (t, self.transforms[t]))
    