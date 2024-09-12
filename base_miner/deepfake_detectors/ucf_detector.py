import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore INFO and WARN messages

import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from PIL import Image
from huggingface_hub import hf_hub_download
from imutils import face_utils
from skimage import transform as trans
import gc

from base_miner.UCF.config.constants import (
    CONFIG_PATH,
    WEIGHTS_PATH,
    WEIGHTS_HF_PATH,
    DFB_CKPT,
    BACKBONE_CKPT,
    DLIB_FACE_PREDICTOR_PATH
)

from base_miner.UCF.detectors import DETECTOR
from base_miner.deepfake_detectors import DETECTOR_REGISTRY, DeepfakeDetector

@DETECTOR_REGISTRY.register_module(module_name='UCF')
class UCFDetector(DeepfakeDetector):
    """
    This class initializes a pretrained UCF model by loading the necessary configurations, checkpoints,
    and  face detection tools required for training and inference. It sets up the device (CPU or GPU), 
    loads the specified weights, and initializes the face detector and predictor from dlib.

    Attributes:
        config_path (str): Path to the configuration YAML file for the UCF model.
        weights_dir (str): Directory path where UCF and Xception backbone weights are stored.
        weights_hf_repo_name (str): Name of the Hugging Face repository containing the model weights.
        ucf_checkpoint_name (str): Filename of the UCF model checkpoint.
        backbone_checkpoint_name (str): Filename of the backbone model checkpoint.
        predictor_path (str): Path to the dlib face predictor file.
        specific_task_number (int): Number of different fake training dataset/forgery methods
                                    for UCF to disentangle (DeepfakeBench default is 5, the
                                    num of datasets of FF++)."""
    
    def __init__(self, model_name: str = 'UCF', config_path=CONFIG_PATH, weights_dir=WEIGHTS_PATH, 
             weights_hf_repo_name=WEIGHTS_HF_PATH, ucf_checkpoint_name=DFB_CKPT, 
             backbone_checkpoint_name=BACKBONE_CKPT, predictor_path=DLIB_FACE_PREDICTOR_PATH, 
             specific_task_number=5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = Path(config_path)
        self.weights_dir = Path(weights_dir)
        self.hugging_face_repo_name = weights_hf_repo_name
        self.ucf_checkpoint_name = ucf_checkpoint_name
        self.backbone_checkpoint_name = backbone_checkpoint_name
        self.config = self.load_config()
        self.config['specific_task_number'] = specific_task_number

        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(predictor_path)
        
        self.init_cudnn()
        self.init_seed()
        self.ensure_weights_are_available(self.ucf_checkpoint_name)
        self.ensure_weights_are_available(self.backbone_checkpoint_name)
        super().__init__(model_name)

    def ensure_weights_are_available(self, model_filename):
        destination_path = self.weights_dir / model_filename
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created directory {destination_path.parent}.")
        if not destination_path.exists():
            model_path = hf_hub_download(self.hugging_face_repo_name, model_filename)
            model = torch.load(model_path, map_location=self.device)
            torch.save(model, destination_path)
            print(f"Downloaded {model_filename} to {destination_path}.")
        else:
            print(f"{model_filename} already present at {destination_path}.")

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"The config file at {self.config_path} does not exist.")
        with self.config_path.open('r') as file:
            return yaml.safe_load(file)
        
    def init_cudnn(self):
        if self.config.get('cudnn'):
            cudnn.benchmark = True

    def init_seed(self):
        seed_value = self.config.get('manualSeed')
        if seed_value:
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

    def load_model(self):
        model_class = DETECTOR[self.config['model_name']]
        self.model = model_class(self.config).to(self.device)
        self.model.eval()
        weights_path = self.weights_dir / self.ucf_checkpoint_name
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=True)
            print('Loaded checkpoint successfully.')
        except FileNotFoundError:
            print('Failed to load the pretrained weights.')

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

    def detect_faces(self, image):
        """Detect faces in a PIL Image and return the count and the face coordinates.

        Args:
            image (PIL.Image): An RGB image object.
    
        Returns:
            tuple: A tuple containing the number of faces detected and their coordinates,
                   or (None, None, None) if no faces are detected.
        """
        
        # Convert RGB PIL Image to numpy array
        image_np = np.array(image)
    
        # Detect faces using dlib's frontal face detector (takes RGB).
        faces = self.face_detector(image_np, 1)
        
        # Check if any faces were detected
        if len(faces):
            return faces, len(faces)
        else:
            return None, None

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
    
    def preprocess(self, image, res=256, faces=None):
        """Preprocess the image for model inference.
        
        This function handles optional face alignment and cropping if faces are detected,
        then it applies resizing, normalization, and conversion to tensor format.
    
        Args:
            image (PIL.Image): The RGB image to preprocess.
            res (int): The resolution to which the image is resized.
            faces (list): List of face detections, where each face is represented as coordinates.
    
        Returns:
            torch.Tensor: The preprocessed image tensor, ready for model inference.
        """
        if faces:
            rgb_image_arr = np.array(image)
            # If faces are detected, crop and align the largest face.
            cropped_face, landmark, mask_face = self.align_and_crop_faces(
                rgb_image_arr, faces, res=res, mask=None
            )
            
            # Convert the cropped face back to a PIL Image if cropping was successful.
            if cropped_face is not None:
                image = Image.fromarray(cropped_face)
            else:
                # Log a message if face cropping failed.
                print("Largest face was not successfully cropped.")
        
        # Convert image to RGB format to ensure consistent color handling.
        image = image.convert('RGB')
    
        # Define transformation sequence for image preprocessing.
        transform = transforms.Compose([
            transforms.Resize((res, res), interpolation=Image.LANCZOS),  # Resize image to specified resolution.
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
            transforms.Normalize(mean=self.config['mean'], std=self.config['std'])  # Normalize the image tensor.
        ])
        
        # Apply transformations and add a batch dimension for model inference.
        image_tensor = transform(image).unsqueeze(0)
        
        # Move the image tensor to the specified device (e.g., GPU).
        return image_tensor.to(self.device)

    def infer(self, image_tensor):
        """ Perform inference using the model. """
        with torch.no_grad():
            self.model({'image': image_tensor}, inference=True)
        return self.model.prob[-1]

    def __call__(self, image: Image) -> float:
        faces, _ = self.detect_faces(image)
        image_tensor = self.preprocess(image, faces=faces)
        
        # Perform inference
        with torch.no_grad():
            self.model({'image': image_tensor}, inference=True)
        
        return self.model.prob[-1]
    
    def free_memory(self):
        """ Frees up memory by setting model and large data structures to None. """
        print("Freeing up memory...")

        if self.model is not None:
            self.model.cpu()  # Move model to CPU to free up GPU memory (if applicable)
            del self.model
            self.model = None

        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None

        if self.face_predictor is not None:
            del self.face_predictor
            self.face_predictor = None

        gc.collect()

        # If using GPUs and PyTorch, clear the cache as well
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Memory freed successfully.")