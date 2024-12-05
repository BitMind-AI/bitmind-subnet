from PIL import Image
from _dlib_pybind11 import rectangles
import numpy as np
import dlib

from base_miner.gating_mechanisms import Gate
from base_miner.DFB.config.constants import DLIB_FACE_PREDICTOR_PATH
from base_miner.registry import GATE_REGISTRY
from base_miner.gating_mechanisms.utils import get_face_landmarks, align_and_crop_face


@GATE_REGISTRY.register_module(module_name='FACE')
class FaceGate(Gate):
    """
    Gate subclass for face content detection and preprocessing.

    Attributes:
        gate_name (str): The name of the gate.
        predictor_path (str): Path to dlib face landmark model.
    """
    
    def __init__(self, gate_name: str = 'FaceGate', predictor_path=DLIB_FACE_PREDICTOR_PATH):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(predictor_path)
        super().__init__(gate_name, "face")

    def preprocess(self, image: np.ndarray, faces: rectangles, res=256) -> any:
        """
        Align and crop the largest face in the image

        Args:
            image: Input image array
            faces: Output out of a dlib face detection model
            res: NxN image size

        Returns:
            preprocessed image with largest face aligned and cropped
        """

        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Get the landmarks/parts for the face in box d only with the five key points
        face_shape = self.face_predictor(image, face)
        landmarks = get_face_landmarks(face_shape)
        cropped_face, _ = align_and_crop_face(image, landmarks, outsize=(res, res))
        # return cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Convert the cropped face back to a PIL Image if cropping was successful.
        if cropped_face is not None:
            image = Image.fromarray(cropped_face)
        else:
            print("Largest face was not successfully cropped.")
        return image
    
    def __call__(self, image: Image, res: int = 256) -> any:
        """
        Perform face detection and image aligning and cropping to the face.

        Args:
            image (PIL.Image): The image to classify and preprocess if content is detected.

        Returns:
            image (PIL.Image): The processed face image or original image if no faces.
        """
        image_np = np.array(image)
        faces = self.face_detector(image_np, 1)
        if faces is None or len(faces) == 0:
            return image, False

        processed_image = self.preprocess(image_np, faces, res)
        return processed_image, True
