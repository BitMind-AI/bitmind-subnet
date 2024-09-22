from PIL import Image
import numpy as np
import cv2
import dlib
from imutils import face_utils
from skimage import transform as trans
from detectors.gating_mechanisms import Gate
from detectors.UCF.config.constants import DLIB_FACE_PREDICTOR_PATH
from detectors import GATE_REGISTRY


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

    # use skimage transformation
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

    def align_and_crop_largest_face(
            self,
            image: np.ndarray,
            faces: np.ndarray,
            res=256,
            mask=None):
    
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = self.get_keypts(image, face)
        cropped_face, mask_face = img_align_crop(image, landmarks, outsize=(res, res), mask=mask)
        #return cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        return cropped_face

    def preprocess(self, image: np.ndarray, faces: np.ndarray, res=256) -> any:
        """Preprocess the image based on its content type."""

        # If faces are detected, crop and align the largest face.
        cropped_face = self.align_and_crop_largest_face(
            image, faces, res=res, mask=None
        )
        
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
