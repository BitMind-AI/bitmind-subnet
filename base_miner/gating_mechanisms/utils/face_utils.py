from skimage import transform as trans
import numpy as np
import cv2


def get_face_landmarks(face):
    """
    Args:
        face: dlib face rectangle object for face

    Returns:
        numpy array containing key points for eyes, nose and mouth
    """
    leye = np.array([face.part(37).x, face.part(37).y]).reshape(-1, 2)
    reye = np.array([face.part(44).x, face.part(44).y]).reshape(-1, 2)
    nose = np.array([face.part(30).x, face.part(30).y]).reshape(-1, 2)
    lmouth = np.array([face.part(49).x, face.part(49).y]).reshape(-1, 2)
    rmouth = np.array([face.part(55).x, face.part(55).y]).reshape(-1, 2)
    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def align_and_crop_face(
        img: np.ndarray,
        landmarks: np.ndarray, outsize: tuple, scale=1.3, mask=None):
    """
    Align and crop the face according to the given landmarks
    Args:
        img: input image containing the face
        landmarks: 5 key points of face, determined by get_face_landmarks
        outsize: size to use in scaling
        scale: margin
        mask: optional face mask to transform alongside the face

    Returns:
        cropped and aligned face, optionally with a correspondingly
         cropped and aligned mask
    """
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

    src = landmarks.astype(np.float32)

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
