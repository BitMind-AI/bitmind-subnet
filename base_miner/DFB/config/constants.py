import os

CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(CONFIGS_DIR, ".."))  # Points to bitmind-subnet/base_miner/DFB/
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights")

CONFIG_PATHS = {
    'UCF': os.path.join(CONFIGS_DIR, "ucf.yaml"),
    'TALL': os.path.join(CONFIGS_DIR, "tall.yaml") 
}

HF_REPOS = {
    "UCF": "bitmind/ucf",
    "TALL": "bitmind/tall"
}

BACKBONE_CKPT = "xception_best.pth"

DLIB_FACE_PREDICTOR_PATH = os.path.abspath(os.path.join(BASE_PATH, "../../bitmind/dataset_processing/dlib_tools/shape_predictor_81_face_landmarks.dat"))