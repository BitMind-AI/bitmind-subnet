import os

# Path to the directory containing the constants.py file
CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))

# The base directory for UCF-related files, i.e., UCF directory
UCF_BASE_PATH = os.path.abspath(os.path.join(CONFIGS_DIR, ".."))  # Points to bitmind-subnet/base_miner/UCF/
# Absolute paths for the required files and directories
CONFIG_PATH = os.path.join(CONFIGS_DIR, "ucf.yaml")  # Path to the ucf.yaml file
WEIGHTS_DIR = os.path.join(UCF_BASE_PATH, "weights/") # Path to pretrained weights directory

HF_REPO = "bitmind/ucf"
BACKBONE_CKPT = "xception_best.pth"

DLIB_FACE_PREDICTOR_PATH = os.path.abspath(os.path.join(UCF_BASE_PATH, "../../bitmind/dataset_processing/dlib_tools/shape_predictor_81_face_landmarks.dat"))