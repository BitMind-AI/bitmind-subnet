import os

# Path to the directory containing the constants.py file
UCF_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))

# The base directory for UCF-related files, i.e., UCF directory
UCF_BASE_PATH = os.path.abspath(os.path.join(UCF_CONFIGS_DIR, ".."))  # Points to bitmind-subnet/base_miner/UCF/
# Absolute paths for the required files and directories
CONFIGS_DIR = os.path.join(UCF_BASE_PATH, "config/")
CONFIG_PATH = os.path.join(CONFIGS_DIR, "ucf.yaml")  # Path to the ucf.yaml file
WEIGHTS_DIR = os.path.join(UCF_BASE_PATH, "weights/") # Path to pretrained weights directory

WEIGHTS_HF_PATH = "bitmind/ucf"
PRETRAINED_CONFIG = "pretrained_config.yaml"
DFB_CKPT = "ucf_best.pth"
BM_CKPT = "ucf_bitmind_best.pth"
BACKBONE_CKPT = "xception_best.pth"
BM_FACE_CKPT = "ucf_bitmind_face.pth"
BM_18K_CKPT = "ucf-bitmind-18k.pth"

DLIB_FACE_PREDICTOR_PATH = os.path.abspath(os.path.join(UCF_BASE_PATH, "../../bitmind/dataset_processing/dlib_tools/shape_predictor_81_face_landmarks.dat"))