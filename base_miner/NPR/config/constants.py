import os

# Path to the directory containing the constants.py file
CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))

# The base directory for NPR-related files, i.e., NPR directory
NPR_BASE_PATH = os.path.abspath(os.path.join(CONFIGS_DIR, ".."))  # Points to dfd-arena/detectors/NPR/
# Absolute paths for the required files and directories
WEIGHTS_DIR = os.path.join(NPR_BASE_PATH, "weights/") # Path to pretrained weights directory

