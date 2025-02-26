from pathlib import Path

HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
TARGET_IMAGE_SIZE = (256, 256)


IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/bm-real"},
        {"path": "bitmind/open-images-v7"},
        {"path": "bitmind/celeb-a-hq"},
        {"path": "bitmind/ffhq-256"},
        {"path": "bitmind/MS-COCO-unique-256"}
    ],
    "fake": [
        {"path": "bitmind/bm-realvisxl"},
        {"path": "bitmind/bm-mobius"},
        {"path": "bitmind/bm-sdxl"}
    ]
}

# see bitmind-subnet/create_video_dataset_example.sh 
VIDEO_DATASETS = {
    "real": [
        {"path": ""}  # example: {"path": "~/.cache/sn34/train_dataset/real_frames_dataset"}
    ],
    "fake": [
        {"path": ""}  # example: {"path": "~/.cache/sn34/train_dataset/synthetic_frames_dataset"}
    ]
}

FACE_IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/ffhq-256_training_faces", "name": "base_transforms"},
        {"path": "bitmind/celeb-a-hq_training_faces", "name": "base_transforms"}

    ],
    "fake": [
        {"path": "bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces", "name": "base_transforms"},
        {"path": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces", "name": "base_transforms"}
    ]
}
