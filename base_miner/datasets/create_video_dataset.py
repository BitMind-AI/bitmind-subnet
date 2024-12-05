from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from multiprocessing import Pool, cpu_count

import cv2
import glob
import os

import argparse
from PIL import Image
from datasets import Dataset, DatasetInfo, Image as HFImage, Split
from datasets.features import Features, Sequence, Value
from tqdm import tqdm


def process_single_video(args: Tuple[Path, Path, int, Optional[int], bool]) -> Tuple[str, int]:
    """
    Extract frames from a single video

    Args:
        args: Tuple containing (video_file, output_dir, frame_rate, max_frames, overwrite)

    Returns:
        Tuple of (video_name, number_of_frames_saved)
    """
    video_file, output_dir, frame_rate, max_frames, overwrite = args
    video_name = video_file.stem
    video_output_dir = output_dir / video_name
    
    if video_output_dir.exists() and not overwrite:
        return video_name, 0
    
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    video_capture = cv2.VideoCapture(str(video_file))
    frame_idx = 0
    saved_frame_count = 0
    
    while True:
        success, frame = video_capture.read()
        if not success or (max_frames is not None and saved_frame_count >= max_frames):
            break
        
        if frame_idx % frame_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frame_filename = video_output_dir / f"frame_{frame_idx:05d}.png"
            pil_image.save(frame_filename)
            saved_frame_count += 1
        
        frame_idx += 1
    
    video_capture.release()
    return video_name, saved_frame_count


def extract_frames_from_videos(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    num_videos: Optional[int] = None,
    frame_rate: int = 1,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
    num_workers: Optional[int] = None
) -> None:
    """
    Extract frames from videos (mp4s -> directories of PILs) using multiprocessing

    Args:
        input_dir: Directory containing input MP4 files
        output_dir: Directory where extracted frames will be saved
        num_videos: Number of videos to process. If None, processes all videos
        frame_rate: Extract one frame every 'frame_rate' frames
        max_frames: Maximum number of frames to extract per video
        overwrite: If True, overwrites existing frame directories
        num_workers: Number of worker processes to use. If None, uses CPU count
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_dir.glob("*.mp4"))
    if num_videos is not None:
        video_files = video_files[:num_videos]
    
    if not num_workers:
        num_workers = cpu_count()
    
    print(f'Processing {len(video_files)} videos using {num_workers} workers')
    
    # Prepare arguments for each video
    process_args = [
        (video_file, output_dir, frame_rate, max_frames, overwrite)
        for video_file in video_files
    ]
    
    # Process videos in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, process_args),
            total=len(video_files),
            desc="Extracting frames"
        ))
    
    # Print results
    for video_name, frame_count in results:
        if frame_count > 0:
            print(f"Extracted {frame_count} frames from {video_name}")
        else:
            print(f"Skipped {video_name} (already exists)")


def create_video_frames_dataset(
    frames_dir: Union[str, Path],
    dataset_name: str = "video_frames",
    validate_frames: bool = False,
    delete_corrupted: bool = False,
) -> Dataset:
    """Create a HuggingFace dataset from a directory of video frames."""
    frames_dir = Path(frames_dir)
    video_data: Dict[str, Dict[str, List]] = defaultdict(lambda: {'frames': [], 'frame_numbers': []})
    
    for video_dir in tqdm(sorted(os.listdir(frames_dir)), desc='processing video frames'):
        video_path = frames_dir / video_dir
        
        if not video_path.is_dir():
            continue
        
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_files.extend(glob.glob(str(video_path / ext)))
        
        image_files.sort()

        # Validate images before adding them to the dataset
        if validate_frames:
            valid_frames = []
            valid_frame_numbers = []
            for img_path in tqdm(image_files, desc="Checking image files"):
                try:
                    # Attempt to fully load the image to verify it's valid
                    with Image.open(img_path) as img:
                        img.load()  # Force load the image data
                        frame_num = int(Path(img_path).stem.split('_')[1])
                        valid_frames.append(img_path)
                        valid_frame_numbers.append(frame_num)
                except Exception as e:
                    print(f"Skipping corrupted image {img_path}: {str(e)}")
                    if delete_corrupted:
                        print(f"Deleting {img_path} (delete_corrupted = true)")
                        Path(img_path).unlink()
                    continue
            if valid_frames:  # Only add videos that have valid frames
                video_data[video_dir]['frames'] = valid_frames
                video_data[video_dir]['frame_numbers'] = valid_frame_numbers
        else:
            video_data[video_dir]['frames'] = image_files
            video_data[video_dir]['frame_numbers'] = list(range(len(image_files)))
            print(video_data[video_dir]['frames'][:10])
            print(video_data[video_dir]['frame_numbers'][:10])
   
    dataset_dict = {
        "video_id": [],
        "frames": [],
        "frame_numbers": [],
        "num_frames": []
    }
    
    for video_id, data in video_data.items():
        if data['frames']:  # Double check we have frames
            dataset_dict["video_id"].append(video_id)
            dataset_dict["frames"].append(data["frames"])
            dataset_dict["frame_numbers"].append(data["frame_numbers"])
            dataset_dict["num_frames"].append(len(data["frames"]))
    
    features = Features({
        "video_id": Value("string"),
        "frames": Sequence(Value("string")),
        "frame_numbers": Sequence(Value("int64")),
        "num_frames": Value("int64")
    })
    
    dataset_info = DatasetInfo(
        description="Video frames dataset",
        features=features,
        supervised_keys=None,
        homepage="",
        citation="",
        task_templates=None,
        dataset_name=dataset_name
    )
    
    # Create dataset with validated images
    dataset = Dataset.from_dict(
        dataset_dict,
        info=dataset_info,
        features=features
    )
    
    # Convert to HuggingFace image format with error handling
    def convert_frames_to_images(example):
        converted_frames = []
        for frame_path in example["frames"]:
            try:
                converted_frames.append(HFImage().encode_example(frame_path))
            except Exception as e:
                print(f"Error converting {frame_path}: {str(e)}")
                continue
        example["frames"] = converted_frames
        return example
    
    #dataset = dataset.map(convert_frames_to_images)
    return dataset


def main() -> None:
    """Parse command line arguments and run the dataset creation pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and create a HuggingFace dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing input MP4 files."
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Path to the directory where extracted frames will be saved."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path where the HuggingFace dataset will be saved."
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=None,
        help="Number of videos to process. If not specified, processes all videos."
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=5,
        help="Extract one frame every 'frame_rate' frames."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract per video."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrites existing frame directories."
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="If set, skips the frame extraction step and only creates the dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="video_frames",
        help="Name for the local HuggingFace dataset to be created."
    )
    
    args = parser.parse_args()
    
    if not args.skip_extraction:
        print("Extracting frames from videos...")
        extract_frames_from_videos(
            input_dir=args.input_dir,
            output_dir=args.frames_dir,
            num_videos=args.num_videos,
            frame_rate=args.frame_rate,
            max_frames=args.max_frames,
            overwrite=args.overwrite
        )
    
    print("\nCreating HuggingFace dataset...")
    dataset = create_video_frames_dataset(
        args.frames_dir,
        dataset_name=args.dataset_name
    )
    print(dataset.info)
    print(f"\nSaving dataset to {args.dataset_dir}")
    dataset.save_to_disk(args.dataset_dir)
    
    print(f"\nDataset creation complete!")
    print(f"Total number of videos: {len(dataset)}")
    print(f"Features: {dataset.features}")
    print("Frame counts:", dataset["num_frames"])
    print(f"Dataset name: {dataset.info.dataset_name}")


if __name__ == "__main__":
    main()
