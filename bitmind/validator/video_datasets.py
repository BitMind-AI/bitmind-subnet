from pathlib import Path
import bittensor as bt
import numpy as np
import subprocess
import os    


def download_openvid1m_zips(output_directory, download_all=False, num_zips=1):
    """ Downloads a configurable number of video data zips from the OpenVid-1M huggingface dataset """
    output_path = Path(output_directory)
    error_log_path = output_path / "download_log.txt"
    zip_indices = range(0, 186)
    if not download_all:
        zip_indices = np.random.choice(zip_indices, num_zips)
    for i in zip_indices:
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
        file_path = output_path / f"OpenVid_part{i}.zip"
        if file_path.exists():
            bt.logging.warning(f"file {file_path} exits.")
            continue
        command = ["wget", "-O", str(file_path), url]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            bt.logging.info(f"file {url} saved to {file_path}")
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e.stderr}\n"
            bt.logging.error(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            
            part_urls = [
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partaa",
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partab"
            ]
            for part_url in part_urls:
                part_file_path = output_path / Path(part_url).name
                if part_file_path.exists():
                    bt.logging.warning(f"file {part_file_path} exits.")
                    continue

            part_command = ["wget", "-O", str(part_file_path), part_url]
            try:
               result = subprocess.run(part_command, check=True, capture_output=True, text=True)
               bt.logging.info(f"file {part_url} saved to {part_file_path}")
            except subprocess.CalledProcessError as part_e:
               part_error_message = f"file {part_url} download failed: {part_e.stderr}\n"
               bt.logging.error(part_error_message)
               with open(error_log_path, "a") as error_log_file:
                   error_log_file.write(part_error_message)
            file_path = output_path / f"OpenVid_part{i}.zip"
            cat_command = f"cat {output_path}/OpenVid_part{i}_part* > {file_path}"
            subprocess.run(cat_command, shell=True)
    """
    data_folder = output_path / "data" / "train"
    data_folder.mkdir(parents=True, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
    ]
    for data_url in data_urls:
        data_path = data_folder / Path(data_url).name
        command = ["wget", "-O", str(data_path), data_url]
        subprocess.run(command, check=True)
    """