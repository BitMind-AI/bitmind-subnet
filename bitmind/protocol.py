# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 Bitmind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Union
from pydantic import BaseModel, Field
from torchvision import transforms
from io import BytesIO
from PIL import Image
import bittensor as bt
import numpy as np
import base64
import pydantic
import torch
import zlib

from bitmind.validator.config import TARGET_IMAGE_SIZE
from bitmind.utils.image_transforms import get_base_transforms
base_transforms = get_base_transforms(TARGET_IMAGE_SIZE)


# ---- miner ----
# Example usage:
#   def miner_forward( synapse: ImageSynapse ) -> ImageSynapse:
#       ...
#       synapse.predictions = deepfake_detection_model_outputs
#       return synapse
#   axon = bt.axon().attach( miner_forward ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   b64_images = [b64_img_1, ..., b64_img_n]
#   predictions = dendrite.query( ImageSynapse( images = b64_images ) )
#   assert len(predictions) == len(b64_images)

def prepare_synapse(input_data, modality):
    if isinstance(input_data, torch.Tensor):
        input_data = transforms.ToPILImage()(input_data.cpu().detach())
    if isinstance(input_data, list) and isinstance(input_data[0], torch.Tensor):
        for i, img in enumerate(input_data):
            input_data[i] = transforms.ToPILImage()(img.cpu().detach())

    if modality == 'image':
        return prepare_image_synapse(input_data)
    elif modality == 'video':
        return prepare_video_synapse(input_data)
    else:
        raise NotImplementedError(f"Unsupported modality: {modality}")


def prepare_image_synapse(image: Image):
    """
    Prepares an image for use with ImageSynapse object.

    Args:
        image (Image): The input image to be prepared.

    Returns:
        ImageSynapse: An instance of ImageSynapse containing the encoded image and a default prediction value.
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    b64_encoded_image = base64.b64encode(image_bytes.getvalue())
    return ImageSynapse(image=b64_encoded_image)


class ImageSynapse(bt.Synapse):
    """
    This protocol helps in handling image/prediction request and response communication between
    the miner and the validator.

    Attributes:
    - image: a bas64 encoded images
    - prediction: a float  indicating the probabilty that the image is AI generated/modified.
        >.5 is considered generated/modified, <= 0.5 is considered real.
    """

    testnet_label: int = -1  # for easier miner eval on testnet

    # Required request input, filled by sending dendrite caller.
    image: str = pydantic.Field(
        title="Image",
        description="A base64 encoded image",
        default="",
        frozen=False
    )

    prediction: Union[float, List[float]] = pydantic.Field(
        title="Prediction",
        description="Probability vector for [real, synthetic, semi-synthetic] classes.",
        default=[-1., -1., -1.],
        frozen=False
    )

    def deserialize(self) -> np.ndarray:
        """
        Deserialize the output. Backwards compatible with binary float outputs.

        Returns:
        - float: The deserialized miner prediction probabilities
        """
        p = self.prediction
        if isinstance(p, float):
            if p == -1:
                return np.array([-1., -1., -1.])
            else:
                return np.array([1-p, p, 0.])
        elif isinstance(p, list):
            return np.array(p)
        else:
            raise ValueError(f"Unsupported prediction type: {type(p)}")


def prepare_video_synapse(frames: List[Image.Image]):
    """
    """
    frame_bytes = []
    for frame in frames:
        buffer = BytesIO()
        frame.save(buffer, format="JPEG")
        frame_bytes.append(buffer.getvalue())

    combined_bytes = b''.join(frame_bytes)
    compressed_data = zlib.compress(combined_bytes)
    encoded_data = base64.b85encode(compressed_data).decode('utf-8')
    return VideoSynapse(video=encoded_data)

class VideoSynapse(bt.Synapse):
    """
    Naive initial VideoSynapse 
    Better option would be to modify the Dendrite interface to allow multipart/form-data here:
    https://github.com/opentensor/bittensor/blob/master/bittensor/core/dendrite.py#L533
    Another higher lift option would be to look into Epistula or Fiber
    """

    testnet_label: int = -1  # for easier miner eval on testnet

    # Required request input, filled by sending dendrite caller.
    video: str = pydantic.Field(
        title="Video",
        description="A wildly inefficient means of sending video data",
        default="",
        frozen=False
    )

    # Optional request output, filled by receiving axon.
    prediction: Union[float, List[float]] = pydantic.Field(
        title="Prediction",
        description="Probability vector for [real, synthetic, semi-synthetic] classes.",
        default=[-1., -1., -1.],
        frozen=False
    )

    def deserialize(self) -> np.ndarray:
        """
        Deserialize the output. Backwards compatible with binary float outputs.

        Returns:
        - float: The deserialized miner prediction probabilities
        """
        p = self.prediction
        if isinstance(p, float):
            if p == -1:
                return np.array([-1., -1., -1.])
            else:
                return np.array([1-p, p, 0.])
        elif isinstance(p, list):
            return np.array(p)
        else:
            raise ValueError(f"Unsupported prediction type: {type(p)}")


def decode_video_synapse(synapse: VideoSynapse) -> List[torch.Tensor]:
    """
    V1 of a function for decoding a VideoSynapse object back into a list of torch tensors.

    Args:
        synapse: VideoSynapse object containing the encoded video data

    Returns:
        List of torch tensors, each representing a frame from the video
    """
    compressed_data = base64.b85decode(synapse.video.encode('utf-8'))
    combined_bytes = zlib.decompress(compressed_data)

    # Split the combined bytes into individual JPEG files
    # Look for JPEG markers: FF D8 (start) and FF D9 (end)
    frames = []
    current_pos = 0
    data_length = len(combined_bytes)

    while current_pos < data_length:
        # Find start of JPEG (FF D8)
        while current_pos < data_length - 1:
            if combined_bytes[current_pos] == 0xFF and combined_bytes[current_pos + 1] == 0xD8:
                break
            current_pos += 1

        if current_pos >= data_length - 1:
            break

        start_pos = current_pos

        # Find end of JPEG (FF D9)
        while current_pos < data_length - 1:
            if combined_bytes[current_pos] == 0xFF and combined_bytes[current_pos + 1] == 0xD9:
                current_pos += 2
                break
            current_pos += 1

        if current_pos > start_pos:
            # Extract the JPEG data
            jpeg_data = combined_bytes[start_pos:current_pos]
            try:
                img = Image.open(BytesIO(jpeg_data))
                frames.append(img)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    bt.logging.info('transforming video inputs')
    frames = base_transforms(frames)

    frames = torch.stack(frames, dim=0)
    frames = frames.unsqueeze(0)
    print(f'decoded video into tensor with shape {frames.shape}')
    return frames
