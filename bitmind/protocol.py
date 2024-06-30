
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

from typing import List
from pydantic import root_validator, validator
from io import BytesIO
from PIL import Image
import bittensor as bt
import pydantic
import base64


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
    return ImageSynapse(image=b64_encoded_image, prediction=-1.)


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


class ImageSynapse(bt.Synapse):
    """
    This protocol helps in handling image/prediction request and response communication between
    the miner and the validator.

    Attributes:
    - image: a bas64 encoded images
    - predictions: a floats indicating the probabilty that the image is AI generated/modified. 
        >.5 is considered generated/modified, <= 0.5 is considered real.
    """

    # Required request input, filled by sending dendrite caller.
    image: str = pydantic.Field(
        title="Image",
        description="A base64 encoded image",
        allow_mutation=False
    )

    # Optional request output, filled by receiving axon.
    prediction: float = pydantic.Field(
        title="Prediction",
        description="Probability that the image is AI generated/modified",
        allow_mutation=True
    )

    def deserialize(self) -> List[float]:
        """
        Deserialize the output. This method retrieves the response from
        the miner, deserializes it and returns it as the output of the dendrite.query() call.

        Returns:
        - List[float]: The deserialized response, which in this case is the list of deepfake
        prediction probabilities
        """
        return self.prediction
