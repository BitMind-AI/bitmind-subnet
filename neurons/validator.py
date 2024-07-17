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

import bittensor as bt
import time

from neurons.validator_proxy import ValidatorProxy
from bitmind.validator import forward
from bitmind.base.validator import BaseValidatorNeuron
from bitmind.random_image_generator import RandomImageGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.constants import DATASET_META


class Validator(BaseValidatorNeuron):
    """
    The BitMind Validator's `forward` function sends single-image challenges to miners every 30 seconds, where each
    image has a 50/50 chance of being real or fake. In service of this task, the Validator class has two key members -
    self.real_iamge_datasets and self.random_image_generator. The former is a list of ImageDataset objects, which
    contain real images. The latter is an ML pipeline that combines an LLM for prompt generation and diffusion
    models that ingest prompts output by the LLM to produce synthetic iamges.

    The BitMind Validator also encapsuluates a ValidatorProxy, which is used to service organic requests from
    our consumer-facing application. If you wish to participate in this system, run your validator with the
     --proxy.port argument set to an exposed port on your machine.
    """
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        print("Loading real datasets")
        self.real_image_datasets = [
            ImageDataset(ds['path'], 'test', ds.get('name', None), ds['create_splits'])
            for ds in DATASET_META['real']
        ]

        self.last_responding_miner_uids = []
        self.random_image_generator = RandomImageGenerator(use_random_diffuser=True, diffuser_name=None)
        self.validator_proxy = ValidatorProxy(self)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
