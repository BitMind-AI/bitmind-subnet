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
import wandb
import time

from neurons.validator_proxy import ValidatorProxy
from bitmind.validator import forward
from bitmind.base.validator import BaseValidatorNeuron
from bitmind.synthetic_image_generation.synthetic_image_generator import SyntheticImageGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.constants import VALIDATOR_DATASET_META, WANDB_PROJECT, WANDB_ENTITY
import bitmind


class Validator(BaseValidatorNeuron):
    """
    The BitMind Validator's `forward` function sends single-image challenges to miners every 30 seconds, where each
    image has a 50/50 chance of being real or fake. In service of this task, the Validator class has two key members -
    self.real_image_datasets and self.synthetic_image_generator. The former is a list of ImageDataset objects, which
    contain real images. The latter is an ML pipeline that combines an LLM for prompt generation and diffusion
    models that ingest prompts output by the LLM to produce synthetic images.

    The BitMind Validator also encapsuluates a ValidatorProxy, which is used to service organic requests from
    our consumer-facing application. If you wish to participate in this system, run your validator with the
     --proxy.port argument set to an exposed port on your machine.
    """
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.last_responding_miner_uids = []
        self.validator_proxy = ValidatorProxy(self)
        
        bt.logging.info("init_wandb()")
        self.init_wandb()
        
        bt.logging.info("Loading real datasets")
        self.real_image_datasets = [
            ImageDataset(ds['path'], 'train', ds.get('name', None))
            for ds in VALIDATOR_DATASET_META['real']
        ]
        self.total_real_images = sum([
            len(ds) for ds in self.real_image_datasets
        ])

        self.synthetic_image_generator = SyntheticImageGenerator(
            prompt_type='annotation', use_random_diffuser=True, diffuser_name=None)

        self._fake_prob = self.config.get('fake_prob', 0.5)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)

    def init_wandb(self):
        if self.config.wandb.off:
            return

        run_name = f'validator-{self.uid}-{bitmind.__version__}'
        self.config.run_name = run_name
        self.config.uid = self.uid
        self.config.hotkey = self.wallet.hotkey.ss58_address
        self.config.version = bitmind.__version__
        self.config.type = self.neuron_type

        # Initialize the wandb run for the single project
        print("Initializing W&B")
        try:
            run = wandb.init(
                name=run_name,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config=self.config,
                dir=self.config.full_path,
                reinit=True
            )
        except wandb.UsageError as e:
            bt.logging.warning(e)
            bt.logging.warning("Did you run wandb login?")
            return

        # Sign the run to ensure it's from the correct hotkey
        signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config, allow_val_change=True)

        bt.logging.success(f"Started wandb run for project '{WANDB_PROJECT}'")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running | uid {validator.uid} | {time.time()}")
            time.sleep(5)