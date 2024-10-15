import time
import asyncio
import random
import bittensor as bt
import numpy as np
from typing import List
from PIL import Image

from bitmind.constants import DIFFUSER_NAMES
from bitmind.validator.miner_performance_tracker import MinerPerformanceTracker


def create_random_image():
    random_data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(random_data)


class MockImageDataset:
    def __init__(
            self,
            huggingface_dataset_path: str,
            huggingface_datset_split: str = 'train',
            huggingface_datset_name: str = None,
            create_splits: bool = False,
            download_mode: str = None):

        self.huggingface_dataset_path = huggingface_dataset_path
        self.huggingface_datset_name = huggingface_datset_name
        self.dataset = ""
        self.sampled_images_idx = []

    def __getitem__(self, index: int) -> dict:
        return {
            'image': create_random_image(),
            'id': index,
            'source': self.huggingface_dataset_path
        }

    def sample(self, k=1):
        return [self.__getitem__(i) for i in range(k)], [i for i in range(k)]


class MockSyntheticImageGenerator:
    def __init__(self, prompt_type, use_random_diffuser, diffuser_name):
        self.prompt_type = prompt_type
        self.diffuser_name = diffuser_name
        self.use_random_diffuser = use_random_diffuser

    def generate(self, k=1, real_images=None):
        if self.use_random_diffuser:
            self.load_diffuser('random')
        else:
            self.load_diffuser(self.diffuser_name)

        return [{
            'prompt': f'mock {self.prompt_type} prompt',
            'image': create_random_image(),
            'id': i
        } for i in range(k)]

    def load_diffuser(self, diffuser_name) -> None:
        """
        loads a huggingface diffuser model.
        """
        if diffuser_name == 'random':
            diffuser_name = np.random.choice(DIFFUSER_NAMES, 1)[0]
        self.diffuser_name = diffuser_name


class MockValidator:
    def __init__(self, config):
        self.config = config
        subtensor = MockSubtensor(config.netuid, wallet=bt.MockWallet())

        self.performance_tracker = MinerPerformanceTracker()

        self.metagraph = MockMetagraph(
            netuid=config.netuid,
            subtensor=subtensor
        )
        self.dendrite = MockDendrite(bt.MockWallet())
        self.real_image_datasets = [
            MockImageDataset(
                f"fake-path/dataset-{i}",
                'train',
                None,
                False)
            for i in range(3)
        ]
        self.synthetic_image_generator = MockSyntheticImageGenerator(
            prompt_type='annotation', use_random_diffuser=True, diffuser_name=None)

        self._fake_prob = config.fake_prob

    def update_scores(self, rewards, miner_uids):
        pass


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)
        bt.MockSubtensor.reset()  # reset chain state so test cases don't interfere with one another

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            try:
                self.force_register_neuron(
                    netuid=netuid,
                    hotkey=wallet.hotkey.ss58_address,
                    coldkey=wallet.coldkey.ss58_address,
                    balance=100000,
                    stake=100000,
                )
            except Exception as e:
                print(f"Skipping force_register_neuron: {e}")

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            try:
                self.force_register_neuron(
                    netuid=netuid,
                    hotkey=f"miner-hotkey-{i}",
                    coldkey="mock-coldkey",
                    balance=100000,
                    stake=100000,
                )
            except Exception as e:
                print(f"Skipping force_register_neuron: {e}")


class MockMetagraph(bt.metagraph):
    def __init__(self, netuid, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)
        self.default_ip = "127.0.0.0"
        self.default_port = 8092

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = self.default_ip
            axon.port = self.default_port

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # Update the status code and status message of the dendrite to match the axon
                    # TODO (developer): replace with your own expected synapse data
                    s.prediction = np.random.rand(1)[0]
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    s.dendrite.process_time = str(process_time)
                else:
                    s.prediction = -1
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    s.dendrite.process_time = str(timeout)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)
