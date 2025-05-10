import argparse
from threading import Thread
from typing import Callable, List
import bittensor as bt
import copy
import inspect
import traceback

from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from nest_asyncio import asyncio
from substrateinterface import SubstrateInterface
import signal

from bitmind import (
    __spec_version__ as spec_version,
)
from bitmind.metagraph import run_block_callback_thread
from bitmind.types import NeuronType
from bitmind.utils import ExitContext, on_block_interval
from bitmind.config import (
    add_args,
    add_validator_args,
    add_miner_args,
    add_proxy_args,
    validate_config_and_neuron_path,
)


class BaseNeuron:
    config: "bt.config"
    neuron_type: NeuronType
    exit_context = ExitContext()
    next_sync_block = None
    block_callbacks: List[Callable] = []
    substrate_thread: Thread

    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    @on_block_interval("epoch_length")
    async def maybe_sync_metagraph(self, block):
        self.check_registered()
        bt.logging.info("Resyncing Metagraph")
        self.metagraph.sync(subtensor=self.subtensor)

        if self.neuron_type == NeuronType.VALIDATOR:
            bt.logging.info("Metagraph updated, re-syncing hotkeys and moving averages")
            self.eval_engine.sync_to_metagraph()

    async def run_callbacks(self, block):
        if (
            hasattr(self, "initialization_complete")
            and not self.initialization_complete
        ):
            bt.logging.debug(
                f"Skipping callbacks at block {block} during initialization"
            )
            return

        for callback in self.block_callbacks:
            try:
                res = callback(block)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                bt.logging.error(
                    f"Failed running callback {callback.__name__}: {str(e)}"
                )
                bt.logging.error(traceback.format_exc())

    def __init__(self, config=None):
        bt.logging.info(
            f"Bittensor Version: {bt.__version__} | SN34 Version {spec_version}"
        )

        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        add_args(parser)

        if self.neuron_type == NeuronType.VALIDATOR:
            bt.axon.add_args(parser)
            add_validator_args(parser)
        if self.neuron_type == NeuronType.VALIDATOR_PROXY:
            add_validator_args(parser)
            add_proxy_args(parser)
        if self.neuron_type == NeuronType.MINER:
            bt.axon.add_args(parser)
            add_miner_args(parser)

        self.config = bt.config(parser)
        if config:
            base_config = copy.deepcopy(config)
            self.config.merge(base_config)

        validate_config_and_neuron_path(self.config)

        ## Add kill signals
        signal.signal(signal.SIGINT, self.exit_context.startExit)
        signal.signal(signal.SIGTERM, self.exit_context.startExit)

        ## LOGGING
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_info()
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        ## BITTENSOR INITIALIZATION
        bt.logging.success(self.config)
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(
            config=self.config, network=self.config.subtensor.chain_endpoint
        )
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        self.loop = asyncio.get_event_loop()
        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")
        bt.logging.debug(f"Metagraph: {self.metagraph}")

        ## CHECK IF REGG'D
        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        ## Substrate, Subtensor and Metagraph
        self.substrate = SubstrateInterface(
            ss58_format=SS58_FORMAT,
            use_remote_preset=True,
            url=self.config.subtensor.chain_endpoint,
            type_registry=TYPE_REGISTRY,
        )

        self.block_callbacks.append(self.maybe_sync_metagraph)
        self.substrate_thread = run_block_callback_thread(
            self.substrate, self.run_callbacks
        )
