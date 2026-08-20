import asyncio
from contextlib import suppress

import pytest

from gas.utils.metagraph import BlockSubscriptionStale, SubstrateConnectionManager


def make_manager():
    return SubstrateConnectionManager("ws://example.test", 42, {})


def test_stale_subscription_is_cancelled_and_reaped():
    async def run():
        manager = make_manager()
        manager.BLOCK_STALENESS_TIMEOUT = 0.01
        subscription_cancelled = asyncio.Event()

        async def hung_subscription(_callback):
            try:
                await asyncio.Future()
            finally:
                subscription_cancelled.set()

        async def callback(_block):
            pass

        manager._connect_and_subscribe = hung_subscription

        with pytest.raises(BlockSubscriptionStale):
            await manager._run_subscription_attempt(callback)

        assert subscription_cancelled.is_set()

    asyncio.run(run())


def test_cancelling_manager_cancels_inner_subscription():
    async def run():
        manager = make_manager()
        subscription_started = asyncio.Event()
        subscription_cancelled = asyncio.Event()

        async def hung_subscription(_callback):
            subscription_started.set()
            try:
                await asyncio.Future()
            finally:
                subscription_cancelled.set()

        async def callback(_block):
            pass

        manager._connect_and_subscribe = hung_subscription
        manager.task = asyncio.create_task(manager.start_subscription(callback))
        await subscription_started.wait()

        manager.stop()
        with suppress(asyncio.CancelledError):
            await manager.task

        assert subscription_cancelled.is_set()
        assert manager.running is False

    asyncio.run(run())


def test_block_activity_keeps_subscription_healthy():
    async def run():
        manager = make_manager()
        manager.BLOCK_STALENESS_TIMEOUT = 0.05
        delivered_blocks = []

        async def healthy_subscription(callback):
            for block in range(4):
                await callback(block)
                await asyncio.sleep(0.01)
            await asyncio.Future()

        async def callback(block):
            delivered_blocks.append(block)

        manager._connect_and_subscribe = healthy_subscription
        attempt = asyncio.create_task(manager._run_subscription_attempt(callback))
        await asyncio.sleep(0.045)

        assert not attempt.done()
        assert delivered_blocks == [0, 1, 2, 3]

        attempt.cancel()
        with suppress(asyncio.CancelledError):
            await attempt

    asyncio.run(run())
