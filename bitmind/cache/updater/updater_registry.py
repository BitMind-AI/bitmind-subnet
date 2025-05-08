from typing import Dict, Optional

import bittensor as bt

from bitmind.cache.updater import BaseUpdater


class UpdaterRegistry:
    """
    Registry for cache updaters.
    """

    def __init__(self):
        self._updaters: Dict[str, BaseUpdater] = {}

    def register(self, name: str, updater: BaseUpdater) -> None:
        if name in self._updaters:
            bt.logging.warning(f"Updater {name} already registered, will be replaced")
        self._updaters[name] = updater

    def get(self, name: str) -> Optional[BaseUpdater]:
        return self._updaters.get(name)

    def get_all(self) -> Dict[str, BaseUpdater]:
        return dict(self._updaters)

    def deregister(self, name: str) -> None:
        if name in self._updaters:
            del self._updaters[name]
