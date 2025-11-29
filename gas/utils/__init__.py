from .utils import (
    print_info,
    fail_with_none,
    on_block_interval,
    ExitContext,
    get_metadata,
    get_file_modality,
    run_in_thread,
)

from .metagraph import (
    get_miner_uids,
    create_set_weights,
)

from .autoupdater import autoupdate

from .state_manager import (
    StateManager,
    save_validator_state,
    load_validator_state,
)


# Lazy imports for transforms to avoid pulling in heavy dependencies (diffusers, etc.)
# These are only loaded when explicitly accessed
def __getattr__(name):
    if name in (
        "apply_random_augmentations",
        "get_base_transforms", 
        "get_random_augmentations",
        "get_random_augmentations_medium",
        "get_random_augmentations_hard",
    ):
        from .transforms import (
            apply_random_augmentations,
            get_base_transforms,
            get_random_augmentations,
            get_random_augmentations_medium,
            get_random_augmentations_hard,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core utilities
    "print_info",
    "fail_with_none", 
    "on_block_interval",
    "ExitContext",
    "get_metadata",
    "get_file_modality",
    "run_in_thread",
    # Metagraph utilities
    "get_miner_uids",
    "create_set_weights",
    # Autoupdater
    "autoupdate",
    # State management
    "StateManager",
    "save_validator_state",
    "load_validator_state",
    # Transforms (lazy loaded)
    "apply_random_augmentations",
    "get_base_transforms",
    "get_random_augmentations",
    "get_random_augmentations_medium", 
    "get_random_augmentations_hard",
] 