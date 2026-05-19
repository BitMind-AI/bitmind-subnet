__all__ = ["ContentManager", "MediaStorage", "PromptEntry", "MediaEntry", "Media"]


def __getattr__(name):
    if name == "ContentManager":
        from .content_manager import ContentManager
        return ContentManager
    if name == "MediaStorage":
        from .media_storage import MediaStorage
        return MediaStorage
    if name in {"Media", "PromptEntry", "MediaEntry"}:
        from .types import Media, PromptEntry, MediaEntry
        return {
            "Media": Media,
            "PromptEntry": PromptEntry,
            "MediaEntry": MediaEntry,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
