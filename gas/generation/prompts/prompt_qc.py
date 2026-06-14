"""Validity gate for composed prompts.

A composed prompt must look like a usable text-to-media prompt — not LLM
meta-chatter, not structural markup, not wildly off the committed length
band, and (for plain registers) free of the literary house-style tells the
register forbids. Rejected prompts are retried once by the composer with the
rejection reason appended; a second failure drops the sample.

Pure stdlib; unit-testable without models.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from gas.generation.prompts.register_sampler import PromptSpec

# Tokenizer for word-boundary-aware phrase matching.
_WORD_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())

# Tolerance around the committed word band (fraction of the bound).
_BAND_TOLERANCE = 0.20

# Case-insensitive prefixes/fragments that mark LLM meta-output rather than
# a prompt. Checked anywhere in the first 80 chars.
_META_PATTERNS = (
    "here is",
    "here's",
    "this prompt",
    "committed shot spec",
    "as an ai",
    "i cannot",
    "i can't",
    "sure!",
    "certainly",
    "note:",
    "output:",
)

_STRUCTURE_RE = re.compile(r"(^|\n)\s*([-*•]\s|\d+\.\s|#+\s)")


def validate(text: str, spec: Optional[PromptSpec]) -> Tuple[bool, str]:
    """Validate a composed prompt against its committed spec.

    Args:
        text: The cleaned composer output.
        spec: The PromptSpec it was composed under, or None (legacy path —
            only meta/markup checks apply).

    Returns:
        (ok, reason): ok=True with empty reason, or ok=False with a short
        human-readable reason suitable for appending to a retry message.
    """
    if not text or not text.strip():
        return False, "empty output"

    head = text[:80].lower()
    for pat in _META_PATTERNS:
        if pat in head:
            return False, f"meta-text detected ({pat!r})"

    if _STRUCTURE_RE.search(text):
        return False, "structural markup (bullets/headings/numbering)"

    if spec is None:
        return True, ""

    words = len(re.findall(r"\S+", text))
    lo, hi = spec.length_words
    min_ok = int(lo * (1 - _BAND_TOLERANCE))
    max_ok = int(hi * (1 + _BAND_TOLERANCE))
    if words < min_ok or words > max_ok:
        return False, (
            f"length {words} words outside committed band "
            f"{lo}-{hi} (tolerance {min_ok}-{max_ok})"
        )

    if spec.style_strictness == "plain" and spec.banned_phrases:
        words = _tokens(text)
        for phrase in spec.banned_phrases:
            phrase_tokens = _tokens(phrase)
            plen = len(phrase_tokens)
            for i in range(len(words) - plen + 1):
                if words[i : i + plen] == phrase_tokens:
                    return False, f"banned phrase for plain register: {phrase!r}"

    return True, ""
