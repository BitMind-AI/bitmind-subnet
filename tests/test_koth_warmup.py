"""Activation metadata gates discriminator payouts without changing generators."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from gas.koth_weights import (
    build_koth_weights,
    chains_by_modality,
    discriminator_emissions_enabled,
    kings_by_modality,
)


@pytest.mark.parametrize("metadata", [
    None,
    {},
    {"emissions_enabled": True},
    {"emissions_enabled": "true", "emissions_start_at": "2026-09-01T00:00:00Z"},
    {"emissions_enabled": True, "emissions_start_at": "invalid"},
    {"emissions_enabled": True, "emissions_start_at": "2026-09-01T00:00:00"},
    {"emissions_enabled": True, "emissions_start_at": 123},
])
def test_missing_or_malformed_activation_keeps_burning(metadata):
    assert not discriminator_emissions_enabled(metadata)


def test_activation_requires_elapsed_boundary_and_explicit_enabled_response():
    boundary = datetime(2026, 9, 1, tzinfo=timezone.utc)
    cached = {"emissions_enabled": False, "emissions_start_at": boundary.isoformat()}
    assert not discriminator_emissions_enabled(cached, now=boundary + timedelta(days=1))
    enabled = {**cached, "emissions_enabled": True}
    assert not discriminator_emissions_enabled(enabled, now=boundary - timedelta(seconds=1))
    assert discriminator_emissions_enabled(enabled, now=boundary)
    assert discriminator_emissions_enabled(enabled, now=boundary + timedelta(seconds=1))


def test_warmup_burns_kings_and_residuals_then_resumes_without_changing_generators():
    boundary = datetime(2026, 9, 1, tzinfo=timezone.utc)
    payload = {
        "emissions_start_at": boundary.isoformat(),
        "emissions_enabled": False,
        "kings": [{"modality": "image", "ss58_address": "current"}],
        "chain": {"image": ["current", "previous"]},
    }
    args = dict(
        n=5, scores=np.array([0., 0., 0., 2., 1.]), generator_uids=[3, 4],
        kings=kings_by_modality(payload), chains=chains_by_modality(payload),
        uid_for_hotkey={"current": 1, "previous": 2}.get, burn_uid=0,
        split={"image": .3, "video": .2, "audio": .1, "generator": .4},
    )
    warmup = build_koth_weights(
        **args, emissions_enabled=discriminator_emissions_enabled(payload, now=boundary)
    )
    payload["emissions_enabled"] = True
    live = build_koth_weights(
        **args, emissions_enabled=discriminator_emissions_enabled(payload, now=boundary)
    )
    assert warmup[1] == warmup[2] == 0
    assert warmup[0] == pytest.approx(.6)
    assert live[1] > 0 and live[2] > 0
    assert live[1] + live[2] == pytest.approx(.3)
    assert live[0] == pytest.approx(.3)  # Vacant video/audio lanes still burn.
    np.testing.assert_allclose(warmup[3:], live[3:])
    assert warmup.sum() == pytest.approx(1)
    assert live.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="burn UID is unavailable"):
        build_koth_weights(**{**args, "burn_uid": None}, emissions_enabled=False)
