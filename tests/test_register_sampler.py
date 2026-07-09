"""Tests for gas.generation.prompts.register_sampler."""

from gas.generation.prompts.register_sampler import (
    REGISTERS,
    sample_spec,
)


def test_sample_spec_fields():
    spec = sample_spec(modality="video", rng_seed=7)
    assert spec.register in {r.name for r in REGISTERS}
    assert spec.length_band in {"short", "medium", "long"}
    assert spec.event_count in (0, 1, 2)
    assert isinstance(spec.camera_motion, str) and spec.camera_motion
    assert spec.length_words[0] < spec.length_words[1]
    assert spec.style_strictness in {"plain", "free"}


def test_deterministic_with_seed():
    a = sample_spec(modality="video", rng_seed=42)
    b = sample_spec(modality="video", rng_seed=42)
    assert a == b


def test_distribution_not_degenerate():
    specs = [sample_spec(modality="video", rng_seed=i) for i in range(300)]
    regs = {s.register for s in specs}
    assert len(regs) >= 8, f"only {len(regs)} registers in 300 draws"
    shorts = sum(s.length_band == "short" for s in specs)
    assert shorts > 30, f"only {shorts} short prompts in 300 draws"
    no_event = sum(s.event_count == 0 for s in specs)
    assert 0 < no_event < 300


def test_image_modality_has_no_events_or_camera_motion():
    spec = sample_spec(modality="image", rng_seed=5)
    assert spec.event_count == 0
    assert spec.camera_motion == ""


def test_cctv_always_static():
    static_specs = [
        s for s in (sample_spec(modality="video", rng_seed=i) for i in range(500))
        if s.register == "cctv_surveillance"
    ]
    assert static_specs, "cctv never sampled in 500 draws"
    assert all(s.camera_motion == "static" for s in static_specs)


def test_plain_registers_have_banned_phrases():
    specs = [sample_spec(modality="video", rng_seed=i) for i in range(200)]
    plains = [s for s in specs if s.style_strictness == "plain"]
    assert plains
    assert all(s.banned_phrases for s in plains)


def test_weights_override():
    specs = [
        sample_spec(
            modality="video",
            rng_seed=i,
            weights_override={"dashcam": 1.0},
        )
        for i in range(50)
    ]
    assert all(s.register == "dashcam" for s in specs)


def test_to_dict_roundtrip_json():
    import json

    spec = sample_spec(modality="video", rng_seed=1)
    blob = json.dumps(spec.to_dict())
    assert json.loads(blob)["register"] == spec.register
