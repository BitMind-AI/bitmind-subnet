"""Unit tests for generator fool-rate qualification and pay gating."""

from types import SimpleNamespace

from gas.evaluation.rewards import (
    GeneratorQualification,
    combine_generator_rewards,
    get_generator_qualification,
)


def _metagraph(*hotkeys):
    return SimpleNamespace(hotkeys=list(hotkeys))


def _row(ss58, modality, fooled, not_fooled):
    return {
        "ss58_address": ss58,
        "modality": modality,
        "fooled_count": fooled,
        "not_fooled_count": not_fooled,
    }


def test_qualifies_image_and_video_independently():
    hk_a, hk_b, hk_c = "5A", "5B", "5C"
    results = [
        _row(hk_a, "image", 2, 18),
        _row(hk_a, "video", 0, 20),
        _row(hk_b, "image", 0, 50),
        _row(hk_b, "video", 2, 18),
        _row(hk_c, "image", 3, 7),
        _row(hk_c, "video", 5, 5),
    ]

    q = get_generator_qualification(results, _metagraph(hk_a, hk_b, hk_c))

    assert q[0].qualified_image is True
    assert q[0].qualified_video is False
    assert q[0].image_rate == 0.1
    assert q[0].video_rate == 0.0

    assert q[1].qualified_image is False
    assert q[1].qualified_video is True
    assert q[1].video_rate == 0.1

    assert q[2].qualified_image is False
    assert q[2].qualified_video is False
    assert q[2].image_n == 10
    assert q[2].video_n == 10


def test_cutoff_is_exclusive_and_rows_sum_by_modality():
    hk = "5A"
    results = [
        _row(hk, "image", 1, 49),  # 2% is not > 2%
        _row(hk, "image", 0, 0),
        _row(hk, "video", 1, 99),  # 1% is not > 1%
    ]
    q = get_generator_qualification(results, _metagraph(hk))
    assert q[0].image_n == 50
    assert q[0].image_rate == 0.02
    assert q[0].qualified_image is False
    assert q[0].video_n == 100
    assert q[0].video_rate == 0.01
    assert q[0].qualified_video is False


def test_skips_unknown_hotkeys_and_modalities():
    results = [
        _row("5Unknown", "image", 10, 10),
        _row("5A", "audio", 10, 10),
        _row("5A", "image", 5, 15),
    ]
    q = get_generator_qualification(results, _metagraph("5A"))
    assert set(q) == {0}
    assert q[0].image_n == 20
    assert q[0].qualified_image is True
    assert q[0].video_n == 0


def test_empty_or_invalid_results_return_none():
    assert get_generator_qualification([], _metagraph("5A")) is None
    assert get_generator_qualification(None, _metagraph("5A")) is None
    assert get_generator_qualification({"data": []}, _metagraph("5A")) is None
    assert get_generator_qualification(["not-a-dict"], _metagraph("5A")) is None
    assert get_generator_qualification(
        [_row("5Unknown", "image", 10, 10)], _metagraph("5A")
    ) is None


def test_all_unqualified_rows_still_replace_cache():
    q = get_generator_qualification(
        [_row("5A", "image", 0, 50)], _metagraph("5A")
    )
    assert q is not None
    assert q[0].qualified_image is False
    assert q[0].image_n == 50


def test_combine_pays_only_cleared_modality():
    base = {
        0: {"image": 10.0, "video": 4.0},
        1: {"image": 8.0, "video": 20.0},
        2: {"image": 5.0, "video": 5.0},
    }
    qualification = {
        0: GeneratorQualification(qualified_image=True, qualified_video=False),
        1: GeneratorQualification(qualified_image=False, qualified_video=True),
    }

    rewards = combine_generator_rewards(base, qualification)
    assert rewards[0] == 0.30 * 10.0
    assert rewards[1] == 0.70 * 20.0
    assert 2 not in rewards


def test_combine_omits_zero_total():
    base = {0: {"image": 1.0, "video": 1.0}}
    rewards = combine_generator_rewards(base, {})
    assert rewards == {}
