"""Unit tests for KOTH validator weight vectors."""

import numpy as np
import pytest

from gas.koth_weights import (
    assign_residual_shares,
    build_koth_weights,
    chains_by_modality,
    kings_by_modality,
)

ESCROW = {
    "image": "5EUJFyH4ZSSiD3C8sM698nsVE26Tq98LoBwkmopmWZqaZqCA",
    "video": "5G6BJ1Z6LeDptRn5GTw74QSDmG1FP3eqVque5JhUb5zeEyQa",
    "audio": "5F9Qo4jqurfx3qHsC2kQtvge7Si5aW1BfYKwpxnnpVxouPyF",
}


def test_kings_by_modality_reads_hotkeys():
    payload = {
        "kings": [
            {"modality": "image", "ss58_address": "5Img"},
            {"modality": "video", "ss58_address": "5Vid"},
        ]
    }
    assert kings_by_modality(payload) == {"image": "5Img", "video": "5Vid"}
    assert kings_by_modality(None) == {}


def test_weights_go_to_king_uids_not_escrow():
    hotkeys = {"5Img": 1, "5Vid": 2, "5Aud": 3, "5Burn": 0}
    kings = {"image": "5Img", "video": "5Vid", "audio": "5Aud"}
    scores = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 1.0])

    weights = build_koth_weights(
        n=6,
        scores=scores,
        generator_uids=[4, 5],
        kings=kings,
        uid_for_hotkey=hotkeys.get,
        burn_uid=0,
    )

    assert abs(weights[1] - 0.40) < 1e-9
    assert abs(weights[2] - 0.40) < 1e-9
    assert abs(weights[3] - 0.04) < 1e-9
    assert abs(weights[4] - 0.16 * 2 / 3) < 1e-9
    assert abs(weights[5] - 0.16 * 1 / 3) < 1e-9
    assert weights[0] == 0.0
    for escrow in ESCROW.values():
        assert escrow not in kings.values()
        assert hotkeys.get(escrow) is None


def test_missing_king_burns_that_lane():
    weights = build_koth_weights(
        n=4,
        scores=np.zeros(4),
        generator_uids=[],
        kings={"image": "5Img"},
        uid_for_hotkey=lambda hk: 1 if hk == "5Img" else None,
        burn_uid=0,
    )
    assert abs(weights[1] - 0.40) < 1e-9
    assert abs(weights[0] - 0.60) < 1e-9  # video + audio + generator


def test_api_down_empty_kings_burns_discriminator_shares():
    weights = build_koth_weights(
        n=3,
        scores=np.array([0.0, 4.0, 0.0]),
        generator_uids=[1],
        kings={},
        uid_for_hotkey=lambda hk: None,
        burn_uid=0,
    )
    assert abs(weights[0] - 0.84) < 1e-9
    assert abs(weights[1] - 0.16) < 1e-9


def test_required_burn_fails_when_burn_uid_is_unavailable():
    with pytest.raises(ValueError, match="burn UID is unavailable"):
        build_koth_weights(
            n=3,
            scores=np.array([0.0, 1.0, 0.0]),
            generator_uids=[1],
            kings={},
            uid_for_hotkey=lambda hk: None,
            burn_uid=None,
        )


@pytest.mark.parametrize(
    "split",
    [
        {"image": 0.4, "video": 0.4, "audio": -0.01, "generator": 0.21},
        {"image": 0.4, "video": 0.4, "audio": 0.04},
    ],
)
def test_invalid_split_is_rejected(split):
    with pytest.raises(ValueError, match="Invalid KOTH split"):
        build_koth_weights(
            n=1,
            scores=np.zeros(1),
            generator_uids=[],
            kings={},
            uid_for_hotkey=lambda hk: None,
            burn_uid=0,
            split=split,
        )


def test_residual_rolls_unused_slots_to_current():
    assert assign_residual_shares(["5A"]) == [
        {"ss58_address": "5A", "share": 1.0, "role": "current"}
    ]
    assert assign_residual_shares(["5A", "5B"]) == [
        {"ss58_address": "5A", "share": 0.90, "role": "current"},
        {"ss58_address": "5B", "share": 0.10, "role": "previous"},
    ]


def test_chains_by_modality_recomputes_shares_and_falls_back():
    payload = {
        "kings": [{"modality": "audio", "ss58_address": "5Aud"}],
        "chain": {
            "image": [
                {"ss58_address": "5Img", "share": 0.5, "role": "current"},
                {"ss58_address": "5Prev", "share": 0.5, "role": "previous"},
            ]
        },
    }
    chains = chains_by_modality(payload)
    assert [member["share"] for member in chains["image"]] == [0.90, 0.10]
    assert chains["audio"][0]["ss58_address"] == "5Aud"
    assert chains["audio"][0]["share"] == 1.0


def test_lane_residual_splits_across_last_three_kings():
    hotkeys = {"5Img": 1, "5Prev": 2, "5Two": 3, "5Burn": 0}
    weights = build_koth_weights(
        n=5,
        scores=np.zeros(5),
        generator_uids=[],
        kings={"image": "5Img"},
        uid_for_hotkey=hotkeys.get,
        burn_uid=0,
        chains={
            "image": assign_residual_shares(["5Img", "5Prev", "5Two"]),
        },
    )
    assert abs(weights[1] - 0.40 * 0.85) < 1e-9
    assert abs(weights[2] - 0.40 * 0.10) < 1e-9
    assert abs(weights[3] - 0.40 * 0.05) < 1e-9
    assert abs(weights[0] - 0.60) < 1e-9  # video + audio + generator


def test_unresolvable_previous_king_rolls_to_current():
    hotkeys = {"5Img": 1, "5Burn": 0}
    weights = build_koth_weights(
        n=3,
        scores=np.zeros(3),
        generator_uids=[],
        kings={"image": "5Img"},
        uid_for_hotkey=hotkeys.get,
        burn_uid=0,
        chains={"image": assign_residual_shares(["5Img", "5Gone"])},
    )
    assert abs(weights[1] - 0.40) < 1e-9
    assert abs(weights[0] - 0.60) < 1e-9
