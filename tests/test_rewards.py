import pytest
import numpy as np

from bitmind.validator.reward import get_rewards


@pytest.mark.parametrize(
    "label, responses, expected_rewards",
    [
        (
            1.,
            [0.5897, 0.3216, 0.7943, 0.8807, 0.2457, 0.2585, 0.3706, 0.5042, 0.9864, 0.6010],
            [1., 0., 1., 1., 0., 0., 0., 1., 1., 1.]
        ), (
            0.,
            [0.5897, 0.3216, 0.7943, 0.8807, 0.2457, 0.2585, 0.3706, 0.5042, 0.9864, 0.6010],
            [0., 1., 0., 0., 1., 1., 1., 0., 0., 0.]
        )
    ]
)
def test_rewards(label, responses, expected_rewards):
    rewards = get_rewards(label, responses)
    assert np.allclose(rewards, np.array(expected_rewards)), \
        f"Expected rewards {expected_rewards} but got {rewards}"
