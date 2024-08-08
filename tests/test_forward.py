from types import SimpleNamespace
import pytest

from bitmind.utils.mock import MockValidator
from bitmind.validator.forward import forward


@pytest.mark.asyncio
async def test_validator_forward():
    print("Configuring mock config and mock validator")
    mock_config = SimpleNamespace(
        neuron=SimpleNamespace(
            prompt_type="annotation",
            sample_size=10,
            vpermit_tao_limit=1000
        ),
        wandb=SimpleNamespace(off=True)
    )

    mock_neuron = MockValidator(mock_config)

    print("Calling forward with mock validator")
    try:
        await forward(self=mock_neuron)

    except Exception as e:
        pytest.fail(f"validator forward raised an exception: {e}")
