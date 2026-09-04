import asyncio
from types import SimpleNamespace

import neurons.discriminator.push_model as push_model
from neurons.discriminator.push_model import (
    _accept_already_uploaded,
    push_separate_models,
    should_retry_register,
)


def test_default_retries_stop_after_three():
    assert should_retry_register(1, 3) is True
    assert should_retry_register(2, 3) is True
    assert should_retry_register(3, 3) is False


def test_zero_retries_is_unlimited():
    assert should_retry_register(1, 0) is True
    assert should_retry_register(99, 0) is True


def test_already_uploaded_is_success():
    result = {"already_uploaded": True, "success": False}
    assert _accept_already_uploaded("image", result, skip_chain=False) is True
    assert _accept_already_uploaded("image", result, skip_chain=True) is True


def test_chain_registration_failure_does_not_fail_push(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.touch()
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))

    monkeypatch.setattr(
        push_model,
        "upload_single_modality",
        lambda *args, **kwargs: {
            "success": True,
            "r2_key": "model-key",
            "file_hash": "file-hash",
        },
    )
    monkeypatch.setattr(push_model.bt, "Subtensor", lambda **kwargs: object())

    class FailingMetadataStore:
        async def store_model_metadata(self, wallet, model_id):
            raise RuntimeError("chain unavailable")

    monkeypatch.setattr(
        push_model,
        "ChainModelMetadataStore",
        lambda subtensor, netuid: FailingMetadataStore(),
    )

    success = asyncio.run(
        push_separate_models(
            image_model_path=str(model_path),
            wallet=wallet,
            retry_delay_secs=0,
            max_retries=1,
        )
    )

    assert success is True


def test_already_uploaded_without_r2_key_still_succeeds(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.touch()
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))

    monkeypatch.setattr(
        push_model,
        "upload_single_modality",
        lambda *args, **kwargs: {
            "success": False,
            "already_uploaded": True,
            "file_hash": "file-hash",
        },
    )
    monkeypatch.setattr(push_model.bt, "Subtensor", lambda **kwargs: object())
    monkeypatch.setattr(
        push_model,
        "ChainModelMetadataStore",
        lambda subtensor, netuid: object(),
    )

    success = asyncio.run(
        push_separate_models(
            image_model_path=str(model_path),
            wallet=wallet,
            retry_delay_secs=0,
            max_retries=1,
        )
    )

    assert success is True
