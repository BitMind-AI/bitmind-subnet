from neurons.discriminator.push_model import (
    _accept_already_uploaded,
    should_retry_register,
)


def test_default_retries_stop_after_three():
    assert should_retry_register(1, 3) is True
    assert should_retry_register(2, 3) is True
    assert should_retry_register(3, 3) is False


def test_zero_retries_is_unlimited():
    assert should_retry_register(1, 0) is True
    assert should_retry_register(99, 0) is True


def test_already_uploaded_is_success_when_skipping_chain():
    result = {"already_uploaded": True, "success": False}
    assert _accept_already_uploaded("image", result, skip_chain=True) is True


def test_already_uploaded_fails_when_chain_register_needs_r2_key():
    result = {"already_uploaded": True, "success": False}
    assert _accept_already_uploaded("image", result, skip_chain=False) is False
