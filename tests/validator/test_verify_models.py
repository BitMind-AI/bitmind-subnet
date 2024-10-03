import pytest
import os
from unittest.mock import patch, MagicMock, call
from bitmind.validator.verify_models import is_model_cached, main
from bitmind.constants import DIFFUSER_NAMES, IMAGE_ANNOTATION_MODEL, TEXT_MODERATION_MODEL

@pytest.fixture
def mock_expanduser():
    with patch('os.path.expanduser') as mock:
        mock.return_value = '/mock/home/.cache/huggingface/'
        yield mock

@pytest.fixture
def mock_isdir():
    with patch('os.path.isdir') as mock:
        yield mock

def test_is_model_cached(mock_expanduser, mock_isdir):
    mock_isdir.return_value = True
    assert is_model_cached('test/model') == True
    mock_isdir.assert_called_with('/mock/home/.cache/huggingface/models--test--model')

    mock_isdir.return_value = False
    assert is_model_cached('test/model') == False

@patch('bitmind.validator.verify_models.SyntheticImageGenerator')
@patch('bitmind.validator.verify_models.is_model_cached')
def test_main(mock_is_model_cached, MockSyntheticImageGenerator):
    # Setup mock_is_model_cached to simulate caching behavior
    # Assume no models are cached
    mock_is_model_cached.side_effect = lambda model_name: False

    # Call the main function
    main()

    # Expected calls with varying parameters based on model type
    expected_calls = [
        call(prompt_type='annotation', use_random_diffuser=True, diffuser_name=None),  # For IMAGE_ANNOTATION_MODEL and TEXT_MODERATION_MODEL
        *[call(prompt_type='annotation', use_random_diffuser=False, diffuser_name=name) for name in DIFFUSER_NAMES]  # For each name in DIFFUSER_NAMES
    ]

    # Verify all calls to SyntheticImageGenerator with the correct parameters
    MockSyntheticImageGenerator.assert_has_calls(expected_calls, any_order=True)

if __name__ == "__main__":
    pytest.main()