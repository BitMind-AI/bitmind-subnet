import pytest
import os
from unittest.mock import patch, MagicMock, call
from bitmind.validator.verify_models import is_model_cached, main
from bitmind.validator.config import T2I_MODEL_NAMES, IMAGE_ANNOTATION_MODEL, TEXT_MODERATION_MODEL

@pytest.fixture
def mock_expanduser():
    """
    Fixture to mock os.path.expanduser.

    Returns:
        MagicMock: A mock object representing os.path.expanduser.
    """
    with patch('os.path.expanduser') as mock:
        mock.return_value = '/mock/home/.cache/huggingface/'
        yield mock

@pytest.fixture
def mock_isdir():
    """
    Fixture to mock os.path.isdir.

    Returns:
        MagicMock: A mock object representing os.path.isdir.
    """
    with patch('os.path.isdir') as mock:
        yield mock

def test_is_model_cached(mock_expanduser, mock_isdir):
    """
    Test the is_model_cached function.

    This test verifies the functionality of the is_model_cached function
    under different scenarios of model caching.

    Args:
        mock_expanduser (MagicMock): Mocked os.path.expanduser function.
        mock_isdir (MagicMock): Mocked os.path.isdir function.

    The test performs the following:
    1. Checks if the function correctly identifies a cached model.
    2. Verifies the correct path is checked for the model cache.
    3. Checks if the function correctly identifies a non-cached model.

    This test is useful for:
    - Ensuring correct behavior of model cache checking
    - Verifying path construction for model cache
    """
    mock_isdir.return_value = True
    assert is_model_cached('test/model') == True
    mock_isdir.assert_called_with('/mock/home/.cache/huggingface/models--test--model')

    mock_isdir.return_value = False
    assert is_model_cached('test/model') == False

@patch('bitmind.validator.verify_models.SyntheticImageGenerator')
@patch('bitmind.validator.verify_models.is_model_cached')
def test_main(mock_is_model_cached, MockSyntheticImageGenerator):
    """
    Test the main function of the verify_models module.

    This test verifies the behavior of the main function, particularly
    its interaction with SyntheticImageGenerator under various scenarios.

    Args:
        mock_is_model_cached (MagicMock): Mocked is_model_cached function.
        MockSyntheticImageGenerator (MagicMock): Mocked SyntheticImageGenerator class.

    The test performs the following:
    1. Simulates a scenario where no models are cached.
    2. Calls the main function.
    3. Verifies that SyntheticImageGenerator is called with correct parameters
       for different model types and diffuser names.

    This test is useful for:
    - Ensuring correct initialization of SyntheticImageGenerator
    - Verifying handling of different model types and diffuser names
    - Regression testing for the main function's behavior
    """
    # Setup mock_is_model_cached to simulate caching behavior
    # Assume no models are cached
    mock_is_model_cached.side_effect = lambda model_name: False

    # Call the main function
    main()

    # Expected calls with varying parameters based on model type
    expected_calls = [
        call(prompt_type='annotation', use_random_diffuser=True, diffuser_name=None),  # For IMAGE_ANNOTATION_MODEL and TEXT_MODERATION_MODEL
        *[call(prompt_type='annotation', use_random_diffuser=False, diffuser_name=name) for name in T2I_MODEL_NAMES]  # For each name in T2I_MODEL_NAMES
    ]

    # Verify all calls to SyntheticImageGenerator with the correct parameters
    MockSyntheticImageGenerator.assert_has_calls(expected_calls, any_order=True)

if __name__ == "__main__":
    pytest.main()