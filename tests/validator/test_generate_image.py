import pytest
from unittest.mock import patch, MagicMock
from bitmind.synthetic_image_generation.synthetic_image_generator import SyntheticImageGenerator
from bitmind.constants import DIFFUSER_NAMES
from PIL import Image


@pytest.fixture
def mock_diffuser():
    """
    Fixture to mock the diffuser models (StableDiffusionXLPipeline and FluxPipeline).

    Returns:
        MagicMock: A mock object representing the diffuser pipeline.
    """
    with patch('bitmind.synthetic_image_generation.synthetic_image_generator.StableDiffusionXLPipeline') as mock_sdxl:
        with patch('bitmind.synthetic_image_generation.synthetic_image_generator.FluxPipeline') as mock_flux:
            mock_pipeline = MagicMock()
            test_image = Image.new('RGB', (256, 256))
            mock_pipeline.return_value = {"images": [test_image]}
            mock_pipeline.tokenizer_2 = MagicMock()
            mock_sdxl.from_pretrained.return_value = mock_pipeline
            mock_flux.from_pretrained.return_value = mock_pipeline
            yield mock_pipeline


@pytest.fixture
def mock_image_annotation_generator():
    """
    Fixture to mock the ImageAnnotationGenerator.

    Returns:
        MagicMock: A mock object representing the ImageAnnotationGenerator.
    """
    with patch('bitmind.synthetic_image_generation.synthetic_image_generator.ImageAnnotationGenerator') as mock:
        instance = mock.return_value
        instance.process_image.return_value = [{'description': 'A test caption'}]
        yield instance


@pytest.mark.parametrize("diffuser_name", DIFFUSER_NAMES)
def test_generate_image_with_diffusers(mock_diffuser, mock_image_annotation_generator, diffuser_name):
    """
    Test the image generation process using different diffusion models.

    This test verifies the functionality of the SyntheticImageGenerator class,
    specifically its generate method. It checks the correct initialization,
    proper loading of the diffuser model, successful generation of synthetic images,
    and the correct structure and content of the generated data.

    Args:
        mock_diffuser (MagicMock): Mocked diffuser pipeline.
        mock_image_annotation_generator (MagicMock): Mocked image annotation generator.
        diffuser_name (str): Name of the diffuser model to test.

    The test performs the following:
    1. Initializes SyntheticImageGenerator with specific parameters.
    2. Mocks the load_diffuser method and sets the diffuser attribute.
    3. Generates synthetic image data.
    4. Verifies the structure and content of the generated data.
    5. Checks for correct method calls and interactions.

    This test is useful for:
    - Validating the image generation process
    - Integration testing with different diffuser models
    """
    generator = SyntheticImageGenerator(
        prompt_type='annotation',
        use_random_diffuser=False,
        diffuser_name=diffuser_name
    )
    
    real_image = {'source': 'test_dataset', 'id': 'test_id'}
    
    with patch.object(generator, 'load_diffuser') as mock_load_diffuser:
        mock_load_diffuser.return_value = mock_diffuser
        generator.diffuser = mock_diffuser
        
        mock_output = MagicMock()
        mock_output.images = [Image.new('RGB', (256, 256))]
        mock_diffuser.return_value = mock_output
        
        with patch.object(generator, 'get_tokenizer_with_min_len') as mock_get_tokenizer:
            mock_get_tokenizer.return_value = (MagicMock(), 77)
            
            generated_data = generator.generate(k=1, real_images=[real_image])
        
    assert isinstance(generated_data, list)
    assert len(generated_data) == 1
    
    image_data = generated_data[0]
    assert isinstance(image_data, dict)
    assert 'prompt' in image_data
    assert 'image' in image_data
    assert 'id' in image_data
    assert 'gen_time' in image_data
    
    assert isinstance(image_data['image'], Image.Image)
    assert image_data['prompt'] == "A test caption"
    assert isinstance(image_data['id'], str)
    assert isinstance(image_data['gen_time'], float)

    mock_image_annotation_generator.process_image.assert_called_once()
    mock_load_diffuser.assert_called_once_with(diffuser_name)

    assert image_data['image'].size == (256, 256)
    assert image_data['image'].mode == 'RGB'


if __name__ == "__main__":
    pytest.main()
