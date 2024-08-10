import pytest

from bitmind.utils.mock import create_random_image
from tests.fixtures.image_transforms import (
    TRANSFORMS,
    TRANSFORM_PIPELINES
)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_create_transform(transform):
    tform = transform()
    assert tform is not None


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_transform_has_expected_methods(transform):
    tform = transform()
    has_forward = hasattr(tform, 'forward')
    has_call = hasattr(tform, '__call__')
    assert has_call or has_forward
    if has_call:
        assert callable(getattr(tform, '__call__'))
    elif has_forward:
        assert callable(getattr(tform, 'forward'))


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_invoke_transform(transform):
    image = create_random_image()

    try:
        transformed_image = transform()(image)
    except Exception as e:
        pytest.fail(f"transform pipeline invocation raised an exception: {e}")

    assert transformed_image is not None


@pytest.mark.parametrize("transform_pipeline", TRANSFORM_PIPELINES)
def test_invoke_transform_pipeline(transform_pipeline):
    image = create_random_image()

    try:
        transformed_image = transform_pipeline(image)
    except Exception as e:
        pytest.fail(f"transform pipeline invocation raised an exception: {e}")

    assert transformed_image is not None


