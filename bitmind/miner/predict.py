from bitmind.image_transforms import base_transforms


def predict(model, image):
    image = base_transforms(image).unsqueeze(0).float()
    out = model(image).sigmoid().flatten().tolist()
    return out[0]