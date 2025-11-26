from PIL import Image, ImageFilter, ImageOps


def preprocess_icon_image(image: Image.Image) -> Image.Image:
    """
    Normalize icon by converting to grayscale, removing extra padding,
    and sharpening the final result.
    """

    gray = image.convert("L")
    contrasted = ImageOps.autocontrast(gray, cutoff=2)
    inverted = ImageOps.invert(contrasted)
    bbox = inverted.getbbox()
    cropped = image.crop(bbox) if bbox else image

    resized = cropped.resize(
        (max(64, cropped.width), max(64, cropped.height)),
        Image.Resampling.LANCZOS,
    )
    sharpened = resized.filter(ImageFilter.SHARPEN)
    return sharpened.convert("RGB")


