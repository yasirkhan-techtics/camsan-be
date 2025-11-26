import os

from fastapi import Depends
from PIL import Image

from lib.image_processing import preprocess_icon_image


class PreprocessingService:
    """Handles icon preprocessing such as cropping and shearing."""

    def preprocess_icon(self, local_path: str) -> str:
        image = Image.open(local_path)
        processed = preprocess_icon_image(image)
        base, ext = os.path.splitext(local_path)
        processed_path = f"{base}_preprocessed{ext or '.png'}"
        processed.save(processed_path)
        # Close file handles to avoid Windows lock
        processed.close()
        image.close()
        return processed_path


def get_preprocessing_service() -> PreprocessingService:
    return PreprocessingService()


