# legend_detector.py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import google.genai as genai
from google.genai.types import GenerateContentConfig


PROMPT = (
    "Accurately detect and localize the region in the drawing that contains the legend section. "
    "The box_2d should be [ymin, xmin, ymax, xmax] normalized within a 0–1000 scale. "
    "Return only one bounding box representing the complete legend area. "
    "Please be accurate and recheck before generating the response."
)


def detect_legend_bbox(image_path, model_name="gemini-2.5-flash"):
    """
    Detects legend bounding box from an engineering drawing using Gemini Vision.

    Args:
        image_path (str): path to input image
        model_name (str): Gemini model name

    Returns:
        bbox_abs (tuple): (x1, y1, x2, y2) absolute pixel coordinates
        cropped_image (PIL.Image): cropped legend image
        bbox_norm (list): [ymin, xmin, ymax, xmax] normalized 0–1000
    """

    # Load image
    image = Image.open(image_path)
    width, height = image.size

    # Initialize client
    client = genai.Client()

    config = GenerateContentConfig(
        response_mime_type="application/json"
    )

    # Call Gemini Vision
    response = client.models.generate_content(
        model=model_name,
        contents=[image, PROMPT],
        config=config
    )

    # Parse JSON response
    bbox_norm = response.text.strip()
    bbox_norm = eval(bbox_norm)     # MUST be JSON: [ymin, xmin, ymax, xmax]

    ymin, xmin, ymax, xmax = bbox_norm

    # Convert normalized 0–1000 → absolute pixels
    y1 = int((ymin / 1000) * height)
    x1 = int((xmin / 1000) * width)
    y2 = int((ymax / 1000) * height)
    x2 = int((xmax / 1000) * width)

    bbox_abs = (x1, y1, x2, y2)

    # Crop region
    cropped_image = image.crop((x1, y1, x2, y2))

    return bbox_abs, cropped_image, bbox_norm


# OPTIONAL — Test
if __name__ == "__main__":
    bbox_abs, cropped, bbox_norm = detect_legend_bbox("output_images/page_5.jpg")
    print("Normalized BBox:", bbox_norm)
    print("Absolute BBox:", bbox_abs)

    plt.imshow(cropped)
    plt.axis("off")
    plt.show()

