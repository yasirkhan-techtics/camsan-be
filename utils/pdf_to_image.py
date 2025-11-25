from pdf2image import convert_from_path
from PIL import Image
import os

def pdf_to_images(
    pdf_path,
    dpi=300,
    modify_fn=None  # optional function(img)->modified_img
):
    """
    Convert PDF to images one page at a time (low-memory).
    Saves output to /output_images next to the PDF.

    Returns:
        List of saved image paths.
    """

    # Where the PDF is located
    base_dir = os.path.dirname(os.path.abspath(pdf_path))
    output_folder = os.path.join(base_dir, "output_images")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_paths = []

    # Get total number of pages first (light operation)
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)
    total_pages = info["Pages"]

    # Process each page individually
    for page_num in range(1, total_pages + 1):
        # Convert ONLY the current page
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )

        img = images[0]  # only single page returned

        # Apply processing if provided
        if modify_fn is not None:
            img = modify_fn(img)

        out_path = os.path.join(output_folder, f"page_{page_num}.png")
        img.save(out_path, "PNG")

        saved_paths.append(out_path)

        # Manually free memory
        del img
        del images

    return saved_paths
