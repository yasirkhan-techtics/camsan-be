import os
from typing import Callable, List, Optional

from pdf2image import convert_from_path


def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    modify_fn: Optional[Callable] = None,
    poppler_path: Optional[str] = None,
) -> List[str]:
    """
    Convert PDF to images one page at a time (low-memory).

    Returns list of saved image paths located alongside the PDF in an
    `output_images` folder.
    """

    base_dir = os.path.dirname(os.path.abspath(pdf_path))
    output_folder = os.path.join(base_dir, "output_images")
    os.makedirs(output_folder, exist_ok=True)

    saved_paths: List[str] = []

    from pdf2image.pdf2image import pdfinfo_from_path

    print(f"🔍 PDF Utils - poppler_path parameter: {poppler_path}")
    print(f"🔍 PDF Utils - poppler_path exists: {os.path.exists(poppler_path) if poppler_path else 'N/A'}")
    
    info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=poppler_path)
    total_pages = info["Pages"]

    for page_num in range(1, total_pages + 1):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            poppler_path=poppler_path,
        )

        img = images[0]

        if modify_fn is not None:
            img = modify_fn(img)

        out_path = os.path.join(output_folder, f"page_{page_num}.png")
        img.save(out_path, "PNG")
        saved_paths.append(out_path)

        del img
        del images

    return saved_paths


