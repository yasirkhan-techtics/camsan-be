import os
import tempfile
from typing import List

from fastapi import Depends, HTTPException
from PIL import Image
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models.page import PDFPage
from models.project import Project
from services.storage_service import StorageService, get_storage_service
from lib.pdf_utils import pdf_to_images


class PDFService:
    """Convert PDFs to images and persist pages."""

    def __init__(self, storage_service: StorageService):
        self.storage = storage_service
        self.settings = get_settings()

    def convert_pdf_to_images(self, db: Session, project: Project) -> List[PDFPage]:
        print(f"🔍 PDF Service - Settings poppler_path: {self.settings.poppler_path}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_pdf = os.path.join(tmp_dir, "source.pdf")
            self.storage.download_file(project.pdf_file_url, local_pdf)

            image_paths = pdf_to_images(local_pdf, poppler_path=self.settings.poppler_path)
            if not image_paths:
                raise HTTPException(status_code=500, detail="PDF conversion produced no images.")

            pages: List[PDFPage] = []
            for idx, image_path in enumerate(image_paths, start=1):
                filename = f"{project.id}_page_{idx}.png"
                url = self.storage.upload_file(
                    local_path=image_path,
                    mime_type="image/png",
                    filename=filename,
                    project_name=project.name,
                    project_id=str(project.id),
                )
                with Image.open(image_path) as img:
                    width, height = img.size
                page = PDFPage(
                    project_id=project.id,
                    page_number=idx,
                    image_url=url,
                    width=width,
                    height=height,
                    processed=True,
                )
                db.add(page)
                pages.append(page)

            db.commit()
            for page in pages:
                db.refresh(page)
            return pages


def get_pdf_service(
    storage_service: StorageService = Depends(get_storage_service),
) -> PDFService:
    return PDFService(storage_service=storage_service)


