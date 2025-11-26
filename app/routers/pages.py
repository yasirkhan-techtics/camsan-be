from typing import List
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session

from database import get_db
from models.page import PDFPage
from models.project import Project
from schemas.page import PageResponse
from services.pdf_service import PDFService, get_pdf_service
from utils.state_manager import StateManager

router = APIRouter()


@router.post("/{project_id}/process-pdf", response_model=List[PageResponse])
def process_pdf(
    project_id: UUID,
    db: Session = Depends(get_db),
    pdf_service: PDFService = Depends(get_pdf_service),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    pages = pdf_service.convert_pdf_to_images(db, project)
    StateManager(db).transition(project, "pages_extracted", "pdf_processed")
    return [PageResponse.model_validate(page) for page in pages]


@router.get("/{project_id}/pages", response_model=List[PageResponse])
def list_pages(project_id: UUID, db: Session = Depends(get_db)):
    pages = (
        db.query(PDFPage).filter(PDFPage.project_id == project_id).order_by(PDFPage.page_number).all()
    )
    if not pages:
        project_exists = db.query(Project.id).filter(Project.id == project_id).first()
        if not project_exists:
            raise HTTPException(status_code=404, detail="Project not found.")
    return [PageResponse.model_validate(page) for page in pages]


