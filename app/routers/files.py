"""File serving endpoints - abstracts storage backend from frontend."""
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session

from database import get_db
from models.project import Project
from models.page import PDFPage
from models.legend import LegendTable, LegendItem
from models.detection import IconTemplate, LabelTemplate
from services.storage_service import StorageService, get_storage_service

router = APIRouter()


@router.get("/projects/{project_id}/pdf")
def get_project_pdf(
    project_id: UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Serve the project's PDF file."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.pdf_file_url:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        project.pdf_file_url,
        media_type="application/pdf",
        filename=f"{project.name}.pdf"
    )


@router.get("/pages/{page_id}/image")
def get_page_image(
    page_id: UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Serve a page image."""
    page = db.query(PDFPage).filter(PDFPage.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    
    if not page.image_url:
        raise HTTPException(status_code=404, detail="Page image not found")
    
    return FileResponse(
        page.image_url,
        media_type="image/png",
        filename=f"page_{page.page_number}.png"
    )


@router.get("/legend-tables/{legend_table_id}/image")
def get_legend_table_image(
    legend_table_id: UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Serve a legend table cropped image."""
    legend_table = db.query(LegendTable).filter(LegendTable.id == legend_table_id).first()
    if not legend_table:
        raise HTTPException(status_code=404, detail="Legend table not found")
    
    if not legend_table.cropped_image_url:
        raise HTTPException(status_code=404, detail="Legend table image not found")
    
    return FileResponse(
        legend_table.cropped_image_url,
        media_type="image/png",
        filename=f"legend_table_{legend_table_id}.png"
    )


@router.get("/legend-items/{legend_item_id}/icon-template")
def get_icon_template(
    legend_item_id: UUID,
    preprocessed: bool = False,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Serve an icon template image."""
    legend_item = db.query(LegendItem).filter(LegendItem.id == legend_item_id).first()
    if not legend_item:
        raise HTTPException(status_code=404, detail="Legend item not found")
    
    if not legend_item.icon_template:
        raise HTTPException(status_code=404, detail="Icon template not found")
    
    if preprocessed:
        if not legend_item.icon_template.preprocessed_icon_url:
            raise HTTPException(status_code=404, detail="Preprocessed icon not found")
        file_url = legend_item.icon_template.preprocessed_icon_url
    else:
        if not legend_item.icon_template.cropped_icon_url:
            raise HTTPException(status_code=404, detail="Icon template not found")
        file_url = legend_item.icon_template.cropped_icon_url
    
    return FileResponse(
        file_url,
        media_type="image/png",
        filename=f"icon_template_{legend_item_id}.png"
    )


@router.get("/legend-items/{legend_item_id}/label-template")
def get_label_template(
    legend_item_id: UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Serve a label template image."""
    legend_item = db.query(LegendItem).filter(LegendItem.id == legend_item_id).first()
    if not legend_item:
        raise HTTPException(status_code=404, detail="Legend item not found")
    
    if not legend_item.label_template:
        raise HTTPException(status_code=404, detail="Label template not found")
    
    if not legend_item.label_template.cropped_label_url:
        raise HTTPException(status_code=404, detail="Label template not found")
    
    return FileResponse(
        legend_item.label_template.cropped_label_url,
        media_type="image/png",
        filename=f"label_template_{legend_item_id}.png"
    )

