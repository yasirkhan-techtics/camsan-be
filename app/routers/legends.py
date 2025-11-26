import os
import tempfile
from typing import List
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter
from PIL import Image
from sqlalchemy.orm import Session

from database import get_db
from models.legend import LegendItem, LegendTable
from models.page import PDFPage
from models.project import Project
from schemas.legend import LegendItemResponse, LegendTableResponse, UpdateBBoxRequest
from schemas.page import PageResponse
from services.legend_service import LegendService, get_legend_service
from services.storage_service import StorageService, get_storage_service
from utils.state_manager import StateManager

router = APIRouter()


from pydantic import BaseModel

class CreateLegendTableRequest(BaseModel):
    bbox_normalized: List[float]
    page_number: int


@router.post("/{project_id}/legends", response_model=LegendTableResponse, status_code=201)
def create_legend_table(
    project_id: UUID,
    request: CreateLegendTableRequest,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Create a new legend table from user-drawn bounding box."""
    print(f"🎯 create_legend_table called for project: {project_id}")
    print(f"   Request data: bbox={request.bbox_normalized}, page={request.page_number}")
    
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        print(f"   ❌ Project not found: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found.")
    print(f"   ✅ Project found: {project.name}")
    
    # Find the page by page_number
    print(f"   🔍 Looking for page number: {request.page_number}")
    page = (
        db.query(PDFPage)
        .filter(PDFPage.project_id == project_id, PDFPage.page_number == request.page_number)
        .first()
    )
    if not page:
        print(f"   ❌ Page {request.page_number} not found for project {project_id}")
        raise HTTPException(status_code=404, detail=f"Page {request.page_number} not found.")
    print(f"   ✅ Page found: {page.id}")

    bbox_norm = request.bbox_normalized
    print(f"   📐 Calculating bbox from normalized: {bbox_norm}")
    
    # Calculate absolute bbox from normalized
    width = page.width or 0
    height = page.height or 0
    print(f"   📏 Page dimensions: {width} × {height}")
    ymin, xmin, ymax, xmax = bbox_norm
    y1 = int((ymin / 1000) * height)
    x1 = int((xmin / 1000) * width)
    y2 = int((ymax / 1000) * height)
    x2 = int((xmax / 1000) * width)
    bbox_abs = (x1, y1, x2, y2)
    print(f"   📐 Absolute bbox: {bbox_abs}")

    # Crop and save the legend table image
    print(f"   🖼️ Starting image processing...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download page image
            print(f"   📥 Downloading page image from: {page.image_url}")
            local_image = os.path.join(tmp_dir, f"{page.id}.png")
            storage.download_file(page.image_url, local_image)
            print(f"   ✅ Downloaded to: {local_image}")

            # Crop the image
            print(f"   ✂️ Cropping image...")
            full_image = Image.open(local_image)
            cropped_image = full_image.crop((x1, y1, x2, y2))

            # Save cropped image
            cropped_path = os.path.join(tmp_dir, "legend_cropped.png")
            cropped_image.save(cropped_path)
            print(f"   ✅ Cropped image saved")

            # Upload cropped image
            print(f"   📤 Uploading cropped image...")
            upload_url = storage.upload_file(
                local_path=cropped_path,
                mime_type="image/png",
                filename=f"{project_id}_{page.page_number}_legend_manual.png",
                project_name=project.name,
                project_id=str(project.id),
            )
            print(f"   ✅ Uploaded to: {upload_url}")

            # Create legend table record
            print(f"   💾 Creating legend table record...")
            legend_table = LegendTable(
                project_id=project_id,
                page_id=page.id,
                bbox_normalized=bbox_norm,
                bbox_absolute=bbox_abs,
                cropped_image_url=upload_url,
                extraction_status="detected",
            )
            db.add(legend_table)
            db.commit()
            db.refresh(legend_table)
            print(f"   ✅ Legend table created: {legend_table.id}")

        print(f"   🎉 Returning response...")
        return LegendTableResponse.model_validate(legend_table)
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@router.post("/{project_id}/detect-legends", response_model=List[LegendTableResponse])
def detect_legends(
    project_id: UUID,
    db: Session = Depends(get_db),
    legend_service: LegendService = Depends(get_legend_service),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    tables = legend_service.detect_legends(db, project)
    StateManager(db).transition(project, "legends_detected", "legend_detected")
    return [LegendTableResponse.model_validate(table) for table in tables]


@router.get(
    "/{project_id}/legends",
    response_model=List[LegendTableResponse],
)
def list_legends(project_id: UUID, db: Session = Depends(get_db)):
    tables = (
        db.query(LegendTable)
        .filter(LegendTable.project_id == project_id)
        .order_by(LegendTable.created_at)
        .all()
    )
    if not tables:
        project_exists = db.query(Project.id).filter(Project.id == project_id).first()
        if not project_exists:
            raise HTTPException(status_code=404, detail="Project not found.")
    return [LegendTableResponse.model_validate(table) for table in tables]


@router.post(
    "/{project_id}/legends/{legend_table_id}/extract-items",
    response_model=List[LegendItemResponse],
)
def extract_legend_items(
    project_id: UUID,
    legend_table_id: UUID,
    db: Session = Depends(get_db),
    legend_service: LegendService = Depends(get_legend_service),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    table = (
        db.query(LegendTable)
        .filter(
            LegendTable.id == legend_table_id, LegendTable.project_id == project_id
        )
        .first()
    )
    if not table:
        raise HTTPException(status_code=404, detail="Legend table not found.")

    items = legend_service.extract_legend_items(db, table)
    return [LegendItemResponse.model_validate(item) for item in items]


@router.get(
    "/{project_id}/legends/{legend_table_id}/items",
    response_model=List[LegendItemResponse],
)
def list_legend_items(
    project_id: UUID,
    legend_table_id: UUID,
    db: Session = Depends(get_db),
):
    items = (
        db.query(LegendItem)
        .filter(
            LegendItem.legend_table_id == legend_table_id,
            LegendItem.legend_table.has(project_id=project_id),
        )
        .order_by(LegendItem.order_index)
        .all()
    )
    return [LegendItemResponse.model_validate(item) for item in items]


@router.get(
    "/{project_id}/legend-items/{legend_item_id}/page",
    response_model=PageResponse,
)
def get_legend_item_page(
    project_id: UUID,
    legend_item_id: UUID,
    db: Session = Depends(get_db),
):
    legend_item = (
        db.query(LegendItem)
        .filter(LegendItem.id == legend_item_id)
        .join(LegendItem.legend_table)
        .filter(LegendItem.legend_table.has(project_id=project_id))
        .first()
    )
    if not legend_item:
        raise HTTPException(status_code=404, detail="Legend item not found.")

    page = legend_item.legend_table.page
    if not page:
        raise HTTPException(status_code=404, detail="Legend page not found.")
    return PageResponse.model_validate(page)


@router.put(
    "/{project_id}/legends/{legend_id}/bbox",
    response_model=LegendTableResponse,
)
def update_legend_bbox(
    project_id: UUID,
    legend_id: UUID,
    bbox_update: UpdateBBoxRequest,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Update the bounding box of a legend table and regenerate the cropped image."""
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    legend_table = (
        db.query(LegendTable)
        .filter(LegendTable.id == legend_id, LegendTable.project_id == project_id)
        .first()
    )
    if not legend_table:
        raise HTTPException(status_code=404, detail="Legend table not found.")

    page = legend_table.page
    if not page:
        raise HTTPException(status_code=404, detail="Page not found for legend table.")

    # Update normalized bbox
    bbox_norm = bbox_update.bbox_normalized
    legend_table.bbox_normalized = bbox_norm

    # Calculate absolute bbox from normalized
    width = page.width or 0
    height = page.height or 0
    ymin, xmin, ymax, xmax = bbox_norm
    y1 = int((ymin / 1000) * height)
    x1 = int((xmin / 1000) * width)
    y2 = int((ymax / 1000) * height)
    x2 = int((xmax / 1000) * width)
    legend_table.bbox_absolute = (x1, y1, x2, y2)

    # Regenerate cropped image
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download page image
        local_image = os.path.join(tmp_dir, f"{page.id}.png")
        storage.download_file(page.image_url, local_image)

        # Crop the image
        full_image = Image.open(local_image)
        cropped_image = full_image.crop((x1, y1, x2, y2))

        # Save cropped image
        cropped_path = os.path.join(tmp_dir, f"{legend_id}_cropped.png")
        cropped_image.save(cropped_path)

        # Delete old cropped image if exists
        if legend_table.cropped_image_url:
            try:
                storage.delete_file(legend_table.cropped_image_url)
            except Exception as e:
                print(f"Error deleting old cropped image: {e}")

        # Upload new cropped image
        upload_url = storage.upload_file(
            local_path=cropped_path,
            mime_type="image/png",
            filename=f"{project_id}_{legend_id}_legend_cropped.png",
            project_name=project.name,
            project_id=str(project.id),
        )
        legend_table.cropped_image_url = upload_url

    db.commit()
    db.refresh(legend_table)
    return LegendTableResponse.model_validate(legend_table)


@router.delete("/{project_id}/legends/{legend_id}", status_code=204)
def delete_legend_table(
    project_id: UUID,
    legend_id: UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Delete a legend table and its associated legend items."""
    legend_table = (
        db.query(LegendTable)
        .filter(LegendTable.id == legend_id, LegendTable.project_id == project_id)
        .first()
    )
    if not legend_table:
        raise HTTPException(status_code=404, detail="Legend table not found.")

    # Delete cropped image file
    if legend_table.cropped_image_url:
        try:
            storage.delete_file(legend_table.cropped_image_url)
        except Exception as e:
            print(f"Error deleting cropped image: {e}")

    # Delete associated legend items first (foreign key constraint)
    db.query(LegendItem).filter(LegendItem.legend_table_id == legend_id).delete()
    
    # Delete the legend table
    db.delete(legend_table)
    db.commit()


