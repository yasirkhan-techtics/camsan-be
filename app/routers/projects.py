import os
import tempfile
from typing import Optional
from uuid import UUID

from fastapi import Depends, File, Form, HTTPException, UploadFile
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.project import Project
from models.legend import LegendTable, LegendItem
from models.detection import DetectionSettings, IconTemplate, LabelTemplate
from schemas.project import (
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectStateResponse,
)
from schemas.detection import (
    DetectionSettingsResponse,
    DetectionSettingsUpdateRequest,
)
from services.storage_service import StorageService, get_storage_service
from services.pdf_service import PDFService, get_pdf_service
from utils.state_manager import StateManager

router = APIRouter()


def _get_or_create_detection_settings(
    db: Session, project_id: UUID
) -> DetectionSettings:
    settings = (
        db.query(DetectionSettings)
        .filter(DetectionSettings.project_id == project_id)
        .first()
    )
    if settings:
        return settings

    settings = DetectionSettings(project_id=project_id)
    db.add(settings)
    db.commit()
    db.refresh(settings)
    return settings


@router.get("", response_model=ProjectListResponse)
def list_projects(db: Session = Depends(get_db)):
    """Get list of all projects with basic information."""
    from schemas.project import ProjectListItem
    
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return ProjectListResponse(
        projects=[ProjectListItem.model_validate(p) for p in projects],
        total=len(projects)
    )


@router.post("", response_model=ProjectResponse)
async def create_project(
    name: str = Form(...),
    pdf_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
    pdf_service: PDFService = Depends(get_pdf_service),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    print(f"📄 Creating project: {name}")
    
    # Create project first to get the ID
    project = Project(name=name, pdf_file_url="", status="uploaded")
    db.add(project)
    db.flush()  # Get the ID without committing

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await pdf_file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        # Upload PDF
        print(f"   📤 Uploading PDF...")
        uploaded_url = storage.upload_file(
            local_path=temp_path,
            mime_type="application/pdf",
            filename=pdf_file.filename,
            project_name=project.name,
            project_id=str(project.id),
        )
        project.pdf_file_url = uploaded_url
        
        # Process PDF to images immediately
        print(f"   🖼️ Processing PDF to images...")
        try:
            pages = pdf_service.convert_pdf_to_images(db, project)
            print(f"   ✅ Created {len(pages)} page images")
            
            project.status = "pages_extracted"
            project.current_step = "legend_detection"
        except Exception as e:
            print(f"   ❌ Error processing PDF: {e}")
            project.status = "error"
            project.error_message = f"Failed to process PDF: {str(e)}"
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process PDF. Please ensure Poppler is installed and POPPLER_PATH is set correctly in .env file. Error: {str(e)}"
            )
        
    finally:
        os.remove(temp_path)

    db.commit()
    db.refresh(project)
    
    print(f"   🎉 Project created successfully with {len(pages)} pages")
    return ProjectResponse.model_validate(project)


@router.get("/{project_id}", response_model=ProjectDetailResponse)
def get_project(project_id: UUID, db: Session = Depends(get_db)):
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .options(
            selectinload(Project.pages),
            selectinload(Project.legend_tables)
                .selectinload(LegendTable.legend_items)
                .selectinload(LegendItem.icon_template),
            selectinload(Project.legend_tables)
                .selectinload(LegendTable.legend_items)
                .selectinload(LegendItem.label_templates),
        )
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    # Debug: Print file paths
    print(f"📄 Project PDF path: {project.pdf_file_url}")
    if project.pages:
        print(f"📄 First page path: {project.pages[0].image_url}")
    
    return ProjectDetailResponse.model_validate(project)


@router.get("/{project_id}/state", response_model=ProjectStateResponse)
def get_project_state(project_id: UUID, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return ProjectStateResponse.model_validate(project)


@router.post("/{project_id}/resume", response_model=ProjectStateResponse)
def resume_project(
    project_id: UUID,
    step: Optional[str] = None,
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    state = StateManager(db)
    state.transition(project, project.status, step or project.current_step)
    return ProjectStateResponse.model_validate(project)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .options(
            selectinload(Project.pages),
            selectinload(Project.legend_tables).selectinload(LegendTable.legend_items),
        )
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    print(f"🗑️ Deleting project: {project.name} (ID: {project.id})")
    
    # First, try to delete project directory (for new structure)
    storage.delete_project_directory(project.name, str(project.id))
    
    # Also delete individual files (for old structure or files outside project dir)
    files_to_delete = []
    
    # Add main PDF file
    if project.pdf_file_url:
        files_to_delete.append(project.pdf_file_url)
    
    # Add page images
    for page in project.pages:
        if page.image_url:
            files_to_delete.append(page.image_url)
    
    # Add legend cropped images and templates
    for legend_table in project.legend_tables:
        if legend_table.cropped_image_url:
            files_to_delete.append(legend_table.cropped_image_url)
        
        # Add icon and label templates for each legend item
        for legend_item in legend_table.legend_items:
            if legend_item.icon_template:
                if legend_item.icon_template.cropped_icon_url:
                    files_to_delete.append(legend_item.icon_template.cropped_icon_url)
                if legend_item.icon_template.preprocessed_icon_url:
                    files_to_delete.append(legend_item.icon_template.preprocessed_icon_url)
            
            # Handle multiple label templates per legend item
            for label_template in legend_item.label_templates:
                if label_template.cropped_label_url:
                    files_to_delete.append(label_template.cropped_label_url)
    
    # Delete all individual files (handles old structure)
    print(f"   📋 Found {len(files_to_delete)} files to delete")
    deleted_count = 0
    for file_url in files_to_delete:
        try:
            storage.delete_file(file_url)
            deleted_count += 1
        except Exception as e:
            print(f"   ⚠️ Failed to delete file {file_url}: {e}")
    
    print(f"   ✅ Deleted {deleted_count}/{len(files_to_delete)} individual files")
    
    # Explicitly delete icon_label_matches first (to avoid FK constraint issues)
    from models.detection import IconLabelMatch, IconDetection, LabelDetection
    
    # Get all detection IDs for this project
    icon_detection_ids = [d.id for d in db.query(IconDetection.id).filter(IconDetection.project_id == project_id).all()]
    label_detection_ids = [d.id for d in db.query(LabelDetection.id).filter(LabelDetection.project_id == project_id).all()]
    
    # Delete matches that reference these detections
    if icon_detection_ids or label_detection_ids:
        matches_deleted = db.query(IconLabelMatch).filter(
            (IconLabelMatch.icon_detection_id.in_(icon_detection_ids)) |
            (IconLabelMatch.label_detection_id.in_(label_detection_ids))
        ).delete(synchronize_session=False)
        print(f"   🔗 Deleted {matches_deleted} icon-label matches")
    
    # Delete project from database (cascade will delete related records)
    db.delete(project)
    db.commit()


@router.get(
    "/{project_id}/detection-settings", response_model=DetectionSettingsResponse
)
def get_detection_settings(project_id: UUID, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    settings = _get_or_create_detection_settings(db, project_id)
    return DetectionSettingsResponse.model_validate(settings)


@router.put(
    "/{project_id}/detection-settings", response_model=DetectionSettingsResponse
)
def update_detection_settings(
    project_id: UUID,
    payload: DetectionSettingsUpdateRequest,
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    settings = _get_or_create_detection_settings(db, project_id)
    update_data = payload.model_dump(exclude_unset=True)

    if not update_data:
        return DetectionSettingsResponse.model_validate(settings)

    icon_min = update_data.get("icon_scale_min", settings.icon_scale_min)
    icon_max = update_data.get("icon_scale_max", settings.icon_scale_max)
    label_min = update_data.get("label_scale_min", settings.label_scale_min)
    label_max = update_data.get("label_scale_max", settings.label_scale_max)

    if icon_min is not None and icon_max is not None and icon_min > icon_max:
        raise HTTPException(
            status_code=400, detail="Icon scale min cannot be greater than max."
        )

    if label_min is not None and label_max is not None and label_min > label_max:
        raise HTTPException(
            status_code=400, detail="Label scale min cannot be greater than max."
        )

    for field, value in update_data.items():
        if value is None:
            continue
        if field.endswith("_threshold") and not (0 < value <= 1.0):
            raise HTTPException(
                status_code=400, detail=f"{field} must be between 0 and 1."
            )
        if field.endswith("_step") and value <= 0:
            raise HTTPException(
                status_code=400, detail=f"{field} must be greater than 0."
            )
        setattr(settings, field, value)

    db.add(settings)
    db.commit()
    db.refresh(settings)
    return DetectionSettingsResponse.model_validate(settings)


