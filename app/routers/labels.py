from typing import List
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session

from database import get_db
from models.detection import LabelDetection, LabelTemplate
from models.project import Project
from schemas.detection import (
    CreateLabelDetectionRequest,
    IconBBoxRequest,
    LabelDetectionResponse,
    UpdateDetectionRequest,
)
from services.label_service import LabelService, get_label_service
from utils.state_manager import StateManager

router = APIRouter()


@router.post(
    "/legend-items/{legend_item_id}/draw-label-bbox",
    response_model=dict,
)
def draw_label_bbox(
    legend_item_id: UUID,
    bbox: IconBBoxRequest,
    label_service: LabelService = Depends(get_label_service),
):
    """Save a label template from user-drawn bounding box."""
    print(f"📝 [LABEL BBOX] Received request to draw label bbox for item: {legend_item_id}")
    template = label_service.save_label_bbox(legend_item_id, bbox.model_dump())
    return {"label_template_id": str(template.id), "url": template.cropped_label_url}


@router.get(
    "/legend-items/{legend_item_id}/label-template",
    response_model=dict,
)
def get_label_template(
    legend_item_id: UUID,
    db: Session = Depends(get_db),
):
    """Get the label template for a legend item."""
    template = (
        db.query(LabelTemplate)
        .filter(LabelTemplate.legend_item_id == legend_item_id)
        .first()
    )
    if not template:
        raise HTTPException(status_code=404, detail="Label template not found.")
    return {"label_template_id": str(template.id), "url": template.cropped_label_url}


@router.post(
    "/legend-items/{legend_item_id}/label-template",
    response_model=dict,
)
def create_label_template(
    legend_item_id: UUID,
    label_service: LabelService = Depends(get_label_service),
):
    """Create a label template by rendering text (deprecated - use draw-label-bbox instead)."""
    template = label_service.create_label_template(legend_item_id)
    return {"label_template_id": str(template.id), "url": template.cropped_label_url}


@router.post(
    "/projects/{project_id}/detect-labels",
    response_model=List[LabelDetectionResponse],
)
def detect_labels(
    project_id: UUID,
    db: Session = Depends(get_db),
    label_service: LabelService = Depends(get_label_service),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        print(f"❌ [LABEL DETECTION] Project not found: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found.")
    
    print(f"🔍 [LABEL DETECTION] Starting label detection for project: {project_id}")
    detections = label_service.detect_labels(project)
    print(f"✅ [LABEL DETECTION] Detection complete! Found {len(detections)} label detections")
    
    # Only transition state if not already past this stage (allows re-detection)
    from models.project import PROJECT_STATUSES
    current_index = PROJECT_STATUSES.index(project.status)
    target_index = PROJECT_STATUSES.index("labeling_in_progress")
    if current_index <= target_index:
        StateManager(db).transition(project, "labeling_in_progress", "label_detection_complete")
    
    return [LabelDetectionResponse.model_validate(det) for det in detections]


@router.get(
    "/projects/{project_id}/label-detections",
    response_model=List[LabelDetectionResponse],
)
def list_label_detections(project_id: UUID, db: Session = Depends(get_db)):
    detections = (
        db.query(LabelDetection)
        .filter(LabelDetection.project_id == project_id)
        .order_by(LabelDetection.created_at)
        .all()
    )
    return [LabelDetectionResponse.model_validate(det) for det in detections]


@router.get("/detections", response_model=List[LabelDetectionResponse])
def get_label_detections(
    legend_item_id: UUID = None,
    project_id: UUID = None,
    db: Session = Depends(get_db),
):
    """Get label detections filtered by legend_item_id or project_id."""
    query = db.query(LabelDetection)
    
    if legend_item_id:
        query = query.join(LabelTemplate).filter(LabelTemplate.legend_item_id == legend_item_id)
    elif project_id:
        query = query.filter(LabelDetection.project_id == project_id)
    
    detections = query.all()
    return [LabelDetectionResponse.model_validate(d) for d in detections]


@router.post("/detections", response_model=LabelDetectionResponse, status_code=201)
def create_label_detection(
    data: CreateLabelDetectionRequest,
    db: Session = Depends(get_db),
):
    """Create a new label detection manually."""
    # Calculate center from bbox
    bbox = data.bbox_normalized
    center_x = (bbox[1] + bbox[3]) / 2
    center_y = (bbox[0] + bbox[2]) / 2
    
    detection = LabelDetection(
        project_id=data.project_id,
        label_template_id=data.label_template_id,
        page_id=data.page_id,
        bbox=bbox,
        center=[center_y, center_x],
        confidence=data.confidence,
        scale=data.scale,
        rotation=data.rotation
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return LabelDetectionResponse.model_validate(detection)


@router.put("/detections/{detection_id}", response_model=LabelDetectionResponse)
def update_label_detection(
    detection_id: UUID,
    data: UpdateDetectionRequest,
    db: Session = Depends(get_db),
):
    """Update an existing label detection."""
    detection = db.query(LabelDetection).filter(LabelDetection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Label detection not found.")
    
    if data.bbox_normalized is not None:
        detection.bbox = data.bbox_normalized
        # Recalculate center
        bbox = data.bbox_normalized
        center_x = (bbox[1] + bbox[3]) / 2
        center_y = (bbox[0] + bbox[2]) / 2
        detection.center = [center_y, center_x]
    
    if data.confidence is not None:
        detection.confidence = data.confidence
    
    db.commit()
    db.refresh(detection)
    return LabelDetectionResponse.model_validate(detection)


@router.delete("/detections/{detection_id}", status_code=204)
def delete_label_detection(
    detection_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete a label detection."""
    detection = db.query(LabelDetection).filter(LabelDetection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Label detection not found.")
    
    db.delete(detection)
    db.commit()


