from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.detection import LabelDetection, LabelTemplate
from models.project import Project
from schemas.detection import (
    CreateLabelDetectionRequest,
    IconBBoxRequest,
    LabelDetectionResponse,
    LabelTemplateResponse,
    LLMVerificationRequest,
    LLMVerificationResponse,
    TagOverlapResolutionResponse,
    UpdateDetectionRequest,
)
from services.label_service import LabelService, get_label_service
from services.llm_verification_service import (
    LLMVerificationService,
    get_llm_verification_service,
)
from services.tag_overlap_service import TagOverlapService, get_tag_overlap_service
from utils.state_manager import StateManager
from sqlalchemy.orm import selectinload


def _detection_to_response(det: LabelDetection) -> LabelDetectionResponse:
    """Convert LabelDetection to response with tag_name from template."""
    # Get tag_name: first try template.tag_name, then fallback to legend_item.label_text
    tag_name = None
    if det.label_template:
        tag_name = det.label_template.tag_name
        # Fallback to legend_item.label_text if tag_name not set
        if not tag_name and det.label_template.legend_item:
            tag_name = det.label_template.legend_item.label_text
    
    data = {
        "id": det.id,
        "project_id": det.project_id,
        "label_template_id": det.label_template_id,
        "page_id": det.page_id,
        "bbox": det.bbox,
        "center": det.center,
        "confidence": det.confidence,
        "scale": det.scale,
        "rotation": det.rotation,
        "verification_status": det.verification_status,
        "tag_name": tag_name,
        "created_at": det.created_at,
        "updated_at": det.updated_at,
    }
    return LabelDetectionResponse(**data)

router = APIRouter()


class LabelBBoxRequest(BaseModel):
    """Request for creating/updating a label bbox with optional tag name."""
    x: int
    y: int
    width: int
    height: int
    tag_name: Optional[str] = None
    label_template_id: Optional[UUID] = None  # For updating existing template


@router.post(
    "/legend-items/{legend_item_id}/draw-label-bbox",
    response_model=dict,
)
def draw_label_bbox(
    legend_item_id: UUID,
    bbox: LabelBBoxRequest,
    label_service: LabelService = Depends(get_label_service),
):
    """
    Save a label template from user-drawn bounding box.
    
    This endpoint supports multiple tags per legend item.
    - To add a new tag: provide bbox and optional tag_name
    - To update existing tag: provide label_template_id along with bbox
    """
    print(f"📝 [LABEL BBOX] Received request to draw label bbox for item: {legend_item_id}")
    template = label_service.save_label_bbox(
        legend_item_id=legend_item_id,
        bbox=bbox.model_dump(),
        tag_name=bbox.tag_name,
        label_template_id=bbox.label_template_id,
    )
    return {
        "label_template_id": str(template.id),
        "url": template.cropped_label_url,
        "tag_name": template.tag_name,
    }


@router.get(
    "/legend-items/{legend_item_id}/label-templates",
    response_model=List[LabelTemplateResponse],
)
def get_label_templates(
    legend_item_id: UUID,
    label_service: LabelService = Depends(get_label_service),
):
    """Get all label templates for a legend item (supports multiple tags per icon)."""
    templates = label_service.get_label_templates(legend_item_id)
    return [LabelTemplateResponse.model_validate(t) for t in templates]


@router.get(
    "/legend-items/{legend_item_id}/label-template",
    response_model=dict,
)
def get_label_template(
    legend_item_id: UUID,
    db: Session = Depends(get_db),
):
    """Get the first label template for a legend item (backward compatibility)."""
    template = (
        db.query(LabelTemplate)
        .filter(LabelTemplate.legend_item_id == legend_item_id)
        .first()
    )
    if not template:
        raise HTTPException(status_code=404, detail="Label template not found.")
    return {
        "label_template_id": str(template.id),
        "url": template.cropped_label_url,
        "tag_name": template.tag_name,
    }


@router.delete(
    "/label-templates/{label_template_id}",
    status_code=204,
)
def delete_label_template(
    label_template_id: UUID,
    label_service: LabelService = Depends(get_label_service),
):
    """Delete a specific label template."""
    label_service.delete_label_template(label_template_id)
    return None


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
    
    # Re-fetch with template and legend_item relationships loaded
    detections = (
        db.query(LabelDetection)
        .filter(LabelDetection.project_id == project_id)
        .options(selectinload(LabelDetection.label_template).selectinload(LabelTemplate.legend_item))
        .order_by(LabelDetection.created_at)
        .all()
    )
    return [_detection_to_response(det) for det in detections]


@router.get(
    "/projects/{project_id}/label-detections",
    response_model=List[LabelDetectionResponse],
)
def list_label_detections(project_id: UUID, db: Session = Depends(get_db)):
    detections = (
        db.query(LabelDetection)
        .filter(LabelDetection.project_id == project_id)
        .options(selectinload(LabelDetection.label_template).selectinload(LabelTemplate.legend_item))
        .order_by(LabelDetection.created_at)
        .all()
    )
    return [_detection_to_response(det) for det in detections]


@router.get("/detections", response_model=List[LabelDetectionResponse])
def get_label_detections(
    legend_item_id: UUID = None,
    project_id: UUID = None,
    db: Session = Depends(get_db),
):
    """Get label detections filtered by legend_item_id or project_id."""
    query = db.query(LabelDetection).options(
        selectinload(LabelDetection.label_template).selectinload(LabelTemplate.legend_item)
    )
    
    if legend_item_id:
        query = query.join(LabelTemplate).filter(LabelTemplate.legend_item_id == legend_item_id)
    elif project_id:
        query = query.filter(LabelDetection.project_id == project_id)
    
    detections = query.all()
    return [_detection_to_response(d) for d in detections]


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


@router.post(
    "/projects/{project_id}/verify-label-detections",
    response_model=LLMVerificationResponse,
)
def verify_label_detections(
    project_id: UUID,
    payload: LLMVerificationRequest = None,
    db: Session = Depends(get_db),
    verification_service: LLMVerificationService = Depends(get_llm_verification_service),
):
    """
    Verify label/tag detections using LLM.
    
    This endpoint:
    1. Calculates dynamic confidence thresholds per tag type
    2. Auto-approves high-confidence detections
    3. Sends low-confidence detections to LLM for verification in batches
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    batch_size = payload.batch_size if payload else 10
    
    print(f"🔍 [LLM VERIFY] Starting label verification for project: {project_id}")
    result = verification_service.verify_label_detections(project, batch_size=batch_size)
    print(f"✅ [LLM VERIFY] Verification complete!")
    
    return LLMVerificationResponse(
        total_detections=result["total_detections"],
        auto_approved=result["auto_approved"],
        llm_approved=result["llm_approved"],
        llm_rejected=result["llm_rejected"],
        threshold_used=result["threshold_used"],
    )


@router.post(
    "/projects/{project_id}/resolve-tag-overlaps",
    response_model=TagOverlapResolutionResponse,
)
def resolve_tag_overlaps(
    project_id: UUID,
    db: Session = Depends(get_db),
    overlap_service: TagOverlapService = Depends(get_tag_overlap_service),
):
    """
    Resolve overlapping tag bounding boxes using LLM.
    
    This endpoint detects tags with >= 90% overlap and uses LLM to determine
    which tag to keep. The duplicate tags are marked as rejected.
    
    Note: This is automatically called during matching, but can be called
    separately if needed.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    print(f"🔍 [OVERLAP] Starting tag overlap resolution for project: {project_id}")
    result = overlap_service.resolve_overlaps(project)
    print(f"✅ [OVERLAP] Overlap resolution complete!")
    
    return TagOverlapResolutionResponse(
        total_tags=result["total_tags"],
        overlapping_clusters_found=result["overlapping_clusters_found"],
        tags_removed=result["tags_removed"],
        tags_kept=result["tags_kept"],
    )


