from typing import List
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session

from database import get_db
from models.detection import IconDetection, IconLabelMatch, IconTemplate, LabelDetection
from models.project import Project, PROJECT_STATUSES
from schemas.detection import (
    BatchVerificationRequest,
    CreateIconDetectionRequest,
    IconBBoxRequest,
    IconDetectionResponse,
    IconLabelMatchResponse,
    IconMatchingForTagsResponse,
    IconTemplateBatchRequest,
    IconTemplateResponse,
    LLMMatcherRequest,
    LLMMatcherResponse,
    LLMVerificationRequest,
    LLMVerificationResponse,
    TagMatchingForIconsResponse,
    UpdateDetectionRequest,
)
from services.batch_service import BatchService, get_batch_service
from services.cascade_reset_service import cascade_reset_from_stage
from services.icon_service import IconService, get_icon_service
from services.llm_matcher_service import LLMMatcherService, get_llm_matcher_service
from services.llm_verification_service import (
    LLMVerificationService,
    get_llm_verification_service,
)
from services.matching_service import MatchingService, get_matching_service
from utils.state_manager import StateManager

router = APIRouter()


@router.post(
    "/legend-items/{legend_item_id}/draw-icon-bbox", response_model=IconTemplateResponse
)
def draw_icon_bbox(
    legend_item_id: UUID,
    bbox: IconBBoxRequest,
    icon_service: IconService = Depends(get_icon_service),
):
    print(f"📦 [ICON BBOX] Saving icon bbox for legend item: {legend_item_id}")
    print(f"📐 [ICON BBOX] Bbox: {bbox.model_dump()}")
    template = icon_service.save_icon_bbox(legend_item_id, bbox.model_dump())
    print(f"✅ [ICON BBOX] Icon template saved successfully!")
    return IconTemplateResponse.model_validate(template)


@router.post(
    "/legend-items/{legend_item_id}/preprocess-icon",
    response_model=IconTemplateResponse,
)
def preprocess_icon_template(
    legend_item_id: UUID,
    icon_service: IconService = Depends(get_icon_service),
):
    print(f"🔧 [PREPROCESS] Starting icon preprocessing for legend item: {legend_item_id}")
    template = icon_service.preprocess_icon(legend_item_id)
    print(f"✅ [PREPROCESS] Icon preprocessing complete!")
    return IconTemplateResponse.model_validate(template)


@router.post(
    "/icon-templates/batch-save",
    response_model=List[IconTemplateResponse],
)
def batch_save_icon_templates(
    payload: IconTemplateBatchRequest,
    icon_service: IconService = Depends(get_icon_service),
):
    templates = [
        icon_service.save_icon_bbox(item.legend_item_id, item.bbox.model_dump())
        for item in payload.templates
    ]
    return [IconTemplateResponse.model_validate(template) for template in templates]


@router.get(
    "/legend-items/{legend_item_id}/icon-template", response_model=IconTemplateResponse
)
def get_icon_template(
    legend_item_id: UUID,
    db: Session = Depends(get_db),
):
    template = (
        db.query(IconTemplate)
        .filter(IconTemplate.legend_item_id == legend_item_id)
        .first()
    )
    if not template:
        raise HTTPException(status_code=404, detail="Icon template not found.")
    return IconTemplateResponse.model_validate(template)


@router.post(
    "/projects/{project_id}/detect-icons", response_model=List[IconDetectionResponse]
)
def detect_icons(
    project_id: UUID,
    db: Session = Depends(get_db),
    icon_service: IconService = Depends(get_icon_service),
):
    print(f"🔍 [ICON DETECTION] Starting icon detection for project: {project_id}")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        print(f"❌ [ICON DETECTION] Project not found: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found.")

    # Cascade reset: Clear only icon detections (not labels)
    cascade_reset_from_stage(db, project_id, stage=0, detection_type="icons")

    print(f"📊 [ICON DETECTION] Project found, running detection service...")
    detections = icon_service.detect_icons(project)
    print(f"✅ [ICON DETECTION] Detection complete! Found {len(detections)} icon detections")
    
    current_idx = PROJECT_STATUSES.index(project.status)
    target_idx = PROJECT_STATUSES.index("icons_extracted")
    if current_idx <= target_idx:
        StateManager(db).transition(project, "icons_extracted", "icon_detection_complete")
    return [IconDetectionResponse.model_validate(det) for det in detections]


@router.get(
    "/projects/{project_id}/icon-detections",
    response_model=List[IconDetectionResponse],
)
def list_icon_detections(project_id: UUID, db: Session = Depends(get_db)):
    results = (
        db.query(IconDetection)
        .filter(IconDetection.project_id == project_id)
        .order_by(IconDetection.created_at)
        .all()
    )
    return [IconDetectionResponse.model_validate(det) for det in results]


@router.post(
    "/projects/{project_id}/match-icons-labels",
    response_model=List[IconLabelMatchResponse],
)
def match_icons_and_labels(
    project_id: UUID,
    db: Session = Depends(get_db),
    matching_service: MatchingService = Depends(get_matching_service),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    # Cascade reset: Clear downstream data (Stage 3 - Basic Matching)
    cascade_reset_from_stage(db, project_id, stage=3)

    matches = matching_service.match_icons_to_labels(project)
    StateManager(db).transition(project, "completed", "matching_complete")
    return [IconLabelMatchResponse.model_validate(match) for match in matches]


@router.post(
    "/projects/{project_id}/icon-detections/batch-verify",
    response_model=List[IconDetectionResponse],
)
def batch_verify_detections(
    project_id: UUID,
    payload: BatchVerificationRequest,
    db: Session = Depends(get_db),
    batch_service: BatchService = Depends(get_batch_service),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    count = (
        db.query(IconDetection)
        .filter(
            IconDetection.project_id == project_id,
            IconDetection.id.in_(payload.detection_ids),
        )
        .count()
    )
    if count != len(payload.detection_ids):
        raise HTTPException(
            status_code=400, detail="One or more detection IDs do not belong to project."
        )

    detections = batch_service.verify_detections(payload.detection_ids)
    return [IconDetectionResponse.model_validate(det) for det in detections]


@router.get(
    "/projects/{project_id}/matched-results",
    response_model=List[IconLabelMatchResponse],
)
def list_matched_results(project_id: UUID, db: Session = Depends(get_db)):
    # Use outer joins to include all match types:
    # - matched: has both icon and label
    # - unmatched_icon: has icon, no label  
    # - unassigned_tag: has label, no icon
    from sqlalchemy import or_
    
    matches = (
        db.query(IconLabelMatch)
        .outerjoin(IconDetection, IconLabelMatch.icon_detection_id == IconDetection.id)
        .outerjoin(LabelDetection, IconLabelMatch.label_detection_id == LabelDetection.id)
        .filter(
            or_(
                IconDetection.project_id == project_id,
                LabelDetection.project_id == project_id,
            )
        )
        .all()
    )
    return [IconLabelMatchResponse.model_validate(match) for match in matches]


@router.get("/detections", response_model=List[IconDetectionResponse])
def get_icon_detections(
    legend_item_id: UUID = None,
    project_id: UUID = None,
    db: Session = Depends(get_db),
):
    """Get icon detections filtered by legend_item_id or project_id."""
    query = db.query(IconDetection)
    
    if legend_item_id:
        query = query.join(IconTemplate).filter(IconTemplate.legend_item_id == legend_item_id)
    elif project_id:
        query = query.filter(IconDetection.project_id == project_id)
    
    detections = query.all()
    return [IconDetectionResponse.model_validate(d) for d in detections]


@router.post("/detections", response_model=IconDetectionResponse, status_code=201)
def create_icon_detection(
    data: CreateIconDetectionRequest,
    db: Session = Depends(get_db),
):
    """Create a new icon detection manually."""
    # Calculate center from bbox
    bbox = data.bbox_normalized
    center_x = (bbox[1] + bbox[3]) / 2
    center_y = (bbox[0] + bbox[2]) / 2
    
    detection = IconDetection(
        project_id=data.project_id,
        icon_template_id=data.icon_template_id,
        page_id=data.page_id,
        bbox=bbox,
        center=[center_y, center_x],
        confidence=data.confidence,
        scale=data.scale,
        rotation=data.rotation,
        verification_status="manual"
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return IconDetectionResponse.model_validate(detection)


@router.put("/detections/{detection_id}", response_model=IconDetectionResponse)
def update_icon_detection(
    detection_id: UUID,
    data: UpdateDetectionRequest,
    db: Session = Depends(get_db),
):
    """Update an existing icon detection."""
    detection = db.query(IconDetection).filter(IconDetection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Icon detection not found.")
    
    if data.bbox_normalized is not None:
        detection.bbox = data.bbox_normalized
        # Recalculate center
        bbox = data.bbox_normalized
        center_x = (bbox[1] + bbox[3]) / 2
        center_y = (bbox[0] + bbox[2]) / 2
        detection.center = [center_y, center_x]
    
    if data.confidence is not None:
        detection.confidence = data.confidence
    
    if data.verification_status is not None:
        detection.verification_status = data.verification_status
    
    db.commit()
    db.refresh(detection)
    return IconDetectionResponse.model_validate(detection)


@router.delete("/detections/{detection_id}", status_code=204)
def delete_icon_detection(
    detection_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete an icon detection."""
    detection = db.query(IconDetection).filter(IconDetection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Icon detection not found.")
    
    db.delete(detection)
    db.commit()


@router.post(
    "/projects/{project_id}/verify-icon-detections",
    response_model=LLMVerificationResponse,
)
def verify_icon_detections(
    project_id: UUID,
    payload: LLMVerificationRequest = None,
    db: Session = Depends(get_db),
    verification_service: LLMVerificationService = Depends(get_llm_verification_service),
):
    """
    Verify icon detections using LLM.
    
    This endpoint:
    1. Calculates dynamic confidence thresholds per icon type
    2. Auto-approves high-confidence detections
    3. Sends low-confidence detections to LLM for verification in batches
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    # Cascade reset: Clear only icon verification (not labels)
    cascade_reset_from_stage(db, project_id, stage=2, detection_type="icons")
    
    batch_size = payload.batch_size if payload else 10
    
    print(f"🔍 [LLM VERIFY] Starting icon verification for project: {project_id}")
    result = verification_service.verify_icon_detections(project, batch_size=batch_size)
    print(f"✅ [LLM VERIFY] Verification complete!")
    
    return LLMVerificationResponse(
        total_detections=result["total_detections"],
        auto_approved=result["auto_approved"],
        llm_approved=result["llm_approved"],
        llm_rejected=result["llm_rejected"],
        threshold_used=result["threshold_used"],
    )


@router.post(
    "/projects/{project_id}/llm-match-unmatched",
    response_model=LLMMatcherResponse,
)
def llm_match_unmatched(
    project_id: UUID,
    payload: LLMMatcherRequest = None,
    db: Session = Depends(get_db),
    matcher_service: LLMMatcherService = Depends(get_llm_matcher_service),
):
    """
    Use LLM to match unmatched icons and unassigned tags (combined - backward compatibility).
    
    This endpoint runs both Phase 5 and Phase 6 together.
    For separate control, use:
    - /match-tags-for-icons (Phase 5)
    - /match-icons-for-tags (Phase 6)
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    save_crops = payload.save_crops if payload else False
    
    print(f"🔍 [LLM MATCH] Starting combined LLM matching for project: {project_id}")
    result = matcher_service.match_unmatched_items(project, save_crops=save_crops)
    print(f"✅ [LLM MATCH] LLM matching complete!")
    
    return LLMMatcherResponse(
        total_unmatched_icons=result["total_unmatched_icons"],
        total_unassigned_tags=result["total_unassigned_tags"],
        icons_matched=result["icons_matched"],
        tags_matched=result["tags_matched"],
        api_calls_made=result["api_calls_made"],
    )


@router.post(
    "/projects/{project_id}/match-tags-for-icons",
    response_model=TagMatchingForIconsResponse,
)
def match_tags_for_unlabeled_icons(
    project_id: UUID,
    payload: LLMMatcherRequest = None,
    db: Session = Depends(get_db),
    matcher_service: LLMMatcherService = Depends(get_llm_matcher_service),
):
    """
    PHASE 5: Use LLM to find matching tags for unlabeled icons.
    
    This endpoint processes icons that have no assigned label and uses LLM
    to find matching tags by analyzing the surrounding area.
    
    Should be called after:
    - Basic distance-based matching (match-icons-labels)
    
    Flow: Raw Detection → LLM Verification → Overlap Removal → Basic Matching → **This Step**
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    # Cascade reset: Clear downstream data (Stage 4 - Tag Matching)
    cascade_reset_from_stage(db, project_id, stage=4)
    
    save_crops = payload.save_crops if payload else False
    
    print(f"🔍 [PHASE 5] Starting tag matching for unlabeled icons: {project_id}")
    result = matcher_service.match_tags_for_unlabeled_icons(project, save_crops=save_crops)
    print(f"✅ [PHASE 5] Tag matching complete!")
    
    return TagMatchingForIconsResponse(
        total_unmatched_icons=result["total_unmatched_icons"],
        icons_matched=result["icons_matched"],
        icons_rejected=result["icons_rejected"],
        api_calls_made=result["api_calls_made"],
    )


@router.post(
    "/projects/{project_id}/match-icons-for-tags",
    response_model=IconMatchingForTagsResponse,
)
def match_icons_for_unlabeled_tags(
    project_id: UUID,
    payload: LLMMatcherRequest = None,
    db: Session = Depends(get_db),
    matcher_service: LLMMatcherService = Depends(get_llm_matcher_service),
):
    """
    PHASE 6: Detect icons for unlabeled tags using LLM + Template Matching.
    
    This endpoint processes tags that have no assigned icon and:
    1. Verifies tag text is correct using LLM
    2. Detects if icon is present near the tag using LLM
    3. Uses template matching to find precise icon bbox
    4. Creates NEW IconDetection records for found icons
    
    Should be called after:
    - Tag matching for icons (match-tags-for-icons)
    
    Flow: Raw Detection → LLM Verification → Overlap Removal → Basic Matching → Tag Matching → **This Step**
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    # Cascade reset: Clear downstream data (Stage 5 - Icon Matching)
    cascade_reset_from_stage(db, project_id, stage=5)
    
    save_crops = payload.save_crops if payload else False
    
    print(f"🔍 [PHASE 6] Starting icon detection for unlabeled tags: {project_id}")
    result = matcher_service.match_icons_for_unlabeled_tags(project, save_crops=save_crops)
    print(f"✅ [PHASE 6] Icon detection complete!")
    
    return IconMatchingForTagsResponse(
        total_unassigned_tags=result["total_unassigned_tags"],
        tags_verified_incorrect=result.get("tags_verified_incorrect", 0),
        icons_detected_by_llm=result.get("icons_detected_by_llm", 0),
        icons_not_found=result.get("icons_not_found", 0),
        template_match_success=result.get("template_match_success", 0),
        template_match_failed=result.get("template_match_failed", 0),
        tags_matched=result["tags_matched"],
        api_calls_made=result["api_calls_made"],
    )


