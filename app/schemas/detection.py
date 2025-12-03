from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IconTemplateResponse(BaseModel):
    id: UUID
    legend_item_id: UUID
    original_bbox: list[Any]
    cropped_icon_url: str
    preprocessed_icon_url: Optional[str]
    template_ready: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LabelTemplateResponse(BaseModel):
    id: UUID
    legend_item_id: UUID
    tag_name: Optional[str]
    original_bbox: Optional[list[Any]]
    cropped_label_url: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DetectionSettingsResponse(BaseModel):
    id: UUID
    project_id: UUID
    icon_scale_min: float
    icon_scale_max: float
    icon_rotation_step: int
    label_scale_min: float
    label_scale_max: float
    label_rotation_step: int
    icon_match_threshold: float
    label_match_threshold: float
    nms_threshold: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DetectionSettingsUpdateRequest(BaseModel):
    icon_scale_min: float | None = None
    icon_scale_max: float | None = None
    icon_rotation_step: int | None = None
    label_scale_min: float | None = None
    label_scale_max: float | None = None
    label_rotation_step: int | None = None
    icon_match_threshold: float | None = None
    label_match_threshold: float | None = None
    nms_threshold: float | None = None


class IconDetectionResponse(BaseModel):
    id: UUID
    project_id: UUID
    icon_template_id: UUID
    page_id: UUID
    bbox: list[float]
    center: list[float]
    confidence: float
    scale: float
    rotation: int
    verification_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LabelDetectionResponse(BaseModel):
    id: UUID
    project_id: UUID
    label_template_id: UUID
    page_id: UUID
    bbox: list[float]
    center: list[float]
    confidence: float
    scale: float
    rotation: int
    verification_status: str
    rejection_source: Optional[str] = None  # "overlap_removal" or "llm_verification"
    tag_name: Optional[str] = None  # Tag text (e.g., "CL1", "CF2") for display
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IconLabelMatchResponse(BaseModel):
    id: UUID
    icon_detection_id: UUID
    label_detection_id: Optional[UUID]
    distance: float
    match_confidence: float
    match_method: str
    match_status: str
    llm_assigned_label: Optional[str] = None  # LLM-assigned label (when no physical tag detection)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IconBBoxRequest(BaseModel):
    x: int
    y: int
    width: int
    height: int


class BatchVerificationRequest(BaseModel):
    detection_ids: list[UUID]


class IconTemplateBatchItem(BaseModel):
    legend_item_id: UUID
    bbox: IconBBoxRequest


class IconTemplateBatchRequest(BaseModel):
    templates: list[IconTemplateBatchItem]


class CreateIconDetectionRequest(BaseModel):
    project_id: UUID
    icon_template_id: UUID
    page_id: UUID
    bbox_normalized: list[float]
    confidence: Optional[float] = 1.0
    scale: Optional[float] = 1.0
    rotation: Optional[int] = 0


class UpdateDetectionRequest(BaseModel):
    bbox_normalized: Optional[list[float]] = None
    confidence: Optional[float] = None
    verification_status: Optional[str] = None


class CreateLabelDetectionRequest(BaseModel):
    project_id: UUID
    label_template_id: UUID
    page_id: UUID
    bbox_normalized: list[float]
    confidence: Optional[float] = 1.0
    scale: Optional[float] = 1.0
    rotation: Optional[int] = 0


# LLM Verification Schemas
class LLMVerificationRequest(BaseModel):
    batch_size: int = Field(default=10, ge=1, le=20, description="Number of detections per LLM batch")


class LLMVerificationResponse(BaseModel):
    total_detections: int
    auto_approved: int = Field(description="High confidence detections auto-approved")
    llm_approved: int = Field(description="Low confidence detections approved by LLM")
    llm_rejected: int = Field(description="Low confidence detections rejected by LLM")
    threshold_used: dict = Field(description="Thresholds calculated for each tag/icon type")


class TagOverlapResolutionResponse(BaseModel):
    total_tags: int
    overlapping_clusters_found: int
    tags_removed: int
    tags_kept: int


# LLM Matcher Schemas
class LLMMatcherRequest(BaseModel):
    save_crops: bool = Field(default=False, description="Save crop images for debugging")


class LLMMatcherResponse(BaseModel):
    """Combined response for backward compatibility (both phases)"""
    total_unmatched_icons: int
    total_unassigned_tags: int
    icons_matched: int
    tags_matched: int
    api_calls_made: int


# Phase 5: Tag matching for unlabeled icons
class TagMatchingForIconsResponse(BaseModel):
    """Response for Phase 5: Tag matching for unlabeled icons"""
    total_unmatched_icons: int = Field(description="Total icons without assigned labels")
    icons_matched: int = Field(description="Icons successfully matched to tags by LLM")
    icons_rejected: int = Field(description="Icons rejected due to template verification failure")
    api_calls_made: int = Field(description="Number of LLM API calls made")


# Phase 6: Icon detection for unlabeled tags
class IconMatchingForTagsResponse(BaseModel):
    """Response for Phase 6: Icon detection for unlabeled tags using LLM + Template Matching"""
    total_unassigned_tags: int = Field(description="Total tags without assigned icons")
    tags_verified_incorrect: int = Field(default=0, description="Tags with incorrect text (skipped)")
    icons_detected_by_llm: int = Field(default=0, description="Icons detected by LLM verification")
    icons_not_found: int = Field(default=0, description="Icons not found by LLM")
    template_match_success: int = Field(default=0, description="Successful template matches for bbox")
    template_match_failed: int = Field(default=0, description="Failed template matches")
    tags_matched: int = Field(description="Tags successfully matched (new icons created)")
    api_calls_made: int = Field(description="Number of LLM API calls made")


# LLM Verification Item Response Schemas
class VerificationItemResult(BaseModel):
    serial_number: int
    detection_id: UUID
    is_valid: bool
    confidence: Optional[str] = None
    reasoning: Optional[str] = None


class LLMVerificationBatchResponse(BaseModel):
    results: List[VerificationItemResult]


