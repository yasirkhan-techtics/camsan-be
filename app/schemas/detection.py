from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


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


