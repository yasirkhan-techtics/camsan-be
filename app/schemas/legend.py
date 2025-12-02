from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel


class UpdateBBoxRequest(BaseModel):
    bbox_normalized: List[float]


class IconTemplateSimple(BaseModel):
    """Simplified icon template for embedding in legend item response"""
    id: UUID
    original_bbox: List[Any]
    cropped_icon_url: str
    preprocessed_icon_url: Optional[str] = None
    template_ready: bool = False

    class Config:
        from_attributes = True


class LabelTemplateSimple(BaseModel):
    """Simplified label template for embedding in legend item response"""
    id: UUID
    tag_name: Optional[str]
    original_bbox: Optional[List[Any]]
    cropped_label_url: str

    class Config:
        from_attributes = True


class LegendItemResponse(BaseModel):
    id: UUID
    legend_table_id: UUID
    description: str
    label_text: Optional[str]  # Kept for backward compatibility
    order_index: int
    icon_bbox_status: str
    icon_template: Optional[IconTemplateSimple] = None  # Icon template for this legend item
    label_templates: List[LabelTemplateSimple] = []  # Multiple tags per legend item
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LegendTableResponse(BaseModel):
    id: UUID
    project_id: UUID
    page_id: UUID
    bbox_normalized: List[float]
    bbox_absolute: List[int]
    cropped_image_url: str
    extraction_status: str
    created_at: datetime
    updated_at: datetime
    legend_items: List[LegendItemResponse] = []

    class Config:
        from_attributes = True


