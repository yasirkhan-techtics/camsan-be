from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class UpdateBBoxRequest(BaseModel):
    bbox_normalized: List[float]


class LegendItemResponse(BaseModel):
    id: UUID
    legend_table_id: UUID
    description: str
    label_text: Optional[str]
    order_index: int
    icon_bbox_status: str
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


