from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PageResponse(BaseModel):
    id: UUID
    project_id: UUID
    page_number: int
    image_url: str
    width: int | None
    height: int | None
    processed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


