from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl

from schemas.legend import LegendTableResponse
from schemas.page import PageResponse


class ProjectListItem(BaseModel):
    id: UUID
    name: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    pdf_file_url: HttpUrl | str
    status: str
    current_step: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectDetailResponse(ProjectResponse):
    pages: List[PageResponse] = []
    legend_tables: List[LegendTableResponse] = []


class ProjectStateResponse(BaseModel):
    id: UUID
    status: str
    current_step: Optional[str]

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    projects: List[ProjectListItem]
    total: int


