from sqlalchemy import Boolean, Column, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from models.base import BaseModel


class PDFPage(BaseModel):
    __tablename__ = "pdf_pages"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_url = Column(Text, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    processed = Column(Boolean, default=False, nullable=False)

    project = relationship("Project", back_populates="pages")
    legend_tables = relationship(
        "LegendTable", back_populates="page", cascade="all, delete"
    )


