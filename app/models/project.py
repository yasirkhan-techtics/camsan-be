from sqlalchemy import Column, Enum, String, Text
from sqlalchemy.orm import relationship

from models.base import BaseModel

PROJECT_STATUSES = (
    "uploaded",
    "pages_extracted",
    "legends_detected",
    "icons_extracted",
    "labeling_in_progress",
    "completed",
    "error",
)


class Project(BaseModel):
    __tablename__ = "projects"

    name = Column(String(255), nullable=False)
    pdf_file_url = Column(Text, nullable=False)
    status = Column(
        Enum(*PROJECT_STATUSES, name="project_status"),
        default="uploaded",
        nullable=False,
    )
    current_step = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)

    pages = relationship("PDFPage", back_populates="project", cascade="all, delete")
    legend_tables = relationship(
        "LegendTable", back_populates="project", cascade="all, delete"
    )
    detection_settings = relationship(
        "DetectionSettings", uselist=False, back_populates="project", cascade="all, delete"
    )


