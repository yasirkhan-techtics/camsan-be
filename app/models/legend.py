from sqlalchemy import Column, Enum, ForeignKey, Integer, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from models.base import BaseModel

LEGEND_EXTRACTION_STATUSES = ("detected", "legends_extracted", "completed")
ICON_BBOX_STATUSES = ("pending", "drawn", "saved")


class LegendTable(BaseModel):
    __tablename__ = "legend_tables"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    page_id = Column(UUID(as_uuid=True), ForeignKey("pdf_pages.id"), nullable=False)
    bbox_normalized = Column(JSON, nullable=False)  # [ymin, xmin, ymax, xmax]
    bbox_absolute = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    cropped_image_url = Column(Text, nullable=False)
    extraction_status = Column(
        Enum(*LEGEND_EXTRACTION_STATUSES, name="legend_extraction_status"),
        default="detected",
        nullable=False,
    )

    project = relationship("Project", back_populates="legend_tables")
    page = relationship("PDFPage", back_populates="legend_tables")
    legend_items = relationship(
        "LegendItem", back_populates="legend_table", cascade="all, delete"
    )


class LegendItem(BaseModel):
    __tablename__ = "legend_items"

    legend_table_id = Column(
        UUID(as_uuid=True), ForeignKey("legend_tables.id"), nullable=False
    )
    description = Column(Text, nullable=False)
    label_text = Column(Text, nullable=True)
    order_index = Column(Integer, nullable=False)
    icon_bbox_status = Column(
        Enum(*ICON_BBOX_STATUSES, name="icon_bbox_status"),
        default="pending",
        nullable=False,
    )

    legend_table = relationship("LegendTable", back_populates="legend_items")
    icon_template = relationship(
        "IconTemplate", uselist=False, back_populates="legend_item", cascade="all, delete"
    )
    # One legend item can have multiple label templates (e.g., CF1, CF2, CF3 for same icon)
    label_templates = relationship(
        "LabelTemplate", back_populates="legend_item", cascade="all, delete"
    )


