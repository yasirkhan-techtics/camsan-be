from sqlalchemy import Boolean, Column, Enum, Float, ForeignKey, Integer, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from models.base import BaseModel

VERIFICATION_STATUSES = ("pending", "verified", "rejected")
MATCH_METHODS = ("distance", "llm_matched")
MATCH_STATUSES = ("matched", "unmatched_icon", "unassigned_tag")


class IconTemplate(BaseModel):
    __tablename__ = "icon_templates"

    legend_item_id = Column(
        UUID(as_uuid=True), ForeignKey("legend_items.id"), nullable=False, unique=True
    )
    original_bbox = Column(JSON, nullable=False)  # [x, y, w, h]
    cropped_icon_url = Column(Text, nullable=False)
    preprocessed_icon_url = Column(Text, nullable=True)
    template_ready = Column(Boolean, default=False, nullable=False)

    legend_item = relationship("LegendItem", back_populates="icon_template")
    detections = relationship(
        "IconDetection", back_populates="icon_template", cascade="all, delete"
    )


class LabelTemplate(BaseModel):
    __tablename__ = "label_templates"

    legend_item_id = Column(
        UUID(as_uuid=True), ForeignKey("legend_items.id"), nullable=False
    )
    # Tag name/text for this label template (e.g., "CF1", "CF2")
    tag_name = Column(Text, nullable=True)
    original_bbox = Column(JSON, nullable=True)  # [x, y, w, h] - bbox in legend table
    cropped_label_url = Column(Text, nullable=False)

    legend_item = relationship("LegendItem", back_populates="label_templates")
    detections = relationship(
        "LabelDetection", back_populates="label_template", cascade="all, delete"
    )


class DetectionSettings(BaseModel):
    __tablename__ = "detection_settings"

    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, unique=True
    )
    icon_scale_min = Column(Float, default=0.8, nullable=False)
    icon_scale_max = Column(Float, default=1.2, nullable=False)
    icon_rotation_step = Column(Integer, default=15, nullable=False)
    label_scale_min = Column(Float, default=0.6, nullable=False)
    label_scale_max = Column(Float, default=1.3, nullable=False)
    label_rotation_step = Column(Integer, default=15, nullable=False)
    icon_match_threshold = Column(Float, default=0.7, nullable=False)
    label_match_threshold = Column(Float, default=0.7, nullable=False)
    nms_threshold = Column(Float, default=0.3, nullable=False)

    project = relationship("Project", back_populates="detection_settings")


class IconDetection(BaseModel):
    __tablename__ = "icon_detections"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    icon_template_id = Column(
        UUID(as_uuid=True), ForeignKey("icon_templates.id"), nullable=False
    )
    page_id = Column(UUID(as_uuid=True), ForeignKey("pdf_pages.id"), nullable=False)
    bbox = Column(JSON, nullable=False)  # [x, y, w, h]
    center = Column(JSON, nullable=False)  # [x, y]
    confidence = Column(Float, nullable=False)
    scale = Column(Float, nullable=False)
    rotation = Column(Integer, nullable=False)
    verification_status = Column(
        Enum(*VERIFICATION_STATUSES, name="verification_status"),
        default="pending",
        nullable=False,
    )

    icon_template = relationship("IconTemplate", back_populates="detections")
    page = relationship("PDFPage")


class LabelDetection(BaseModel):
    __tablename__ = "label_detections"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    label_template_id = Column(
        UUID(as_uuid=True), ForeignKey("label_templates.id"), nullable=False
    )
    page_id = Column(UUID(as_uuid=True), ForeignKey("pdf_pages.id"), nullable=False)
    bbox = Column(JSON, nullable=False)
    center = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    scale = Column(Float, nullable=False)
    rotation = Column(Integer, nullable=False)
    verification_status = Column(
        Enum(*VERIFICATION_STATUSES, name="label_verification_status"),
        default="pending",
        nullable=False,
    )

    label_template = relationship("LabelTemplate", back_populates="detections")
    page = relationship("PDFPage")


class IconLabelMatch(BaseModel):
    __tablename__ = "icon_label_matches"

    icon_detection_id = Column(
        UUID(as_uuid=True), ForeignKey("icon_detections.id"), nullable=False
    )
    label_detection_id = Column(
        UUID(as_uuid=True), ForeignKey("label_detections.id"), nullable=True
    )
    distance = Column(Float, nullable=False)
    match_confidence = Column(Float, nullable=False)
    match_method = Column(
        Enum(*MATCH_METHODS, name="match_method"),
        default="distance",
        nullable=False,
    )
    match_status = Column(
        Enum(*MATCH_STATUSES, name="match_status"),
        default="matched",
        nullable=False,
    )
    # LLM-assigned label text (used when LLM directly reads the label from image)
    # This allows matching without requiring a physical LabelDetection record
    llm_assigned_label = Column(Text, nullable=True)

    icon_detection = relationship("IconDetection", lazy="joined")
    label_detection = relationship("LabelDetection", lazy="joined")


