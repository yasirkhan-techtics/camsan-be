from typing import List, Optional

from pydantic import BaseModel, Field


class LegendItemLLM(BaseModel):
    description: str = Field(..., description="Legend description text")
    label_text: Optional[str] = Field(
        None, description="Optional short label for the legend symbol"
    )
    has_label: bool = Field(
        default=False,
        description="Whether the legend row explicitly contains a textual label",
    )


class LegendExtractionLLMResponse(BaseModel):
    legend_items: List[LegendItemLLM]


class LegendBBoxItem(BaseModel):
    bbox_norm: List[float] = Field(
        ...,
        description="Normalized [ymin, xmin, ymax, xmax] within 0-1000 scale",
        min_length=4,
        max_length=4,
    )
    description: Optional[str] = Field(
        None,
        description="Brief description of this legend table (e.g., 'Electrical Symbols', 'Piping Legend')"
    )


class LegendBBoxLLMResponse(BaseModel):
    legend_tables: List[LegendBBoxItem] = Field(
        ...,
        description="List of all legend tables found on this page. If multiple tables exist, include all of them."
    )
    notes: Optional[str] = Field(
        None,
        description="Any additional notes about the detection"
    )


class LegendBBoxVerificationResponse(BaseModel):
    is_correct: bool = Field(
        ...,
        description="Whether ALL bounding boxes correctly capture ALL legend tables on the page"
    )
    feedback: str = Field(
        ...,
        description="Feedback on what's wrong if incorrect, or confirmation if correct"
    )
    suggested_legend_tables: Optional[List[LegendBBoxItem]] = Field(
        None,
        description="Corrected list of legend table bounding boxes if current ones are wrong/incomplete"
    )


class BatchVerificationItem(BaseModel):
    serial_number: str
    matches: bool


class BatchVerificationResponse(BaseModel):
    results: List[BatchVerificationItem]


# LLM Detection Verification Schemas
class DetectionVerificationItem(BaseModel):
    serial_number: int = Field(..., description="Serial number in the verification table")
    is_valid: bool = Field(..., description="Whether the detection is valid")


class DetectionVerificationResponse(BaseModel):
    results: List[DetectionVerificationItem] = Field(
        ..., description="List of verification results for each detection"
    )


# Tag Overlap Resolution Schemas
class TagOverlapClassification(BaseModel):
    sr_no: int = Field(..., description="Serial number of the overlapping pair")
    selected_tag: str = Field(..., description="The tag name that should be kept")
    confidence: str = Field(default="medium", description="Confidence level: high/medium/low")


class TagOverlapResolutionLLMResponse(BaseModel):
    classifications: List[TagOverlapClassification] = Field(
        ..., description="List of classifications for each overlapping pair"
    )


# LLM Matcher Schemas
class IconMatchResult(BaseModel):
    match_found: bool = Field(..., description="Whether a matching tag was found")
    matched_tag: Optional[str] = Field(None, description="Name of the matched tag")
    confidence: str = Field(default="low", description="Confidence level: high/medium/low")
    reasoning: str = Field(default="", description="Brief explanation for the match")


class TagMatchResult(BaseModel):
    match_found: bool = Field(..., description="Whether a matching icon was found")
    matched_icon_type: Optional[str] = Field(None, description="Type of the matched icon")
    confidence: str = Field(default="low", description="Confidence level: high/medium/low")
    reasoning: str = Field(default="", description="Brief explanation for the match")


