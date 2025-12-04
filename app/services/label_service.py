import os
import tempfile
from typing import List
from uuid import UUID

import cv2
import numpy as np
from fastapi import Depends, HTTPException
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import DetectionSettings, LabelDetection, LabelTemplate
from models.legend import LegendItem, LegendTable
from models.page import PDFPage
from models.project import Project
from services.storage_service import StorageService, get_storage_service


class LabelService:
    """Handles label template detection via template matching."""

    def __init__(self, db: Session, storage_service: StorageService):
        self.db = db
        self.storage = storage_service

    def save_label_bbox(
        self, 
        legend_item_id: UUID, 
        bbox: dict, 
        tag_name: str = None,
        label_template_id: UUID = None,
    ) -> LabelTemplate:
        """
        Save a label template from user-drawn bounding box.
        
        Args:
            legend_item_id: ID of the legend item
            bbox: Bounding box dict with x, y, width, height
            tag_name: Optional tag name (e.g., "CF1", "CF2")
            label_template_id: Optional ID to update existing template
        
        Returns:
            Created or updated LabelTemplate
        """
        print(f"📋 [LABEL BBOX] Saving label bbox for legend item: {legend_item_id}")
        legend_item = (
            self.db.query(LegendItem)
            .options(
                selectinload(LegendItem.legend_table).selectinload(LegendTable.project)
            )
            .filter(LegendItem.id == legend_item_id)
            .first()
        )
        if not legend_item:
            raise HTTPException(status_code=404, detail="Legend item not found.")

        if not legend_item.legend_table:
            raise HTTPException(
                status_code=400, detail="Legend table image is not available."
            )

        project = legend_item.legend_table.project
        if not project:
            raise HTTPException(status_code=400, detail="Project not found.")

        x, y, width, height = (
            int(bbox["x"]),
            int(bbox["y"]),
            int(bbox["width"]),
            int(bbox["height"]),
        )

        # If updating existing template
        existing_template = None
        if label_template_id:
            existing_template = (
                self.db.query(LabelTemplate)
                .filter(
                    LabelTemplate.id == label_template_id,
                    LabelTemplate.legend_item_id == legend_item_id
                )
                .first()
            )
            if existing_template and existing_template.cropped_label_url:
                print(f"   🗑️ Deleting old label image: {existing_template.cropped_label_url}")
                self.storage.delete_file(existing_template.cropped_label_url)

        # Count existing templates to generate unique filename
        template_count = (
            self.db.query(LabelTemplate)
            .filter(LabelTemplate.legend_item_id == legend_item_id)
            .count()
        )

        print(f"   📐 Cropping label from bbox: x={x}, y={y}, w={width}, h={height}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = os.path.join(tmp_dir, "legend.png")
            self.storage.download_file(legend_item.legend_table.cropped_image_url, table_path)

            cropped_path = os.path.join(tmp_dir, "label.png")
            self._crop_image(table_path, cropped_path, (x, y, width, height))

            # Use tag_name in filename if provided, otherwise use index
            tag_suffix = tag_name if tag_name else f"tag_{template_count + 1}"
            upload_url = self.storage.upload_file(
                local_path=cropped_path,
                mime_type="image/png",
                filename=f"{legend_item_id}_label_{tag_suffix}.png",
                project_name=project.name,
                project_id=str(project.id),
            )
            print(f"   ✅ Label image uploaded")

        if existing_template:
            label_template = existing_template
        else:
            label_template = LabelTemplate(legend_item_id=legend_item_id)

        label_template.original_bbox = [x, y, width, height]
        label_template.cropped_label_url = upload_url
        # Auto-set tag_name from legend_item.label_text if not provided
        label_template.tag_name = tag_name or legend_item.label_text

        self.db.add(label_template)
        self.db.commit()
        self.db.refresh(label_template)
        print(f"   ✅ Label template saved! (tag_name={label_template.tag_name})")
        return label_template

    def delete_label_template(self, label_template_id: UUID) -> bool:
        """Delete a specific label template."""
        template = (
            self.db.query(LabelTemplate)
            .filter(LabelTemplate.id == label_template_id)
            .first()
        )
        if not template:
            raise HTTPException(status_code=404, detail="Label template not found.")

        # Delete associated detections first
        self.db.query(LabelDetection).filter(
            LabelDetection.label_template_id == label_template_id
        ).delete()

        # Delete the image file
        if template.cropped_label_url:
            self.storage.delete_file(template.cropped_label_url)

        self.db.delete(template)
        self.db.commit()
        return True

    def get_label_templates(self, legend_item_id: UUID) -> List[LabelTemplate]:
        """Get all label templates for a legend item."""
        return (
            self.db.query(LabelTemplate)
            .filter(LabelTemplate.legend_item_id == legend_item_id)
            .all()
        )

    def create_label_template(self, legend_item_id: UUID) -> LabelTemplate:
        legend_item = (
            self.db.query(LegendItem)
            .filter(LegendItem.id == legend_item_id)
            .first()
        )
        if not legend_item:
            raise HTTPException(status_code=404, detail="Legend item not found.")

        if not legend_item.label_text:
            raise HTTPException(
                status_code=400, detail="Legend item does not contain a label."
            )

        label_image = self._render_label(legend_item.label_text)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            label_image.save(tmp.name)
            upload_url = self.storage.upload_file(
                local_path=tmp.name,
                mime_type="image/png",
                filename=f"{legend_item_id}_label.png",
            )
        os.remove(tmp.name)

        label_template = (
            self.db.query(LabelTemplate)
            .filter(LabelTemplate.legend_item_id == legend_item_id)
            .first()
        )
        if not label_template:
            label_template = LabelTemplate(legend_item_id=legend_item_id)

        label_template.cropped_label_url = upload_url
        # Auto-set tag_name from legend_item.label_text
        label_template.tag_name = legend_item.label_text

        self.db.add(label_template)
        self.db.commit()
        self.db.refresh(label_template)
        return label_template

    def detect_labels(self, project: Project, legend_item_ids: List[UUID] = None) -> List[LabelDetection]:
        query = (
            self.db.query(LabelTemplate)
            .join(LegendItem)
            .join(LegendItem.legend_table)
            .filter(LegendItem.legend_table.has(project_id=project.id))
        )
        
        # Filter by specific legend items if provided
        if legend_item_ids:
            query = query.filter(LabelTemplate.legend_item_id.in_(legend_item_ids))
            print(f"   📌 Filtering to {len(legend_item_ids)} selected legend item(s)")
        
        templates = query.all()
        if not templates:
            raise HTTPException(status_code=400, detail="No label templates available.")

        pages = (
            self.db.query(PDFPage)
            .filter(PDFPage.project_id == project.id)
            .order_by(PDFPage.page_number)
            .all()
        )
        if not pages:
            raise HTTPException(status_code=400, detail="Project has no pages.")

        settings = (
            self.db.query(DetectionSettings)
            .filter(DetectionSettings.project_id == project.id)
            .first()
        )
        if not settings:
            settings = DetectionSettings(project_id=project.id)
            self.db.add(settings)
            self.db.commit()
            self.db.refresh(settings)

        # First, delete any icon_label_matches that reference these label detections
        from models.detection import IconLabelMatch
        
        # Build query for detections to delete - filter by legend_item_ids if provided
        detection_query = self.db.query(LabelDetection.id).filter(
            LabelDetection.project_id == project.id
        )
        if legend_item_ids:
            # Only delete detections for the specific legend items being re-processed
            template_ids = [t.id for t in templates]
            detection_query = detection_query.filter(
                LabelDetection.label_template_id.in_(template_ids)
            )
            print(f"   📌 Only clearing detections for {len(template_ids)} selected template(s)")
        
        label_detection_ids = [d.id for d in detection_query.all()]
        
        if label_detection_ids:
            self.db.query(IconLabelMatch).filter(
                IconLabelMatch.label_detection_id.in_(label_detection_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
        
            # Now delete the label detections (only the filtered ones)
            self.db.query(LabelDetection).filter(
                LabelDetection.id.in_(label_detection_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
            print(f"   ✅ Cleared {len(label_detection_ids)} old detection(s)")
        else:
            print(f"   ✅ No old detections to clear")

        scale_values = self._generate_scales(
            settings.label_scale_min, settings.label_scale_max, step=0.1
        )

        detections: List[LabelDetection] = []
        for template in templates:
            template_path = self._download_template(template)
            try:
                template_gray = cv2.cvtColor(
                    cv2.imread(template_path), cv2.COLOR_BGR2GRAY
                )
                if template_gray is None:
                    continue

                for page in pages:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        page_path = os.path.join(tmp_dir, f"{page.id}.png")
                        self.storage.download_file(page.image_url, page_path)
                        page_gray = cv2.cvtColor(
                            cv2.imread(page_path), cv2.COLOR_BGR2GRAY
                        )

                        matches = self._match_template(
                            page_gray,
                            template_gray,
                            scale_values,
                            settings.label_match_threshold,
                        )

                    for match in matches:
                        detection = LabelDetection(
                            project_id=project.id,
                            label_template_id=template.id,
                            page_id=page.id,
                            bbox=[match["x"], match["y"], match["w"], match["h"]],
                            center=[match["x"] + match["w"] / 2, match["y"] + match["h"] / 2],
                            confidence=match["score"],
                            scale=match["scale"],
                            rotation=0,
                        )
                        self.db.add(detection)
                        detections.append(detection)
            finally:
                if os.path.exists(template_path):
                    os.remove(template_path)

        self.db.commit()
        for det in detections:
            self.db.refresh(det)
        return detections

    @staticmethod
    def _crop_image(
        legend_path: str, output_path: str, bbox: tuple[int, int, int, int]
    ) -> None:
        """Crop an image using the provided bounding box."""
        image = Image.open(legend_path)
        x, y, width, height = bbox
        cropped = image.crop((x, y, x + width, y + height))
        cropped.save(output_path)
        # Close file handles to avoid Windows lock issues
        cropped.close()
        image.close()

    @staticmethod
    def _render_label(text: str) -> Image.Image:
        font_size = 72
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        dummy_img = Image.new("RGB", (10, 10), color="white")
        draw = ImageDraw.Draw(dummy_img)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        width = text_bbox[2] - text_bbox[0] + 40
        height = text_bbox[3] - text_bbox[1] + 40

        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), text, fill="black", font=font)
        return image

    @staticmethod
    def _generate_scales(min_scale: float, max_scale: float, step: float) -> List[float]:
        values = []
        current = min_scale
        while current <= max_scale + 1e-6:
            values.append(round(current, 2))
            current += step
        return values or [1.0]

    @staticmethod
    def _match_template(
        page_gray: np.ndarray,
        template_gray: np.ndarray,
        scales: List[float],
        threshold: float,
    ) -> List[dict]:
        matches = []
        for scale in scales:
            new_w = max(5, int(template_gray.shape[1] * scale))
            new_h = max(5, int(template_gray.shape[0] * scale))
            resized = cv2.resize(template_gray, (new_w, new_h))

            if new_w > page_gray.shape[1] or new_h > page_gray.shape[0]:
                continue

            result = cv2.matchTemplate(page_gray, resized, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                matches.append(
                    {
                        "x": int(pt[0]),
                        "y": int(pt[1]),
                        "w": new_w,
                        "h": new_h,
                        "score": float(result[pt[1], pt[0]]),
                        "scale": scale,
                    }
                )
        return matches

    def _download_template(self, template: LabelTemplate) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            self.storage.download_file(template.cropped_label_url, tmp.name)
            return tmp.name


def get_label_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
) -> LabelService:
    return LabelService(db=db, storage_service=storage_service)


