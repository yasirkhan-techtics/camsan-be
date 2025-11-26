import os
import tempfile
from typing import List

from fastapi import Depends, HTTPException
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import IconDetection, IconTemplate
from models.page import PDFPage
from services.llm_service import LLMService, get_llm_service
from services.storage_service import StorageService, get_storage_service

BATCH_VERIFICATION_PROMPT = (
    "Each row contains the reference icon on the left and the detected symbol on the right "
    "with a serial number. Compare every pair and respond with JSON listing serial_number "
    "and whether the detected symbol matches the reference shape (ignore scale, rotation, thickness). "
    "Do not add commentary."
)


class BatchService:
    """Handles batch table creation and LLM verification."""

    def __init__(
        self,
        db: Session,
        storage_service: StorageService,
        llm_service: LLMService,
    ):
        self.db = db
        self.storage = storage_service
        self.llm = llm_service

    def verify_detections(self, detection_ids: List[str]) -> List[IconDetection]:
        if not detection_ids:
            raise HTTPException(status_code=400, detail="No detection IDs provided.")

        detections = (
            self.db.query(IconDetection)
            .filter(IconDetection.id.in_(detection_ids))
            .options(
                selectinload(IconDetection.icon_template).selectinload(
                    IconTemplate.legend_item
                ),
                selectinload(IconDetection.page),
            )
            .all()
        )

        if not detections:
            raise HTTPException(status_code=404, detail="Detections not found.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = os.path.join(tmp_dir, "batch_table.png")
            rows = []
            for idx, detection in enumerate(detections, start=1):
                template_path = os.path.join(tmp_dir, f"{detection.id}_template.png")
                page_crop_path = os.path.join(tmp_dir, f"{detection.id}_crop.png")

                template_url = (
                    detection.icon_template.preprocessed_icon_url
                    or detection.icon_template.cropped_icon_url
                )
                self.storage.download_file(template_url, template_path)

                page_path = os.path.join(tmp_dir, f"{detection.page_id}.png")
                self.storage.download_file(detection.page.image_url, page_path)
                self._crop_detection(page_path, detection.bbox, page_crop_path)

                rows.append(
                    {
                        "serial": str(idx),
                        "template": template_path,
                        "detected": page_crop_path,
                    }
                )

            self._build_table_image(rows, table_path)
            response = self.llm.verify_batch(BATCH_VERIFICATION_PROMPT, table_path)

        detection_map = {str(idx + 1): det for idx, det in enumerate(detections)}
        for result in response.results:
            detection = detection_map.get(result.serial_number)
            if not detection:
                continue
            detection.verification_status = (
                "verified" if result.matches else "rejected"
            )
            self.db.add(detection)

        self.db.commit()
        for det in detections:
            self.db.refresh(det)
        return detections

    @staticmethod
    def _crop_detection(page_path: str, bbox: List[float], output_path: str) -> None:
        from PIL import Image

        image = Image.open(page_path)
        x, y, w, h = [int(v) for v in bbox]
        crop = image.crop((x, y, x + w, y + h))
        crop.save(output_path)
        # Close file handles to avoid Windows lock
        crop.close()
        image.close()

    @staticmethod
    def _build_table_image(rows: List[dict], output_path: str) -> None:
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            font = ImageFont.load_default()

        images = [
            (
                row["serial"],
                Image.open(row["template"]),
                Image.open(row["detected"]),
            )
            for row in rows
        ]

        col_serial_width = 120
        col_gap = 20
        max_template_width = max(img[1].width for img in images)
        max_detected_width = max(img[2].width for img in images)
        row_height = max(
            max(img[1].height, img[2].height) + 40 for img in images
        )
        table_width = (
            col_serial_width
            + max_template_width
            + max_detected_width
            + 4 * col_gap
        )
        table_height = len(images) * (row_height + col_gap) + col_gap

        table_image = Image.new("RGB", (table_width, table_height), color="white")
        draw = ImageDraw.Draw(table_image)

        y_offset = col_gap
        for serial, template_img, detected_img in images:
            draw.text(
                (col_gap, y_offset + row_height // 2 - 10),
                f"#{serial}",
                fill="black",
                font=font,
            )
            template_y = y_offset + (row_height - template_img.height) // 2
            detected_y = y_offset + (row_height - detected_img.height) // 2
            template_pos = (col_serial_width, template_y)
            detected_pos = (
                col_serial_width + max_template_width + col_gap,
                detected_y,
            )

            table_image.paste(template_img, template_pos)
            table_image.paste(detected_img, detected_pos)
            y_offset += row_height + col_gap

        table_image.save(output_path)
        
        # Close all file handles to avoid Windows lock
        table_image.close()
        for _, template_img, detected_img in images:
            template_img.close()
            detected_img.close()


def get_batch_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> BatchService:
    return BatchService(db=db, storage_service=storage_service, llm_service=llm_service)

