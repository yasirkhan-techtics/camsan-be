import os
import tempfile
from typing import List
from uuid import UUID

import cv2
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import (
    DetectionSettings,
    IconDetection,
    IconTemplate,
)
from models.legend import LegendItem, LegendTable
from models.page import PDFPage
from models.project import Project
from services.preprocessing_service import (
    PreprocessingService,
    get_preprocessing_service,
)
from services.storage_service import StorageService, get_storage_service
from lib.electrical_symbol_detector import ElectricalSymbolDetector


class IconService:
    """Handles icon template creation and detection."""

    def __init__(
        self,
        db: Session,
        storage_service: StorageService,
        preprocessing_service: PreprocessingService,
    ):
        self.db = db
        self.storage = storage_service
        self.preprocessing = preprocessing_service

    def save_icon_bbox(self, legend_item_id: UUID, bbox: dict) -> IconTemplate:
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
        
        print(f"   📐 Bbox coordinates: x={x}, y={y}, width={width}, height={height}")

        # Check if icon template already exists and delete old image
        existing_template = (
            self.db.query(IconTemplate)
            .filter(IconTemplate.legend_item_id == legend_item_id)
            .first()
        )
        if existing_template and existing_template.cropped_icon_url:
            print(f"   🗑️ Deleting old icon template: {existing_template.cropped_icon_url}")
            self.storage.delete_file(existing_template.cropped_icon_url)
            if existing_template.preprocessed_icon_url:
                print(f"   🗑️ Deleting old preprocessed icon: {existing_template.preprocessed_icon_url}")
                self.storage.delete_file(existing_template.preprocessed_icon_url)

        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = os.path.join(tmp_dir, "legend.png")
            self.storage.download_file(legend_item.legend_table.cropped_image_url, table_path)
            
            # Check legend table image size
            from PIL import Image
            legend_img = Image.open(table_path)
            print(f"   📏 Legend table image size: {legend_img.size[0]} × {legend_img.size[1]} px")
            print(f"   ✂️ Cropping region: ({x}, {y}) to ({x+width}, {y+height})")
            
            # Validate bbox is within image bounds
            if x < 0 or y < 0 or x + width > legend_img.size[0] or y + height > legend_img.size[1]:
                print(f"   ⚠️ WARNING: Bbox is outside image bounds!")
                print(f"      Image bounds: (0, 0) to ({legend_img.size[0]}, {legend_img.size[1]})")
                print(f"      Requested bbox: ({x}, {y}) to ({x+width}, {y+height})")
            
            legend_img.close()  # Close file handle to avoid Windows lock

            cropped_path = os.path.join(tmp_dir, "icon.png")
            self._crop_image(table_path, cropped_path, (x, y, width, height))
            
            # Check cropped image size
            cropped_img = Image.open(cropped_path)
            print(f"   ✅ Cropped image size: {cropped_img.size[0]} × {cropped_img.size[1]} px")
            cropped_img.close()  # Close file handle to avoid Windows lock

            upload_url = self.storage.upload_file(
                local_path=cropped_path,
                mime_type="image/png",
                filename=f"{legend_item_id}_icon.png",
                project_name=project.name,
                project_id=str(project.id),
            )

        if existing_template:
            icon_template = existing_template
        else:
            icon_template = IconTemplate(legend_item_id=legend_item_id)

        icon_template.original_bbox = [x, y, width, height]
        icon_template.cropped_icon_url = upload_url
        icon_template.preprocessed_icon_url = None  # Clear preprocessed URL since we deleted the old file
        icon_template.template_ready = False

        legend_item.icon_bbox_status = "drawn"

        self.db.add(icon_template)
        self.db.add(legend_item)
        self.db.commit()
        self.db.refresh(icon_template)
        return icon_template

    def preprocess_icon(self, legend_item_id: UUID) -> IconTemplate:
        icon_template = (
            self.db.query(IconTemplate)
            .filter(IconTemplate.legend_item_id == legend_item_id)
            .join(LegendItem)
            .options(
                selectinload(IconTemplate.legend_item)
                .selectinload(LegendItem.legend_table)
                .selectinload(LegendTable.project)
            )
            .first()
        )
        if not icon_template:
            raise HTTPException(status_code=404, detail="Icon template not found.")

        project = icon_template.legend_item.legend_table.project
        if not project:
            raise HTTPException(status_code=400, detail="Project not found.")

        # Delete old preprocessed image if exists
        if icon_template.preprocessed_icon_url:
            print(f"   🗑️ Deleting old preprocessed icon: {icon_template.preprocessed_icon_url}")
            self.storage.delete_file(icon_template.preprocessed_icon_url)

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_icon = os.path.join(tmp_dir, "icon.png")
            self.storage.download_file(icon_template.cropped_icon_url, local_icon)

            processed_path = self.preprocessing.preprocess_icon(local_icon)
            upload_url = self.storage.upload_file(
                local_path=processed_path,
                mime_type="image/png",
                filename=f"{icon_template.id}_preprocessed.png",
                project_name=project.name,
                project_id=str(project.id),
            )

        icon_template.preprocessed_icon_url = upload_url
        icon_template.template_ready = True
        icon_template.legend_item.icon_bbox_status = "saved"

        self.db.add(icon_template)
        self.db.add(icon_template.legend_item)
        self.db.commit()
        self.db.refresh(icon_template)
        return icon_template

    def detect_icons(self, project: Project) -> List[IconDetection]:
        print(f"   📋 Loading icon templates for project...")
        templates = (
            self.db.query(IconTemplate)
            .join(LegendItem)
            .join(LegendItem.legend_table)
            .filter(LegendItem.legend_table.has(project_id=project.id))
            .all()
        )
        if not templates:
            print(f"   ❌ No icon templates found!")
            raise HTTPException(
                status_code=400, detail="No icon templates available for detection."
            )
        print(f"   ✅ Found {len(templates)} icon template(s)")

        print(f"   📄 Loading PDF pages...")
        pages = (
            self.db.query(PDFPage)
            .filter(PDFPage.project_id == project.id)
            .order_by(PDFPage.page_number)
            .all()
        )
        if not pages:
            print(f"   ❌ No pages found!")
            raise HTTPException(status_code=400, detail="Project has no pages.")
        print(f"   ✅ Found {len(pages)} page(s)")

        print(f"   ⚙️ Loading detection settings...")
        settings = (
            self.db.query(DetectionSettings)
            .filter(DetectionSettings.project_id == project.id)
            .first()
        )
        if not settings:
            print(f"   ⚙️ Creating default settings...")
            settings = DetectionSettings(project_id=project.id)
            self.db.add(settings)
            self.db.commit()
            self.db.refresh(settings)
        print(f"   ✅ Settings loaded (scale: {settings.icon_scale_min}-{settings.icon_scale_max}, threshold: {settings.icon_match_threshold})")

        print(f"   🗑️ Clearing old detections...")
        # First, delete any icon_label_matches that reference these detections
        from models.detection import IconLabelMatch
        icon_detection_ids = [
            d.id for d in self.db.query(IconDetection.id).filter(
                IconDetection.project_id == project.id
            ).all()
        ]
        if icon_detection_ids:
            match_deleted = self.db.query(IconLabelMatch).filter(
                IconLabelMatch.icon_detection_id.in_(icon_detection_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
            print(f"   🗑️ Cleared {match_deleted} old match(es)")
        
        # Now delete the icon detections
        deleted_count = self.db.query(IconDetection).filter(
            IconDetection.project_id == project.id
        ).delete()
        self.db.commit()
        print(f"   ✅ Cleared {deleted_count} old detection(s)")

        detections: List[IconDetection] = []
        print(f"   🔍 Starting detection for {len(templates)} template(s) across {len(pages)} page(s)...")
        
        for idx, template in enumerate(templates, 1):
            print(f"   📌 Template {idx}/{len(templates)}: Processing...")
            template_path = self._download_template(template)
            try:
                scales = self._generate_scales(settings.icon_scale_min, settings.icon_scale_max)
                # Use only 90-degree rotations for speed (most electrical symbols are axis-aligned)
                rotations = [0, 90, 180, 270]
                print(f"      ⚙️ Using {len(scales)} scales and {len(rotations)} rotations (90° steps for speed)")
                
                threshold = settings.icon_match_threshold
                
                detector = ElectricalSymbolDetector(
                    scales=scales,
                    rotations=rotations,
                    threshold=threshold,
                    nms_threshold=settings.nms_threshold,
                )
                template_img = cv2.imread(template_path)
                if template_img is None:
                    print(f"      ❌ Failed to read template image")
                    continue
                
                # Check if template is mostly white/empty (which would match everything)
                template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                mean_intensity = template_gray.mean()
                std_intensity = template_gray.std()
                
                if mean_intensity > 240 or std_intensity < 10:
                    print(f"      ⚠️ WARNING: Template appears to be mostly white/empty!")
                    print(f"         Mean intensity: {mean_intensity:.1f}, Std dev: {std_intensity:.1f}")
                    print(f"         This template may match everything. Consider redrawing the icon bbox.")
                
                detector.template = template_img
                detector.template_gray = template_gray
                print(f"      ✅ Template loaded (size: {template_img.shape}, mean: {mean_intensity:.1f}, std: {std_intensity:.1f})")

                for page_idx, page in enumerate(pages, 1):
                    print(f"      📄 Searching page {page_idx}/{len(pages)} (page #{page.page_number})...")
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        page_path = os.path.join(tmp_dir, f"{page.id}.png")
                        self.storage.download_file(page.image_url, page_path)
                        results, _ = detector.detect_symbols(page_path, visualize=False)
                        print(f"         🎯 Found {len(results)} match(es) on this page")

                    for result in results:
                        bbox = [int(v) for v in result.bbox]
                        center = [int(v) for v in result.center]
                        detection = IconDetection(
                            project_id=project.id,
                            icon_template_id=template.id,
                            page_id=page.id,
                            bbox=bbox,
                            center=center,
                            confidence=result.confidence,
                            scale=result.scale,
                            rotation=result.rotation,
                        )
                        self.db.add(detection)
                        detections.append(detection)
                        
                print(f"      ✅ Template {idx} complete: {len([d for d in detections if d.icon_template_id == template.id])} total detections")
            finally:
                if os.path.exists(template_path):
                    os.remove(template_path)

        print(f"   💾 Saving {len(detections)} detection(s) to database...")
        self.db.commit()
        for det in detections:
            self.db.refresh(det)
        print(f"   ✅ All detections saved!")
        return detections

    @staticmethod
    def _generate_scales(min_scale: float, max_scale: float, step: float = 0.1) -> List[float]:
        scales = []
        current = min_scale
        while current <= max_scale + 1e-6:
            scales.append(round(current, 2))
            current += step
        return scales or [1.0]

    @staticmethod
    def _crop_image(
        legend_path: str, output_path: str, bbox: tuple[int, int, int, int]
    ) -> None:
        from PIL import Image

        image = Image.open(legend_path)
        x, y, width, height = bbox
        cropped = image.crop((x, y, x + width, y + height))
        cropped.save(output_path)
        # Close file handles to avoid Windows lock issues
        cropped.close()
        image.close()

    def _download_template(self, template: IconTemplate) -> str:
        source_url = template.preprocessed_icon_url or template.cropped_icon_url
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            self.storage.download_file(source_url, tmp.name)
            return tmp.name


def get_icon_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
    preprocessing_service: PreprocessingService = Depends(get_preprocessing_service),
) -> IconService:
    return IconService(
        db=db,
        storage_service=storage_service,
        preprocessing_service=preprocessing_service,
    )
