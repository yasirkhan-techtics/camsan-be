"""
LLM-based verification service for icon and label detections.
Calculates dynamic confidence thresholds and verifies low-confidence detections using LLM.
"""

import os
import tempfile
from typing import Dict, List, Tuple, Union
from uuid import UUID

import numpy as np
from fastapi import Depends, HTTPException
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import IconDetection, LabelDetection, IconTemplate, LabelTemplate
from models.page import PDFPage
from models.project import Project
from services.llm_service import LLMService, get_llm_service
from services.storage_service import StorageService, get_storage_service
from schemas.llm_schemas import DetectionVerificationResponse


class LLMVerificationService:
    """
    Service for verifying detection results using LLM.
    Handles both icons and labels with dynamic confidence thresholds.
    """

    def __init__(
        self,
        db: Session,
        storage_service: StorageService,
        llm_service: LLMService,
    ):
        self.db = db
        self.storage = storage_service
        self.llm = llm_service

    def calculate_confidence_threshold(
        self,
        confidences: List[float],
        approaches: List[str] = ["iqr", "percentile", "std", "kmeans"],
    ) -> float:
        """
        Calculate dynamic confidence threshold using multiple approaches.
        Uses MAXIMUM of thresholds to be more conservative (fewer false negatives).

        Args:
            confidences: List of confidence scores
            approaches: List of approaches to use

        Returns:
            Combined threshold value
        """
        if len(confidences) < 3:
            return min(confidences) * 0.9 if confidences else 0.5

        confidences_arr = np.array(confidences)
        thresholds = []

        # IQR-based threshold
        if "iqr" in approaches:
            q1 = np.percentile(confidences_arr, 25)
            q3 = np.percentile(confidences_arr, 75)
            iqr = q3 - q1
            threshold_iqr = q1 - 1.5 * iqr
            thresholds.append(max(threshold_iqr, 0.0))

        # Percentile-based (15th percentile)
        if "percentile" in approaches:
            threshold_percentile = np.percentile(confidences_arr, 15)
            thresholds.append(threshold_percentile)

        # Standard deviation-based
        if "std" in approaches:
            mean_conf = np.mean(confidences_arr)
            std_conf = np.std(confidences_arr)
            threshold_std = mean_conf - 1.5 * std_conf
            thresholds.append(max(threshold_std, 0.0))

        # Simple K-means clustering (2 clusters)
        if "kmeans" in approaches and len(confidences_arr) >= 5:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(confidences_arr.reshape(-1, 1))
                centers = sorted(kmeans.cluster_centers_.flatten())
                threshold_kmeans = (centers[0] + centers[1]) / 2
                thresholds.append(threshold_kmeans)
            except Exception:
                pass

        # Use MAXIMUM to be more conservative (fewer false negatives)
        # This ensures we only auto-approve the most confident detections
        if thresholds:
            combined_threshold = float(np.max(thresholds))
            print(f"         Individual thresholds: {[f'{t:.4f}' for t in thresholds]}")
            print(f"         Selected threshold (max): {combined_threshold:.4f}")
            return combined_threshold

        return 0.5

    def filter_detections_by_confidence(
        self,
        detections: List[Union[IconDetection, LabelDetection]],
        detection_type: str,
    ) -> Tuple[List, List, Dict[str, float]]:
        """
        Filter detections into high and low confidence groups.

        Args:
            detections: List of detection records
            detection_type: 'icon' or 'label'

        Returns:
            high_conf_detections, low_conf_detections, thresholds_dict
        """
        print(f"\n{'='*60}")
        print(f"FILTERING {detection_type.upper()} DETECTIONS BY CONFIDENCE")
        print(f"{'='*60}")

        if not detections:
            return [], [], {}

        thresholds = {}
        high_conf = []
        low_conf = []

        if detection_type == "label":
            # Group labels by their specific tag name (e.g., "CL1", "CL2")
            groups = {}
            for det in detections:
                # Use tag_name from template first, fall back to legend_item.label_text
                tag_name = (
                    det.label_template.tag_name
                    if det.label_template and det.label_template.tag_name
                    else (
                        det.label_template.legend_item.label_text
                        if det.label_template and det.label_template.legend_item
                        else "unknown"
                    )
                )
                if tag_name not in groups:
                    groups[tag_name] = []
                groups[tag_name].append(det)

            for tag_name, group in groups.items():
                confidences = [d.confidence for d in group]
                threshold = self.calculate_confidence_threshold(confidences)
                thresholds[f"tag_{tag_name}"] = threshold

                for det in group:
                    if det.confidence >= threshold:
                        high_conf.append(det)
                    else:
                        low_conf.append(det)

                print(f"\n   Tag '{tag_name}': {len(group)} detections")
                print(f"   Threshold: {threshold:.4f}")
                print(f"   High conf: {len([d for d in group if d.confidence >= threshold])}")
                print(f"   Low conf: {len([d for d in group if d.confidence < threshold])}")

        else:  # icon
            confidences = [d.confidence for d in detections]
            threshold = self.calculate_confidence_threshold(confidences)
            thresholds["icon_all"] = threshold

            for det in detections:
                if det.confidence >= threshold:
                    high_conf.append(det)
                else:
                    low_conf.append(det)

            print(f"\n   Icons: {len(detections)} detections")
            print(f"   Threshold: {threshold:.4f}")
            print(f"   High conf: {len(high_conf)}, Low conf: {len(low_conf)}")

        print(f"\n{'='*60}")
        print(f"FILTERING COMPLETE")
        print(f"   Total High Confidence: {len(high_conf)} (Auto-approved)")
        print(f"   Total Low Confidence: {len(low_conf)} (Need LLM verification)")
        print(f"{'='*60}")

        return high_conf, low_conf, thresholds

    def crop_detection(
        self,
        page_path: str,
        bbox: List[float],
        padding_percent: float = 0.4,
    ) -> Image.Image:
        """
        Crop detection from image with padding.

        Args:
            page_path: Path to page image
            bbox: [x, y, w, h] bounding box
            padding_percent: Padding percentage

        Returns:
            Cropped PIL Image
        """
        img = Image.open(page_path)
        x, y, w, h = [int(v) for v in bbox]

        pad_x = int(w * padding_percent)
        pad_y = int(h * padding_percent)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.width, x + w + pad_x)
        y2 = min(img.height, y + h + pad_y)

        cropped = img.crop((x1, y1, x2, y2))
        img.close()
        return cropped

    def create_verification_table(
        self,
        crops: List[Image.Image],
        detection_ids: List[UUID],
        template_img: Image.Image = None,
    ) -> Image.Image:
        """
        Create verification table image with crops.

        Args:
            crops: List of cropped detection images
            detection_ids: List of detection UUIDs for tracking
            template_img: Optional template image for icons

        Returns:
            Table image
        """
        border_width = 3
        cell_padding = 20
        font_size = 24

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            font_bold = ImageFont.truetype("arialbd.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
                )
                font_bold = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
                )
            except OSError:
                font = ImageFont.load_default()
                font_bold = font

        # Resize crops to consistent max dimensions
        max_crop_width = 400
        max_crop_height = 300
        resized_crops = []

        for crop in crops:
            width_scale = max_crop_width / crop.width if crop.width > max_crop_width else 1.0
            height_scale = (
                max_crop_height / crop.height if crop.height > max_crop_height else 1.0
            )
            scale = min(width_scale, height_scale)

            if scale < 1.0:
                new_width = int(crop.width * scale)
                new_height = int(crop.height * scale)
                resized_crop = crop.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                resized_crops.append(resized_crop)
            else:
                resized_crops.append(crop)

        # Calculate column widths
        col1_width = 80  # Sr. No
        col2_width = max(img.width for img in resized_crops) + 2 * cell_padding
        col2_width = max(col2_width, 200)

        if template_img:
            # Resize template
            if template_img.width > 350 or template_img.height > 300:
                width_scale = 350 / template_img.width if template_img.width > 350 else 1.0
                height_scale = (
                    300 / template_img.height if template_img.height > 300 else 1.0
                )
                scale = min(width_scale, height_scale)
                new_width = int(template_img.width * scale)
                new_height = int(template_img.height * scale)
                template_img = template_img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

            col3_width = template_img.width + 2 * cell_padding
            col3_width = max(col3_width, 200)
            col4_width = 200  # Result column
            total_width = (
                col1_width + col2_width + col3_width + col4_width + 5 * border_width
            )
            headers = ["Sr.", "Detection", "Template", "Valid?"]
            col_widths = [col1_width, col2_width, col3_width, col4_width]
        else:
            col3_width = 200  # Result column
            total_width = col1_width + col2_width + col3_width + 4 * border_width
            headers = ["Sr.", "Image", "Valid?"]
            col_widths = [col1_width, col2_width, col3_width]

        # Calculate row heights
        header_height = 60
        row_heights = []
        for crop in resized_crops:
            if template_img:
                row_height = max(crop.height, template_img.height) + 2 * cell_padding
            else:
                row_height = crop.height + 2 * cell_padding
            row_height = max(row_height, 100)
            row_heights.append(row_height)

        total_height = (
            header_height + sum(row_heights) + (len(resized_crops) + 1) * border_width
        )

        # Create canvas
        table_img = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(table_img)

        # Draw header
        draw.rectangle([0, 0, total_width, header_height], fill="#E8E8E8")
        draw.line([(0, 0), (total_width, 0)], fill="black", width=border_width)
        draw.line(
            [(0, header_height), (total_width, header_height)],
            fill="black",
            width=border_width,
        )

        # Header text
        x = 0
        for header, col_width in zip(headers, col_widths):
            draw.line([(x, 0), (x, header_height)], fill="black", width=border_width)
            draw.text(
                (x + border_width + col_width // 2, header_height // 2),
                header,
                fill="black",
                font=font_bold,
                anchor="mm",
            )
            x += col_width + border_width
        draw.line([(x, 0), (x, header_height)], fill="black", width=border_width)

        # Draw data rows
        y = header_height + border_width

        for idx, (crop, row_height) in enumerate(zip(resized_crops, row_heights)):
            serial = idx + 1

            # Row borders
            draw.line([(0, y), (0, y + row_height)], fill="black", width=border_width)

            x = 0
            for col_width in col_widths:
                x += col_width + border_width
                draw.line([(x, y), (x, y + row_height)], fill="black", width=border_width)

            draw.line(
                [(0, y + row_height), (total_width, y + row_height)],
                fill="black",
                width=border_width,
            )

            # Sr. No
            draw.text(
                (border_width + col1_width // 2, y + row_height // 2),
                str(serial),
                fill="black",
                font=font,
                anchor="mm",
            )

            # Detection image
            img_x = border_width + col1_width + border_width
            img_x += (col2_width - crop.width) // 2
            img_y = y + (row_height - crop.height) // 2
            table_img.paste(crop, (img_x, img_y))

            # Template image (if provided)
            if template_img:
                img_x = (
                    border_width
                    + col1_width
                    + border_width
                    + col2_width
                    + border_width
                )
                img_x += (col3_width - template_img.width) // 2
                img_y = y + (row_height - template_img.height) // 2
                table_img.paste(template_img, (img_x, img_y))

            y += row_height + border_width

        return table_img

    def verify_batch_with_llm(
        self,
        table_path: str,
        verification_type: str,
        tag_name: str = None,
        num_items: int = 0,
    ) -> List[bool]:
        """
        Verify detections using LLM.

        Args:
            table_path: Path to verification table image
            verification_type: 'tag' or 'icon'
            tag_name: Tag name if verifying tags
            num_items: Number of items in the batch

        Returns:
            List of verification results (True = valid, False = invalid)
        """
        if verification_type == "tag":
            prompt = f"""You are verifying detection results for '{tag_name}' tags in a technical drawing.
The table shows {num_items} detected instances with serial numbers. Your task is to verify if each detection is a VALID instance of '{tag_name}'.

Context:
- These are technical P&ID (Piping and Instrumentation Diagram) symbols
- '{tag_name}' represents a specific type of tag/label in the drawing
- Valid detections should clearly show the tag text '{tag_name}' or similar variations
- Invalid detections might be: partial captures, wrong text, noise, or completely different elements

Instructions:
1. Examine each row carefully (Serial numbers 1 through {num_items})
2. For EACH serial number, determine if it's a valid '{tag_name}' detection
3. Respond with ONLY a JSON object with a "results" array

Response format:
{{"results": [{{"serial_number": 1, "matches": true}}, {{"serial_number": 2, "matches": false}}, ...]}}

Where matches=true means valid detection, matches=false means invalid.
CRITICAL: You MUST provide exactly {num_items} results."""
        else:
            prompt = f"""You are verifying icon detection results in a technical drawing by comparing detected instances with a reference template.

The table shows {num_items} detected instances:
- Column 1: Serial number
- Column 2: Detected icon instance  
- Column 3: Reference template icon
- Column 4: Your verification (to be filled)

Context:
- These are technical symbols/icons from P&ID diagrams
- The template shows what the correct icon should look like
- Detected instances should closely match the template in shape, structure, and key features
- Minor variations in size, rotation, or line thickness are acceptable

Instructions:
1. Compare each detected instance with the reference template
2. Focus on: overall shape, key structural elements, symbol characteristics
3. For EACH serial number (1 through {num_items}), determine if the detected instance MATCHES the template

Response format:
{{"results": [{{"serial_number": 1, "matches": true}}, {{"serial_number": 2, "matches": false}}, ...]}}

Where matches=true means detection matches template, matches=false means it doesn't.
CRITICAL: You MUST provide exactly {num_items} results."""

        try:
            response = self.llm.verify_batch(prompt, table_path)
            results = [item.matches for item in response.results]

            if len(results) != num_items:
                print(f"   WARNING: Expected {num_items} results, got {len(results)}")
                return []

            approved = sum(results)
            print(f"   LLM Results: {approved}/{num_items} approved")
            return results

        except Exception as e:
            print(f"   ERROR in LLM verification: {e}")
            return []

    def verify_icon_detections(
        self,
        project: Project,
        batch_size: int = 10,
    ) -> Dict:
        """
        Verify icon detections for a project.

        Args:
            project: Project to verify
            batch_size: Number of detections per LLM batch

        Returns:
            Summary of verification results
        """
        print(f"\n{'='*60}")
        print(f"ICON DETECTION VERIFICATION PIPELINE")
        print(f"{'='*60}")

        # Load non-rejected icon detections with templates
        # (excludes detections already rejected by overlap removal or previous runs)
        detections = (
            self.db.query(IconDetection)
            .filter(
                IconDetection.project_id == project.id,
                IconDetection.verification_status != "rejected",
            )
            .options(
                selectinload(IconDetection.icon_template).selectinload(
                    IconTemplate.legend_item
                ),
                selectinload(IconDetection.page),
            )
            .all()
        )

        if not detections:
            return {
                "total_detections": 0,
                "auto_approved": 0,
                "llm_approved": 0,
                "llm_rejected": 0,
                "threshold_used": {},
            }

        print(f"   Total icon detections: {len(detections)}")

        # Filter by confidence
        high_conf, low_conf, thresholds = self.filter_detections_by_confidence(
            detections, "icon"
        )

        # Auto-approve high confidence
        for det in high_conf:
            det.verification_status = "verified"
            self.db.add(det)

        auto_approved = len(high_conf)
        llm_approved = 0
        llm_rejected = 0

        if low_conf:
            print(f"\n   Processing {len(low_conf)} low-confidence detections...")

            # Group by template for batch processing
            template_groups = {}
            for det in low_conf:
                template_id = det.icon_template_id
                if template_id not in template_groups:
                    template_groups[template_id] = []
                template_groups[template_id].append(det)

            for template_id, group in template_groups.items():
                template = group[0].icon_template
                print(f"\n   Processing template: {template.legend_item.description if template.legend_item else 'Unknown'}")

                # Download template image
                with tempfile.TemporaryDirectory() as tmp_dir:
                    template_path = os.path.join(tmp_dir, "template.png")
                    template_url = (
                        template.preprocessed_icon_url or template.cropped_icon_url
                    )
                    self.storage.download_file(template_url, template_path)
                    template_img = Image.open(template_path)

                    # Process in batches
                    for batch_start in range(0, len(group), batch_size):
                        batch_end = min(batch_start + batch_size, len(group))
                        batch = group[batch_start:batch_end]

                        print(f"      Batch {batch_start // batch_size + 1}: {len(batch)} items")

                        # Create crops
                        crops = []
                        for det in batch:
                            page_path = os.path.join(tmp_dir, f"page_{det.page_id}.png")
                            if not os.path.exists(page_path):
                                self.storage.download_file(det.page.image_url, page_path)
                            crop = self.crop_detection(page_path, det.bbox, padding_percent=0.3)
                            crops.append(crop)

                        # Create table
                        table_img = self.create_verification_table(
                            crops,
                            [det.id for det in batch],
                            template_img,
                        )
                        table_path = os.path.join(tmp_dir, f"table_{batch_start}.png")
                        table_img.save(table_path)

                        # Verify with LLM
                        results = self.verify_batch_with_llm(
                            table_path, "icon", num_items=len(batch)
                        )

                        # Update detections
                        if len(results) == len(batch):
                            for det, is_valid in zip(batch, results):
                                det.verification_status = "verified" if is_valid else "rejected"
                                self.db.add(det)
                                if is_valid:
                                    llm_approved += 1
                                else:
                                    llm_rejected += 1
                        else:
                            # Mark as rejected if verification failed
                            for det in batch:
                                det.verification_status = "rejected"
                                self.db.add(det)
                            llm_rejected += len(batch)

                        # Clean up crops
                        for crop in crops:
                            crop.close()

                    template_img.close()

        self.db.commit()

        summary = {
            "total_detections": len(detections),
            "auto_approved": auto_approved,
            "llm_approved": llm_approved,
            "llm_rejected": llm_rejected,
            "threshold_used": thresholds,
        }

        print(f"\n{'='*60}")
        print(f"VERIFICATION COMPLETE")
        print(f"   Auto-approved: {auto_approved}")
        print(f"   LLM approved: {llm_approved}")
        print(f"   LLM rejected: {llm_rejected}")
        print(f"{'='*60}")

        return summary

    def verify_label_detections(
        self,
        project: Project,
        batch_size: int = 10,
    ) -> Dict:
        """
        Verify label detections for a project.

        Args:
            project: Project to verify
            batch_size: Number of detections per LLM batch

        Returns:
            Summary of verification results
        """
        print(f"\n{'='*60}")
        print(f"LABEL DETECTION VERIFICATION PIPELINE")
        print(f"{'='*60}")

        # Load non-rejected label detections with templates
        # (excludes detections already rejected by overlap removal or previous runs)
        detections = (
            self.db.query(LabelDetection)
            .filter(
                LabelDetection.project_id == project.id,
                LabelDetection.verification_status != "rejected",
            )
            .options(
                selectinload(LabelDetection.label_template).selectinload(
                    LabelTemplate.legend_item
                ),
                selectinload(LabelDetection.page),
            )
            .all()
        )

        if not detections:
            return {
                "total_detections": 0,
                "auto_approved": 0,
                "llm_approved": 0,
                "llm_rejected": 0,
                "threshold_used": {},
            }

        print(f"   Total label detections: {len(detections)}")

        # Filter by confidence
        high_conf, low_conf, thresholds = self.filter_detections_by_confidence(
            detections, "label"
        )

        # Auto-approve high confidence
        for det in high_conf:
            det.verification_status = "verified"
            self.db.add(det)

        auto_approved = len(high_conf)
        llm_approved = 0
        llm_rejected = 0

        if low_conf:
            print(f"\n   Processing {len(low_conf)} low-confidence detections...")

            # Group by specific tag name for batch processing (e.g., "CL1", "CL2")
            tag_groups = {}
            for det in low_conf:
                # Use tag_name from template first, fall back to legend_item.label_text
                tag_name = (
                    det.label_template.tag_name
                    if det.label_template and det.label_template.tag_name
                    else (
                        det.label_template.legend_item.label_text
                        if det.label_template and det.label_template.legend_item
                        else "unknown"
                    )
                )
                if tag_name not in tag_groups:
                    tag_groups[tag_name] = []
                tag_groups[tag_name].append(det)

            with tempfile.TemporaryDirectory() as tmp_dir:
                for tag_name, group in tag_groups.items():
                    print(f"\n   Processing tag: '{tag_name}' ({len(group)} detections)")

                    # Process in batches
                    for batch_start in range(0, len(group), batch_size):
                        batch_end = min(batch_start + batch_size, len(group))
                        batch = group[batch_start:batch_end]

                        print(f"      Batch {batch_start // batch_size + 1}: {len(batch)} items")

                        # Create crops
                        crops = []
                        for det in batch:
                            page_path = os.path.join(tmp_dir, f"page_{det.page_id}.png")
                            if not os.path.exists(page_path):
                                self.storage.download_file(det.page.image_url, page_path)
                            crop = self.crop_detection(page_path, det.bbox, padding_percent=0.4)
                            crops.append(crop)

                        # Create table (no template for tags)
                        table_img = self.create_verification_table(
                            crops,
                            [det.id for det in batch],
                        )
                        table_path = os.path.join(tmp_dir, f"table_{tag_name}_{batch_start}.png")
                        table_img.save(table_path)

                        # Verify with LLM
                        results = self.verify_batch_with_llm(
                            table_path, "tag", tag_name=tag_name, num_items=len(batch)
                        )

                        # Update detections
                        if len(results) == len(batch):
                            for det, is_valid in zip(batch, results):
                                det.verification_status = "verified" if is_valid else "rejected"
                                if not is_valid:
                                    det.rejection_source = "llm_verification"
                                self.db.add(det)
                                if is_valid:
                                    llm_approved += 1
                                else:
                                    llm_rejected += 1
                        else:
                            # Mark as rejected if verification failed
                            for det in batch:
                                det.verification_status = "rejected"
                                det.rejection_source = "llm_verification"
                                self.db.add(det)
                            llm_rejected += len(batch)

                        # Clean up crops
                        for crop in crops:
                            crop.close()

        self.db.commit()

        summary = {
            "total_detections": len(detections),
            "auto_approved": auto_approved,
            "llm_approved": llm_approved,
            "llm_rejected": llm_rejected,
            "threshold_used": thresholds,
        }

        print(f"\n{'='*60}")
        print(f"VERIFICATION COMPLETE")
        print(f"   Auto-approved: {auto_approved}")
        print(f"   LLM approved: {llm_approved}")
        print(f"   LLM rejected: {llm_rejected}")
        print(f"{'='*60}")

        return summary


def get_llm_verification_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> LLMVerificationService:
    return LLMVerificationService(
        db=db,
        storage_service=storage_service,
        llm_service=llm_service,
    )

