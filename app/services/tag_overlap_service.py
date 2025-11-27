"""
Tag overlap resolution service.
Detects overlapping tag bounding boxes and resolves using LLM.
"""

import os
import tempfile
from typing import Dict, List, Tuple
from uuid import UUID

from fastapi import Depends
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import LabelDetection, LabelTemplate
from models.page import PDFPage
from models.project import Project
from services.llm_service import LLMService, get_llm_service
from services.storage_service import StorageService, get_storage_service
from schemas.llm_schemas import TagOverlapResolutionLLMResponse


class TagOverlapService:
    """
    Service to detect and resolve overlapping tag bounding boxes using LLM.
    """

    def __init__(
        self,
        db: Session,
        storage_service: StorageService,
        llm_service: LLMService,
        overlap_threshold: float = 0.9,
        batch_size: int = 10,
    ):
        self.db = db
        self.storage = storage_service
        self.llm = llm_service
        self.overlap_threshold = overlap_threshold
        self.batch_size = batch_size

    def calculate_overlap_ratio(
        self,
        bbox1: List[float],
        bbox2: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate overlap ratio for both boxes.
        
        Args:
            bbox1: [x, y, w, h] for first box
            bbox2: [x, y, w, h] for second box
            
        Returns:
            (overlap_ratio_box1, overlap_ratio_box2)
        """
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2

        # Calculate intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x1_1 + w1, x1_2 + w2)
        inter_y2 = min(y1_1 + h1, y1_2 + h2)

        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area1 = w1 * h1
        area2 = w2 * h2

        ratio1 = intersection / area1 if area1 > 0 else 0
        ratio2 = intersection / area2 if area2 > 0 else 0

        return ratio1, ratio2

    def find_overlapping_pairs(
        self,
        detections: List[LabelDetection],
    ) -> List[Tuple[int, int, float, float]]:
        """
        Find all pairs of detections where bboxes overlap >= threshold.

        Args:
            detections: List of label detections

        Returns:
            List of (idx1, idx2, overlap_ratio1, overlap_ratio2)
        """
        overlapping_pairs = []

        for i in range(len(detections)):
            bbox1 = detections[i].bbox

            for j in range(i + 1, len(detections)):
                bbox2 = detections[j].bbox

                ratio1, ratio2 = self.calculate_overlap_ratio(bbox1, bbox2)

                # Check if either box has >= threshold overlap
                if ratio1 >= self.overlap_threshold or ratio2 >= self.overlap_threshold:
                    overlapping_pairs.append((i, j, ratio1, ratio2))

        return overlapping_pairs

    def get_union_crop(
        self,
        image: Image.Image,
        det1: LabelDetection,
        det2: LabelDetection,
        h_padding_pct: float = 0.20,
        v_padding_pct: float = 0.10,
    ) -> Image.Image:
        """
        Get the union bounding box crop of two overlapping boxes with padding.

        Args:
            image: Full page image
            det1: First detection
            det2: Second detection
            h_padding_pct: Horizontal padding percentage
            v_padding_pct: Vertical padding percentage

        Returns:
            Cropped union image
        """
        x1_1, y1_1, w1, h1 = det1.bbox
        x1_2, y1_2, w2, h2 = det2.bbox

        # Calculate union bbox
        min_x = min(x1_1, x1_2)
        min_y = min(y1_1, y1_2)
        max_x = max(x1_1 + w1, x1_2 + w2)
        max_y = max(y1_1 + h1, y1_2 + h2)

        union_width = max_x - min_x
        union_height = max_y - min_y

        # Add padding
        h_padding = int(union_width * h_padding_pct)
        v_padding = int(union_height * v_padding_pct)

        padded_min_x = max(0, min_x - h_padding)
        padded_min_y = max(0, min_y - v_padding)
        padded_max_x = min(image.width, max_x + h_padding)
        padded_max_y = min(image.height, max_y + v_padding)

        return image.crop((padded_min_x, padded_min_y, padded_max_x, padded_max_y))

    def create_batch_table_image(
        self,
        batch_data: List[Dict],
    ) -> Image.Image:
        """
        Create a visual table with all overlapping pairs in the batch.

        Args:
            batch_data: List of dicts with keys: 'sr_no', 'crop', 'tag1', 'tag2'

        Returns:
            Combined table image
        """
        cell_height = 200
        sr_col_width = 80
        img_col_width = 300
        tags_col_width = 350
        total_width = sr_col_width + img_col_width + tags_col_width

        header_height = 60
        total_height = header_height + (cell_height * len(batch_data))

        # Create white canvas
        table_img = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(table_img)

        try:
            font_header = ImageFont.truetype("arial.ttf", 20)
            font_cell = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            try:
                font_header = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
                )
                font_cell = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
                )
            except OSError:
                font_header = ImageFont.load_default()
                font_cell = ImageFont.load_default()

        # Draw header
        draw.rectangle(
            [0, 0, total_width, header_height], fill="#2c3e50", outline="black", width=2
        )
        draw.text(
            (sr_col_width // 2, header_height // 2),
            "Sr.",
            fill="white",
            font=font_header,
            anchor="mm",
        )
        draw.text(
            (sr_col_width + img_col_width // 2, header_height // 2),
            "Overlapping Image",
            fill="white",
            font=font_header,
            anchor="mm",
        )
        draw.text(
            (sr_col_width + img_col_width + tags_col_width // 2, header_height // 2),
            "Tags to Classify",
            fill="white",
            font=font_header,
            anchor="mm",
        )

        # Draw rows
        for i, data in enumerate(batch_data):
            y_offset = header_height + (i * cell_height)

            # Draw cell borders
            draw.rectangle(
                [0, y_offset, sr_col_width, y_offset + cell_height],
                outline="black",
                width=2,
            )
            draw.rectangle(
                [sr_col_width, y_offset, sr_col_width + img_col_width, y_offset + cell_height],
                outline="black",
                width=2,
            )
            draw.rectangle(
                [sr_col_width + img_col_width, y_offset, total_width, y_offset + cell_height],
                outline="black",
                width=2,
            )

            # Sr. No
            draw.text(
                (sr_col_width // 2, y_offset + cell_height // 2),
                str(data["sr_no"]),
                fill="black",
                font=font_header,
                anchor="mm",
            )

            # Crop image (resize to fit)
            crop = data["crop"].copy()
            crop.thumbnail((img_col_width - 20, cell_height - 20), Image.Resampling.LANCZOS)
            crop_x = sr_col_width + (img_col_width - crop.width) // 2
            crop_y = y_offset + (cell_height - crop.height) // 2
            table_img.paste(crop, (crop_x, crop_y))

            # Tags text
            tags_text = f"Option 1: {data['tag1']}\n\nOption 2: {data['tag2']}"
            text_x = sr_col_width + img_col_width + 10
            text_y = y_offset + cell_height // 2 - 20
            draw.text((text_x, text_y), tags_text, fill="black", font=font_cell)

        return table_img

    def classify_batch_with_llm(
        self,
        batch_data: List[Dict],
        table_path: str,
    ) -> Dict[int, str]:
        """
        Send batch of overlapping pairs to LLM for classification.

        Args:
            batch_data: List of dicts with overlap info
            table_path: Path to the table image

        Returns:
            Dictionary mapping sr_no to selected tag
        """
        pairs_info = []
        for data in batch_data:
            pairs_info.append(
                f"  {data['sr_no']}. Options: \"{data['tag1']}\" OR \"{data['tag2']}\""
            )

        pairs_text = "\n".join(pairs_info)

        prompt = f"""You are analyzing a table with {len(batch_data)} overlapping tag/icon pairs.
Each row shows:
- Sr. No: Identification number
- Overlapping Image: The union crop of two overlapping detections
- Tags to Classify: Two possible tag names (Option 1 and Option 2)

Your task: For each row, determine which tag option correctly identifies the content in the image.

Here are the pairs:
{pairs_text}

Provide your answer as JSON with this exact structure:
{{
  "classifications": [
    {{"sr_no": 1, "selected_tag": "exact_tag_name", "confidence": "high"}},
    {{"sr_no": 2, "selected_tag": "exact_tag_name", "confidence": "medium"}},
    ...
  ]
}}

Rules:
1. Classify ALL {len(batch_data)} pairs in the table
2. For each pair, choose ONLY from its two given options
3. Use EXACT tag names (case-sensitive)
4. Base decisions on visual content only"""

        print(f"      Sending batch of {len(batch_data)} pairs to LLM...")

        try:
            response = self.llm._invoke_with_structured_output(
                prompt, table_path, TagOverlapResolutionLLMResponse
            )

            classifications = {}
            for item in response.classifications:
                sr_no = item.sr_no
                selected_tag = item.selected_tag
                confidence = item.confidence

                if sr_no and selected_tag:
                    classifications[sr_no] = selected_tag
                    print(f"      Sr.{sr_no}: '{selected_tag}' (confidence: {confidence})")

            if len(classifications) != len(batch_data):
                print(
                    f"      WARNING: Expected {len(batch_data)} classifications, got {len(classifications)}"
                )

            return classifications

        except Exception as e:
            print(f"      ERROR parsing LLM response: {e}")
            return {}

    def resolve_overlaps(
        self,
        project: Project,
        page_id: UUID = None,
    ) -> Dict:
        """
        Main processing function: detect overlapping bbox tags and resolve using LLM.

        Args:
            project: Project to process
            page_id: Optional specific page ID to process

        Returns:
            Summary dict with resolution statistics
        """
        print(f"\n{'='*60}")
        print(f"TAG OVERLAP RESOLUTION")
        print(f"{'='*60}")

        # Load verified label detections
        query = (
            self.db.query(LabelDetection)
            .filter(
                LabelDetection.project_id == project.id,
                LabelDetection.verification_status == "verified",
            )
            .options(
                selectinload(LabelDetection.label_template).selectinload(
                    LabelTemplate.legend_item
                ),
                selectinload(LabelDetection.page),
            )
        )

        if page_id:
            query = query.filter(LabelDetection.page_id == page_id)

        detections = query.order_by(LabelDetection.page_id).all()

        if not detections:
            print("   No verified tags found")
            return {
                "total_tags": 0,
                "overlapping_pairs_found": 0,
                "tags_removed": 0,
                "tags_kept": len(detections) if detections else 0,
            }

        print(f"   Total verified tags: {len(detections)}")
        print(f"   Overlap threshold: {self.overlap_threshold * 100}%")

        # Group by page
        page_groups = {}
        for det in detections:
            if det.page_id not in page_groups:
                page_groups[det.page_id] = []
            page_groups[det.page_id].append(det)

        total_pairs_found = 0
        total_removed = 0
        indices_to_remove = set()

        for page_id, page_detections in page_groups.items():
            print(f"\n   Processing page {page_id} ({len(page_detections)} tags)...")

            # Find overlapping pairs
            overlapping_pairs = self.find_overlapping_pairs(page_detections)

            if not overlapping_pairs:
                print(f"      No overlapping tags found on this page")
                continue

            print(f"      Found {len(overlapping_pairs)} overlapping pairs")
            total_pairs_found += len(overlapping_pairs)

            # Download page image
            with tempfile.TemporaryDirectory() as tmp_dir:
                page = page_detections[0].page
                page_path = os.path.join(tmp_dir, f"page_{page_id}.png")
                self.storage.download_file(page.image_url, page_path)
                page_image = Image.open(page_path)

                # Process in batches
                num_batches = (
                    len(overlapping_pairs) + self.batch_size - 1
                ) // self.batch_size

                for batch_num in range(num_batches):
                    batch_start = batch_num * self.batch_size
                    batch_end = min(
                        batch_start + self.batch_size, len(overlapping_pairs)
                    )
                    batch_pairs = overlapping_pairs[batch_start:batch_end]

                    print(f"\n      Batch {batch_num + 1}/{num_batches}")

                    # Prepare batch data
                    batch_data = []
                    for local_idx, (idx1, idx2, overlap1, overlap2) in enumerate(
                        batch_pairs
                    ):
                        sr_no = batch_start + local_idx + 1
                        det1 = page_detections[idx1]
                        det2 = page_detections[idx2]

                        # Get tag names
                        tag1 = (
                            det1.label_template.legend_item.label_text
                            if det1.label_template and det1.label_template.legend_item
                            else f"Tag_{idx1}"
                        )
                        tag2 = (
                            det2.label_template.legend_item.label_text
                            if det2.label_template and det2.label_template.legend_item
                            else f"Tag_{idx2}"
                        )

                        # Get union crop
                        union_crop = self.get_union_crop(page_image, det1, det2)

                        batch_data.append(
                            {
                                "sr_no": sr_no,
                                "crop": union_crop,
                                "tag1": tag1,
                                "tag2": tag2,
                                "idx1": idx1,
                                "idx2": idx2,
                                "det1": det1,
                                "det2": det2,
                            }
                        )

                    # Create table image
                    table_image = self.create_batch_table_image(batch_data)
                    table_path = os.path.join(
                        tmp_dir, f"overlap_table_{batch_num}.png"
                    )
                    table_image.save(table_path)

                    # Classify with LLM
                    classifications = self.classify_batch_with_llm(batch_data, table_path)

                    # Process results
                    for data in batch_data:
                        sr_no = data["sr_no"]
                        selected_tag = classifications.get(sr_no)

                        if selected_tag:
                            # Determine which to keep/remove
                            if selected_tag == data["tag1"]:
                                keep_det = data["det1"]
                                remove_det = data["det2"]
                            else:
                                keep_det = data["det2"]
                                remove_det = data["det1"]

                            # Mark for removal
                            indices_to_remove.add(remove_det.id)
                            print(
                                f"         Sr.{sr_no}: Keep '{selected_tag}', remove other"
                            )
                        else:
                            # If no classification, keep the first one
                            print(
                                f"         Sr.{sr_no}: No classification, keeping first tag"
                            )

                    # Clean up crops
                    for data in batch_data:
                        data["crop"].close()

                page_image.close()

        # Update database - mark removed tags as rejected
        for det_id in indices_to_remove:
            det = self.db.query(LabelDetection).filter(LabelDetection.id == det_id).first()
            if det:
                det.verification_status = "rejected"
                self.db.add(det)
                total_removed += 1

        self.db.commit()

        summary = {
            "total_tags": len(detections),
            "overlapping_pairs_found": total_pairs_found,
            "tags_removed": total_removed,
            "tags_kept": len(detections) - total_removed,
        }

        print(f"\n{'='*60}")
        print(f"OVERLAP RESOLUTION COMPLETE")
        print(f"   Total tags analyzed: {summary['total_tags']}")
        print(f"   Overlapping pairs found: {summary['overlapping_pairs_found']}")
        print(f"   Tags removed: {summary['tags_removed']}")
        print(f"   Tags kept: {summary['tags_kept']}")
        print(f"{'='*60}")

        return summary


def get_tag_overlap_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> TagOverlapService:
    return TagOverlapService(
        db=db,
        storage_service=storage_service,
        llm_service=llm_service,
    )

