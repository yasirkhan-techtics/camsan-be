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
        overlap_threshold: float = 0.5,  # 50% overlap threshold
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

    def find_overlapping_clusters(
        self,
        detections: List[LabelDetection],
    ) -> List[List[int]]:
        """
        Find clusters of overlapping detections using Union-Find.
        Multiple detections at the same location are grouped together.

        Args:
            detections: List of label detections

        Returns:
            List of clusters, where each cluster is a list of detection indices
        """
        n = len(detections)
        if n == 0:
            return []

        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Debug: print sample bboxes to understand the data
        if n > 0:
            print(f"      DEBUG: Sample bboxes (first 3):")
            for i in range(min(3, n)):
                bbox = detections[i].bbox
                print(f"         [{i}] bbox={bbox} (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")

        # Find all overlapping pairs and union them
        overlaps_found = []
        for i in range(n):
            bbox1 = detections[i].bbox
            for j in range(i + 1, n):
                bbox2 = detections[j].bbox
                ratio1, ratio2 = self.calculate_overlap_ratio(bbox1, bbox2)
                # Debug: log any overlap > 10%
                if ratio1 > 0.1 or ratio2 > 0.1:
                    overlaps_found.append((i, j, ratio1, ratio2, bbox1, bbox2))
                if ratio1 >= self.overlap_threshold or ratio2 >= self.overlap_threshold:
                    union(i, j)
        
        if overlaps_found:
            print(f"      DEBUG: Found {len(overlaps_found)} pairs with >10% overlap:")
            for i, j, r1, r2, b1, b2 in overlaps_found[:5]:  # Show first 5
                print(f"         [{i}] bbox={b1} vs [{j}] bbox={b2}: {r1*100:.1f}% / {r2*100:.1f}%")
            if len(overlaps_found) > 5:
                print(f"         ... and {len(overlaps_found) - 5} more")
        else:
            print(f"      DEBUG: No pairs found with >10% overlap among {n*(n-1)//2} comparisons")

        # Group by cluster
        clusters_dict = {}
        for i in range(n):
            root = find(i)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(i)

        # Only return clusters with more than 1 detection (actual overlaps)
        return [indices for indices in clusters_dict.values() if len(indices) > 1]

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

    def get_cluster_union_crop(
        self,
        image: Image.Image,
        detections: List[LabelDetection],
        h_padding_pct: float = 0.20,
        v_padding_pct: float = 0.10,
    ) -> Image.Image:
        """
        Get the union bounding box crop of all detections in a cluster.

        Args:
            image: Full page image
            detections: List of overlapping detections
            h_padding_pct: Horizontal padding percentage
            v_padding_pct: Vertical padding percentage

        Returns:
            Cropped union image
        """
        if not detections:
            return image
        
        # Calculate union of all bboxes
        min_x = min(d.bbox[0] for d in detections)
        min_y = min(d.bbox[1] for d in detections)
        max_x = max(d.bbox[0] + d.bbox[2] for d in detections)
        max_y = max(d.bbox[1] + d.bbox[3] for d in detections)

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

    def create_cluster_table_image(
        self,
        clusters: List[Dict],
    ) -> Image.Image:
        """
        Create a visual table for cluster classification (multiple tag options).

        Args:
            clusters: List of dicts with keys: 'sr_no', 'crop', 'tags'

        Returns:
            Combined table image
        """
        cell_height = 200
        sr_col_width = 80
        img_col_width = 300
        tags_col_width = 400
        total_width = sr_col_width + img_col_width + tags_col_width

        header_height = 60
        total_height = header_height + (cell_height * len(clusters))

        # Create white canvas
        table_img = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(table_img)

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            header_font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            header_font = font

        # Header
        header_color = "#2c3e50"
        draw.rectangle([0, 0, total_width, header_height], fill=header_color)
        draw.text((sr_col_width // 2, 20), "Sr.", fill="white", font=header_font, anchor="mm")
        draw.text((sr_col_width + img_col_width // 2, 20), "Overlapping Image", fill="white", font=header_font, anchor="mm")
        draw.text((sr_col_width + img_col_width + tags_col_width // 2, 20), "Tag Options", fill="white", font=header_font, anchor="mm")

        # Rows
        for idx, cluster in enumerate(clusters):
            y_start = header_height + (idx * cell_height)

            # Draw row border
            draw.rectangle(
                [0, y_start, total_width, y_start + cell_height],
                outline="gray",
                width=1,
            )

            # Sr. No
            draw.text(
                (sr_col_width // 2, y_start + cell_height // 2),
                str(cluster["sr_no"]),
                fill="black",
                font=font,
                anchor="mm",
            )

            # Draw column separators
            draw.line([(sr_col_width, y_start), (sr_col_width, y_start + cell_height)], fill="gray", width=1)
            draw.line([(sr_col_width + img_col_width, y_start), (sr_col_width + img_col_width, y_start + cell_height)], fill="gray", width=1)

            # Paste crop image
            crop = cluster["crop"]
            # Scale to fit
            max_img_width = img_col_width - 20
            max_img_height = cell_height - 20
            crop_ratio = min(max_img_width / crop.width, max_img_height / crop.height)
            new_size = (int(crop.width * crop_ratio), int(crop.height * crop_ratio))
            resized_crop = crop.resize(new_size, Image.Resampling.LANCZOS)

            paste_x = sr_col_width + (img_col_width - new_size[0]) // 2
            paste_y = y_start + (cell_height - new_size[1]) // 2
            table_img.paste(resized_crop, (paste_x, paste_y))

            # Tags column - list all options
            tags_x = sr_col_width + img_col_width + 20
            tags_y = y_start + 20
            for i, tag in enumerate(cluster["tags"]):
                draw.text(
                    (tags_x, tags_y + i * 30),
                    f"Option {i + 1}: {tag}",
                    fill="black",
                    font=font,
                )

        return table_img

    def classify_clusters_with_llm(
        self,
        clusters: List[Dict],
        table_path: str,
    ) -> Dict[int, str]:
        """
        Use LLM to classify which tag is correct for each cluster.

        Args:
            clusters: List of cluster data with 'sr_no' and 'tags'
            table_path: Path to the table image

        Returns:
            Dict mapping sr_no to selected tag name
        """
        # Build prompt with multiple options per cluster
        pairs_desc = []
        for cluster in clusters:
            options = " OR ".join([f'"{tag}"' for tag in cluster["tags"]])
            pairs_desc.append(f"  {cluster['sr_no']}. Options: {options}")

        prompt = f"""You are analyzing a table with {len(clusters)} overlapping tag/icon groups.

Each row shows:
- Sr. No: Identification number
- Overlapping Image: The union crop of overlapping detections at the same location
- Tag Options: Multiple possible tag names detected at this location

Your task: For each row, determine which tag option correctly identifies the content in the image.

Here are the groups:
{chr(10).join(pairs_desc)}

Provide your answer as JSON with this exact structure:
{{
  "classifications": [
    {{"sr_no": 1, "selected_tag": "exact_tag_name", "confidence": "high"}},
    {{"sr_no": 2, "selected_tag": "exact_tag_name", "confidence": "medium"}},
    ...
  ]
}}

Rules:
1. Classify ALL {len(clusters)} groups in the table
2. For each group, choose ONLY from its given options
3. Use EXACT tag names (case-sensitive)
4. Base decisions on visual content only
"""

        try:
            result = self.llm._invoke_with_structured_output(
                prompt, table_path, TagOverlapResolutionLLMResponse
            )

            classifications = {}
            if result and result.classifications:
                for item in result.classifications:
                    classifications[item.sr_no] = item.selected_tag
                    print(f"         LLM: Sr.{item.sr_no} -> '{item.selected_tag}' ({item.confidence})")

            return classifications

        except Exception as e:
            print(f"         LLM classification error: {e}")
            return {}

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

        # Load non-rejected label detections (pending or verified)
        # This works whether overlap removal runs before or after LLM verification
        query = (
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
        )

        if page_id:
            query = query.filter(LabelDetection.page_id == page_id)

        detections = query.order_by(LabelDetection.page_id).all()

        if not detections:
            print("   No tags found for overlap resolution")
            return {
                "total_tags": 0,
                "overlapping_pairs_found": 0,
                "tags_removed": 0,
                "tags_kept": len(detections) if detections else 0,
            }

        print(f"   Total tags to process: {len(detections)}")
        print(f"   Overlap threshold: {self.overlap_threshold * 100}%")

        # Group by page
        page_groups = {}
        for det in detections:
            if det.page_id not in page_groups:
                page_groups[det.page_id] = []
            page_groups[det.page_id].append(det)

        total_clusters_found = 0
        total_removed = 0
        ids_to_remove = set()

        for page_id, page_detections in page_groups.items():
            print(f"\n   Processing page {page_id} ({len(page_detections)} tags)...")

            # Find overlapping clusters (groups of detections at same location)
            overlapping_clusters = self.find_overlapping_clusters(page_detections)

            if not overlapping_clusters:
                print(f"      No overlapping tags found on this page")
                continue

            print(f"      Found {len(overlapping_clusters)} overlap clusters")
            total_clusters_found += len(overlapping_clusters)

            # Download page image
            with tempfile.TemporaryDirectory() as tmp_dir:
                page = page_detections[0].page
                page_path = os.path.join(tmp_dir, f"page_{page_id}.png")
                self.storage.download_file(page.image_url, page_path)
                page_image = Image.open(page_path)

                # Helper to get tag name from detection
                def get_tag_name(det, idx):
                    if det.label_template and det.label_template.tag_name:
                        return det.label_template.tag_name
                    if det.label_template and det.label_template.legend_item:
                        return det.label_template.legend_item.label_text
                    return f"Tag_{idx}"

                # Prepare cluster data for LLM
                clusters_for_llm = []
                same_tag_removed = 0

                for cluster_idx, cluster_indices in enumerate(overlapping_clusters):
                    cluster_dets = [page_detections[i] for i in cluster_indices]
                    cluster_tags = [get_tag_name(det, i) for det, i in zip(cluster_dets, cluster_indices)]
                    unique_tags = list(set(cluster_tags))

                    print(f"\n      Cluster {cluster_idx + 1}: {len(cluster_dets)} detections, tags: {cluster_tags}")

                    # SAME TAG CLUSTER: Keep highest confidence, remove rest
                    if len(unique_tags) == 1:
                        # All same tag - keep highest confidence
                        best_det = max(cluster_dets, key=lambda d: d.confidence)
                        for det in cluster_dets:
                            if det.id != best_det.id:
                                ids_to_remove.add(det.id)
                                same_tag_removed += 1
                                total_removed += 1
                        print(f"         Same tag '{unique_tags[0]}': kept best (conf={best_det.confidence:.3f}), removed {len(cluster_dets) - 1} duplicates")
                        continue

                    # DIFFERENT TAG CLUSTER: Need LLM to decide
                    # Get union crop of all detections in cluster
                    union_crop = self.get_cluster_union_crop(page_image, cluster_dets)
                    
                    clusters_for_llm.append({
                        "sr_no": len(clusters_for_llm) + 1,
                        "crop": union_crop,
                        "tags": unique_tags,
                        "detections": cluster_dets,
                        "tag_to_dets": {tag: [d for d, t in zip(cluster_dets, cluster_tags) if t == tag] for tag in unique_tags},
                    })

                if same_tag_removed > 0:
                    print(f"\n      Resolved {same_tag_removed} same-tag overlaps (no LLM needed)")

                # Skip LLM call if no different-tag clusters
                if not clusters_for_llm:
                    print(f"      No different-tag clusters to classify with LLM")
                    continue

                print(f"\n      Sending {len(clusters_for_llm)} different-tag clusters to LLM")

                # Process clusters in batches
                for batch_start in range(0, len(clusters_for_llm), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(clusters_for_llm))
                    batch_clusters = clusters_for_llm[batch_start:batch_end]

                    # Create table image for clusters
                    table_image = self.create_cluster_table_image(batch_clusters)
                    table_path = os.path.join(tmp_dir, f"overlap_table_{batch_start}.png")
                    table_image.save(table_path)

                    # Classify with LLM
                    classifications = self.classify_clusters_with_llm(batch_clusters, table_path)

                    # Process results
                    for cluster_data in batch_clusters:
                        sr_no = cluster_data["sr_no"]
                        selected_tag = classifications.get(sr_no)

                        if selected_tag and selected_tag in cluster_data["tag_to_dets"]:
                            # Keep all detections of selected tag, remove all others
                            keep_dets = cluster_data["tag_to_dets"][selected_tag]
                            for det in cluster_data["detections"]:
                                if det not in keep_dets:
                                    ids_to_remove.add(det.id)
                                    total_removed += 1
                            print(f"         Cluster {sr_no}: Keep '{selected_tag}' ({len(keep_dets)} dets), remove {len(cluster_data['detections']) - len(keep_dets)} others")
                        else:
                            # No valid classification - keep highest confidence detection
                            best_det = max(cluster_data["detections"], key=lambda d: d.confidence)
                            for det in cluster_data["detections"]:
                                if det.id != best_det.id:
                                    ids_to_remove.add(det.id)
                                    total_removed += 1
                            print(f"         Cluster {sr_no}: No valid classification, kept highest confidence")

                page_image.close()

        # Update database - mark removed tags as rejected
        for det_id in ids_to_remove:
            det = self.db.query(LabelDetection).filter(LabelDetection.id == det_id).first()
            if det:
                det.verification_status = "rejected"
                self.db.add(det)

        self.db.commit()

        summary = {
            "total_tags": len(detections),
            "overlapping_clusters_found": total_clusters_found,
            "tags_removed": total_removed,
            "tags_kept": len(detections) - total_removed,
        }

        print(f"\n{'='*60}")
        print(f"OVERLAP RESOLUTION COMPLETE")
        print(f"   Total tags analyzed: {summary['total_tags']}")
        print(f"   Overlapping clusters found: {summary['overlapping_clusters_found']}")
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

