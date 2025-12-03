"""
LLM-based matcher service for unmatched icons and tags.
Uses Gemini to find matches for icons/tags that couldn't be matched by distance.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import cv2
import numpy as np
from fastapi import Depends
from PIL import Image
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import (
    IconDetection,
    IconLabelMatch,
    IconTemplate,
    LabelDetection,
    LabelTemplate,
)
from models.page import PDFPage
from models.project import Project
from services.llm_service import LLMService, get_llm_service
from services.storage_service import StorageService, get_storage_service
from schemas.llm_schemas import (
    IconMatchResult,
    TagMatchResult,
    TemplateVerificationResult,
    CombinedTagIconVerificationResult,
)


class LLMMatcherService:
    """
    Uses LLM to match unmatched icons and tags.
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

    def _calculate_padding_stats(
        self,
        matches: List[IconLabelMatch],
    ) -> Dict[str, float]:
        """
        Calculate average distance and size statistics from successfully matched pairs.

        Args:
            matches: List of matched IconLabelMatch records

        Returns:
            Dictionary with statistics
        """
        matched = [m for m in matches if m.match_status == "matched" and m.label_detection_id]

        if not matched:
            return {
                "avg_icon_width": 150,
                "avg_icon_height": 150,
                "avg_distance": 120,
                "padding_multiplier": 2.5,
            }

        # Calculate statistics from matched pairs
        distances = [m.distance for m in matched if m.distance > 0]
        avg_distance = sum(distances) / len(distances) if distances else 120

        icon_widths = []
        icon_heights = []
        for m in matched:
            if m.icon_detection and m.icon_detection.bbox:
                icon_widths.append(m.icon_detection.bbox[2])  # width
                icon_heights.append(m.icon_detection.bbox[3])  # height

        avg_icon_width = sum(icon_widths) / len(icon_widths) if icon_widths else 150
        avg_icon_height = sum(icon_heights) / len(icon_heights) if icon_heights else 150

        stats = {
            "avg_icon_width": avg_icon_width,
            "avg_icon_height": avg_icon_height,
            "avg_distance": avg_distance,
            "padding_multiplier": 2.5,
            "num_matched": len(matched),
        }

        print(f"   Padding stats from {len(matched)} matched pairs:")
        print(f"      Avg icon size: {avg_icon_width:.1f}x{avg_icon_height:.1f}px")
        print(f"      Avg distance: {avg_distance:.1f}px")

        return stats

    def _create_icon_verification_crop(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        bbox: List[int],
    ) -> np.ndarray:
        """
        Create a crop of the icon with bounding box overlay for template verification.
        
        Args:
            image: Full image as numpy array
            center: Center coordinates of the icon
            bbox: Icon bounding box [x, y, width, height]
            
        Returns:
            Cropped image with bounding box overlay
        """
        cx, cy = center
        img_h, img_w = image.shape[:2]
        
        icon_x, icon_y, icon_w, icon_h = bbox
        
        # Add padding around the icon
        pad_w = int(icon_w * 0.5)
        pad_h = int(icon_h * 0.5)
        
        crop_x1 = max(0, icon_x - pad_w)
        crop_y1 = max(0, icon_y - pad_h)
        crop_x2 = min(img_w, icon_x + icon_w + pad_w)
        crop_y2 = min(img_h, icon_y + icon_h + pad_h)
        
        icon_crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        # Calculate bbox position in crop coordinates
        x1_in_crop = icon_x - crop_x1
        y1_in_crop = icon_y - crop_y1
        x2_in_crop = x1_in_crop + icon_w
        y2_in_crop = y1_in_crop + icon_h
        
        # Draw semi-transparent magenta bounding box
        overlay = icon_crop.copy()
        cv2.rectangle(overlay, (x1_in_crop, y1_in_crop), (x2_in_crop, y2_in_crop), 
                      (255, 0, 255), thickness=2)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, icon_crop, 1 - alpha, 0, icon_crop)
        
        return icon_crop

    def _create_padded_crop(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        padding_stats: Dict,
        is_icon: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create a padded crop around an item with context.

        Args:
            image: Full image as numpy array
            center: Center coordinates of the item
            padding_stats: Statistics for padding calculation
            is_icon: True if cropping around icon, False for tag

        Returns:
            Cropped image and crop info dictionary
        """
        cx, cy = center
        img_h, img_w = image.shape[:2]

        # Calculate padding based on average distance and size
        if is_icon:
            padding = int(
                (padding_stats["avg_distance"]
                 + max(padding_stats["avg_icon_width"], padding_stats["avg_icon_height"]))
                * padding_stats["padding_multiplier"]
            )
        else:
            # For tags: larger window to see more icons
            padding = int(
                (padding_stats["avg_distance"]
                 + max(padding_stats["avg_icon_width"], padding_stats["avg_icon_height"]))
                * padding_stats["padding_multiplier"]
                * 1.5
            )

        # Calculate crop bounds
        x1 = max(0, cx - padding)
        y1 = max(0, cy - padding)
        x2 = min(img_w, cx + padding)
        y2 = min(img_h, cy + padding)

        # Crop image
        crop = image[y1:y2, x1:x2].copy()

        # Calculate center in crop coordinates
        center_in_crop = (cx - x1, cy - y1)

        # Draw marker at center (MAGENTA crosshair)
        cv2.drawMarker(crop, center_in_crop, (255, 0, 255), cv2.MARKER_CROSS, 30, 3)

        crop_info = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_in_crop": center_in_crop,
            "crop_size": (x2 - x1, y2 - y1),
            "padding": padding,
        }

        return crop, crop_info

    def _overlay_matched_icons(
        self,
        crop: np.ndarray,
        crop_info: Dict,
        matched_icons: List[IconDetection],
    ) -> np.ndarray:
        """
        Overlay semi-transparent boxes on already matched icons in the crop.

        Args:
            crop: Cropped image
            crop_info: Information about the crop
            matched_icons: List of matched icon detections

        Returns:
            Crop with overlaid matched icons
        """
        x1, y1 = crop_info["x1"], crop_info["y1"]
        overlay = crop.copy()

        for icon in matched_icons:
            icon_bbox = icon.bbox
            icon_cx = icon_bbox[0] + icon_bbox[2] / 2
            icon_cy = icon_bbox[1] + icon_bbox[3] / 2
            icon_w = icon_bbox[2]
            icon_h = icon_bbox[3]

            # Check if icon is within crop bounds
            if (
                x1 <= icon_cx <= x1 + crop_info["crop_size"][0]
                and y1 <= icon_cy <= y1 + crop_info["crop_size"][1]
            ):
                # Convert to crop coordinates
                icon_x_crop = int(icon_cx - x1 - icon_w / 2)
                icon_y_crop = int(icon_cy - y1 - icon_h / 2)

                # Draw semi-transparent gray box
                cv2.rectangle(
                    overlay,
                    (icon_x_crop, icon_y_crop),
                    (icon_x_crop + int(icon_w), icon_y_crop + int(icon_h)),
                    (128, 128, 128),
                    -1,
                )

        # Blend with original
        alpha = 0.5
        result = cv2.addWeighted(overlay, alpha, crop, 1 - alpha, 0)

        return result

    def _patch_matched_pairs_with_white(
        self,
        image: np.ndarray,
        matched_pairs: List[IconLabelMatch],
    ) -> np.ndarray:
        """
        Patch already matched icon and tag bounding boxes with white rectangles.
        This prevents the LLM from being confused by already-matched symbols.
        
        Args:
            image: Full page image as numpy array
            matched_pairs: List of matched IconLabelMatch records
            
        Returns:
            Image with white patches over matched pairs
        """
        patched_image = image.copy()
        patched_count = 0
        
        for match in matched_pairs:
            # Patch icon bbox
            if match.icon_detection and match.icon_detection.bbox:
                bbox = match.icon_detection.bbox  # [x, y, w, h]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[0] + bbox[2])
                y2 = int(bbox[1] + bbox[3])
                cv2.rectangle(patched_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
            
            # Patch tag bbox
            if match.label_detection and match.label_detection.bbox:
                bbox = match.label_detection.bbox  # [x, y, w, h]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[0] + bbox[2])
                y2 = int(bbox[1] + bbox[3])
                cv2.rectangle(patched_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
            
            patched_count += 1
        
        if patched_count > 0:
            print(f"   🎨 Patched {patched_count} matched pairs with white boxes")
        
        return patched_image

    def _find_closest_tag_detection(
        self,
        matched_tag_name: str,
        icon_center: Tuple[int, int],
        unassigned_tags: List[LabelDetection],
    ) -> Optional[Tuple[LabelDetection, float]]:
        """
        Find the closest unassigned tag detection matching the given tag name.
        
        Args:
            matched_tag_name: The tag name to search for
            icon_center: Center coordinates of the icon
            unassigned_tags: List of unassigned tag detections
            
        Returns:
            Tuple of (closest tag detection, distance) or None if not found
        """
        matching_tags = []
        
        for tag in unassigned_tags:
            # Get tag name from template or legend item
            tag_name = None
            if tag.label_template:
                tag_name = tag.label_template.tag_name
                if not tag_name and tag.label_template.legend_item:
                    tag_name = tag.label_template.legend_item.label_text
            
            if tag_name == matched_tag_name:
                # Calculate distance (convert to native float for DB compatibility)
                tag_center = (tag.center[0], tag.center[1])
                distance = float(np.sqrt(
                    (tag_center[0] - icon_center[0])**2 +
                    (tag_center[1] - icon_center[1])**2
                ))
                matching_tags.append((tag, distance))
        
        if not matching_tags:
            return None
        
        # Return the closest one
        matching_tags.sort(key=lambda x: x[1])
        return matching_tags[0]

    def _calculate_template_scales(
        self,
        matches: List[IconLabelMatch],
        template_shape: Tuple[int, int],
    ) -> List[float]:
        """
        Calculate scale range for template matching based on matched icon sizes.
        
        Args:
            matches: List of matched IconLabelMatch records
            template_shape: (height, width) of template image
            
        Returns:
            List of scales to try for template matching
        """
        template_h, template_w = template_shape
        scales = []
        
        matched = [m for m in matches if m.match_status == "matched" and m.icon_detection]
        
        for m in matched:
            if m.icon_detection and m.icon_detection.bbox:
                icon_w = m.icon_detection.bbox[2]
                icon_h = m.icon_detection.bbox[3]
                scale_w = icon_w / template_w if template_w > 0 else 1.0
                scale_h = icon_h / template_h if template_h > 0 else 1.0
                scales.append((scale_w + scale_h) / 2)
        
        if not scales:
            return [0.8, 1.0, 1.2]
        
        avg_scale = float(np.mean(scales))
        # Generate scale range (±20% from average)
        scale_range = [
            max(0.5, avg_scale - 0.2),
            avg_scale,
            min(2.0, avg_scale + 0.2)
        ]
        return list(set(scale_range))

    def _template_match_with_rotation(
        self,
        search_area_gray: np.ndarray,
        template_gray: np.ndarray,
        scales: List[float],
        rotation_step: int = 30,
        threshold: float = 0.5,
    ) -> Optional[Dict]:
        """
        Perform template matching with multiple scales and rotations.
        
        Args:
            search_area_gray: Grayscale search area
            template_gray: Grayscale template image
            scales: List of scales to try
            rotation_step: Rotation step in degrees
            threshold: Matching threshold
            
        Returns:
            Dict with match info or None
        """
        best_match = None
        best_score = threshold
        
        rotation_angles = list(range(0, 360, rotation_step))
        
        for scale in scales:
            template_h, template_w = template_gray.shape
            new_w = int(template_w * scale)
            new_h = int(template_h * scale)
            
            if new_w < 10 or new_h < 10:
                continue
            if new_w > search_area_gray.shape[1] or new_h > search_area_gray.shape[0]:
                continue
            
            template_resized = cv2.resize(template_gray, (new_w, new_h))
            
            for angle in rotation_angles:
                if angle == 0:
                    template_rotated = template_resized
                else:
                    center = (new_w // 2, new_h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    template_rotated = cv2.warpAffine(
                        template_resized, M, (new_w, new_h),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=255
                    )
                
                # Skip if rotated template is larger than search area
                if (template_rotated.shape[0] > search_area_gray.shape[0] or
                    template_rotated.shape[1] > search_area_gray.shape[1]):
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(
                    search_area_gray, template_rotated, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = {
                        'score': float(max_val),
                        'location': max_loc,
                        'scale': float(scale),
                        'rotation': angle,
                        'width': template_rotated.shape[1],
                        'height': template_rotated.shape[0]
                    }
        
        return best_match

    def _create_combined_tag_icon_verification_prompt(self, expected_tag: str) -> str:
        """Create combined prompt for tag verification AND icon detection in one call."""
        return f"""You are an expert at analyzing electrical construction drawings.

**YOUR TASK:** Perform TWO verification steps in sequence:

**IMAGES PROVIDED:**
1. **TEMPLATE SYMBOL** - The electrical symbol we're looking for
2. **SEARCH AREA** - Region with MAGENTA box highlighting text label

**STEP 1: TAG VERIFICATION**
Expected tag label: "{expected_tag}"
✓ **TAG CORRECT** if text inside/near magenta box matches "{expected_tag}" (exact or close match, minor OCR variations OK, tag can be rotated at any angle)
✗ **TAG INCORRECT** if text is completely different, unreadable, or missing

**STEP 2: ICON DETECTION** (only if tag is correct)
✓ **ICON DETECTED** if template symbol is clearly visible in search area near the tag
✗ **ICON NOT FOUND** if template symbol is not visible or only partial/different symbols present

**Response Format:**
You MUST respond with VALID JSON in this exact format:
{{
  "tag_correct": true or false,
  "detected_text": "actual text you see in magenta box",
  "tag_confidence": "high" or "medium" or "low",
  "tag_reasoning": "why text matches or doesn't match",
  "icon_detected": true or false,
  "icon_confidence": "high" or "medium" or "low",
  "icon_reasoning": "why icon is/isn't present"
}}

**RULES:**
- Respond ONLY with valid JSON
- Use true/false (lowercase) for boolean values
- If tag_correct is false, set icon_detected to false
"""

    def _create_template_verification_prompt(self) -> str:
        """Create prompt for verifying if detected icon matches the template."""
        return """You are comparing two electrical construction drawing symbols.

**IMAGES PROVIDED:**
1. **TEMPLATE SYMBOL** - The reference symbol we're looking for
2. **DETECTED SYMBOL** - The symbol found in the drawing (marked with MAGENTA bounding box)

**TASK:** Determine if these two symbols are the SAME type of electrical symbol.

**Instructions:**
1. Carefully examine BOTH images
2. Compare the overall shape and structure
3. Look for key identifying features (lines, circles, connections, orientation)
4. Be lenient with:
   - Minor size differences
   - Slight rotation variations
   - Small drawing style differences
5. Be strict with:
   - Overall symbol type (they must be fundamentally the same symbol)
   - Key structural elements

**Response Format:**
You MUST respond with VALID JSON in this exact format:
{
  "is_match": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation comparing both symbols"
}

**IMPORTANT:**
- Compare the ACTUAL symbols in both images
- Respond ONLY with valid JSON
- Use true/false (lowercase) for is_match
- Keep reasoning brief but specific to what you see in BOTH images
"""

    def _create_icon_matching_prompt(
        self,
        icon_name: str,
        available_tags: List[str] = None,
    ) -> str:
        """Create prompt for asking LLM to find matching tag for an icon."""
        # Format available tags as a list
        tags_list = ""
        if available_tags:
            tags_list = "\n".join([f"  - {tag}" for tag in available_tags])

        return f"""You are analyzing an electrical drawing to match a symbol with its label.

**TASK:** Find the label/tag that belongs to the ICON marked with a MAGENTA crosshair (+).

**Icon Type:** {icon_name}

**Available Labels (choose from these only):**
{tags_list}

**Instructions:**
1. Look at the icon marked with the MAGENTA crosshair
2. Search the surrounding area for a text label that identifies THIS specific icon
3. The label should be near the icon (typically within a few icon-widths)
4. Choose ONLY from the available labels listed above
5. For minor OCR variations (like "1" vs "I", "0" vs "O"), match the closest label
6. If multiple instances of same label exist, any match is valid

**Response Format:**
You MUST respond with VALID JSON in this exact format:
{{
  "match_found": true or false,
  "matched_tag": "label_name" or null,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}}

**IMPORTANT:**
- Respond ONLY with valid JSON
- Use true/false (lowercase) for match_found
- Use null (not "null" or "None") if no match
- Keep reasoning brief (one sentence)
"""

    def _create_tag_matching_prompt(
        self,
        tag_name: str,
        available_icon_types: List[str],
    ) -> str:
        """Create prompt for asking LLM to find matching icon for a tag."""
        icons_list = "\n".join([f"  - {icon}" for icon in available_icon_types])

        return f"""You are analyzing an electrical drawing to match labels with their symbols.

**TASK:** Find the ICON that this label belongs to.

**Label/Tag:** {tag_name} (marked with MAGENTA crosshair +)

**Note:** Already matched pairs have been removed (white patches). Look for visible icons only.

**Available Icon Types (choose from these only):**
{icons_list}

**Instructions:**
1. Look at the tag/label marked with the MAGENTA crosshair
2. Search the surrounding area for a nearby icon (symbol) that this label identifies
3. The icon should be near the tag (typically within a few icon-widths)
4. Choose ONLY from the available icon types listed above
5. If the icon type matches one in the list, select it even with minor drawing variations

**Response Format:**
You MUST respond with VALID JSON in this exact format:
{{
  "match_found": true or false,
  "matched_icon_type": "icon_type" or null,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}}

**IMPORTANT:**
- Respond ONLY with valid JSON
- Use true/false (lowercase) for match_found
- Use null (not "null" or "None") if no match
- Keep reasoning brief (one sentence)
"""

    def _get_matching_context(
        self,
        project: Project,
    ) -> Dict:
        """
        Get common context data needed for LLM matching operations.
        
        Returns:
            Dictionary with all_matches, unmatched_icons, unassigned_tags, padding_stats
        """
        # Get all matches for this project
        # Use outer joins to include all match types:
        # - matched: has both icon and label
        # - unmatched_icon: has icon, no label
        # - unassigned_tag: has label, no icon
        all_matches = (
            self.db.query(IconLabelMatch)
            .outerjoin(IconDetection, IconLabelMatch.icon_detection_id == IconDetection.id)
            .outerjoin(LabelDetection, IconLabelMatch.label_detection_id == LabelDetection.id)
            .filter(
                or_(
                    IconDetection.project_id == project.id,
                    LabelDetection.project_id == project.id,
                )
            )
            .options(
                selectinload(IconLabelMatch.icon_detection).selectinload(
                    IconDetection.icon_template
                ).selectinload(IconTemplate.legend_item),
                selectinload(IconLabelMatch.icon_detection).selectinload(
                    IconDetection.page
                ),
                selectinload(IconLabelMatch.label_detection).selectinload(
                    LabelDetection.label_template
                ).selectinload(LabelTemplate.legend_item),
            )
            .all()
        )

        # Calculate padding stats from matched pairs
        padding_stats = self._calculate_padding_stats(all_matches)

        # Get unmatched icons (matches with no label)
        unmatched_icons = [
            m for m in all_matches
            if m.match_status == "unmatched_icon" or m.label_detection_id is None
        ]

        # Get unassigned tags (verified tags not paired with an icon)
        # Only consider tags as "matched" if they're in a record with match_status="matched"
        # Tags in "unassigned_tag" records are still available for LLM matching
        matched_label_ids = {
            m.label_detection_id for m in all_matches 
            if m.label_detection_id and m.match_status == "matched"
        }
        all_verified_labels = (
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
            .all()
        )
        unassigned_tags = [
            t for t in all_verified_labels if t.id not in matched_label_ids
        ]

        return {
            "all_matches": all_matches,
            "unmatched_icons": unmatched_icons,
            "unassigned_tags": unassigned_tags,
            "padding_stats": padding_stats,
            "matched_label_ids": matched_label_ids,
        }

    def match_tags_for_unlabeled_icons(
        self,
        project: Project,
        save_crops: bool = False,
    ) -> Dict:
        """
        PHASE 5: Match unlabeled icons to tags using LLM.
        
        This method processes icons that have no assigned label and uses LLM
        to find matching tags by analyzing the surrounding area.
        
        Args:
            project: Project to process
            save_crops: Whether to save crop images for debugging
            
        Returns:
            Statistics about matching results
        """
        print(f"\n{'='*60}")
        print(f"PHASE 5: TAG MATCHING FOR UNLABELED ICONS")
        print(f"{'='*60}")

        # Get matching context
        context = self._get_matching_context(project)
        all_matches = context["all_matches"]
        unmatched_icons = context["unmatched_icons"]
        unassigned_tags = context["unassigned_tags"]
        padding_stats = context["padding_stats"]

        print(f"   Unmatched icons: {len(unmatched_icons)}")
        print(f"   Available unassigned tags: {len(unassigned_tags)}")

        if not unmatched_icons:
            print("   No unlabeled icons to process!")
            return {
                "total_unmatched_icons": 0,
                "icons_matched": 0,
                "icons_rejected": 0,
                "api_calls_made": 0,
            }

        # Get matched pairs for white patching
        matched_pairs = [
            m for m in all_matches
            if m.match_status == "matched" and m.label_detection_id
        ]

        # Get available tags (unique tag names)
        available_tags = list(set(
            t.label_template.tag_name or t.label_template.legend_item.label_text
            for t in unassigned_tags
            if t.label_template and (t.label_template.tag_name or t.label_template.legend_item)
        ))

        print(f"   Available tag types: {available_tags}")
        print(f"   Matched pairs to white-patch: {len(matched_pairs)}")

        stats = {
            "total_unmatched_icons": len(unmatched_icons),
            "icons_matched": 0,
            "icons_rejected": 0,
            "api_calls_made": 0,
        }

        used_icon_ids = set()
        used_tag_ids = set()  # Track used tags to avoid re-matching

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cache for patched page images
            patched_pages = {}
            
            for match in unmatched_icons:
                icon_det = match.icon_detection
                if not icon_det:
                    continue

                icon_name = (
                    icon_det.icon_template.legend_item.description
                    if icon_det.icon_template and icon_det.icon_template.legend_item
                    else "Unknown"
                )
                center = (int(icon_det.center[0]), int(icon_det.center[1]))
                bbox = icon_det.bbox  # [x, y, width, height]

                print(f"\n      Icon '{icon_name}' at {center}")

                # Download and patch page image (cached per page)
                page_id = str(icon_det.page_id)
                if page_id not in patched_pages:
                    page_path = os.path.join(tmp_dir, f"page_{page_id}.png")
                    if not os.path.exists(page_path):
                        self.storage.download_file(icon_det.page.image_url, page_path)
                    
                    raw_image = cv2.imread(page_path)
                    if raw_image is None:
                        continue
                    
                    # Get matched pairs for this page
                    page_matched_pairs = [
                        m for m in matched_pairs
                        if m.icon_detection and str(m.icon_detection.page_id) == page_id
                    ]
                    
                    # Apply white patching to matched pairs
                    patched_pages[page_id] = self._patch_matched_pairs_with_white(
                        raw_image, page_matched_pairs
                    )
                
                image = patched_pages[page_id]
                if image is None:
                    continue

                # STEP 1: Template Verification
                template_url = (
                    icon_det.icon_template.cropped_icon_url
                    if icon_det.icon_template
                    else None
                )
                
                if template_url:
                    print(f"         STEP 1: Template verification...")
                    
                    # Download template
                    template_path = os.path.join(tmp_dir, f"template_{icon_det.icon_template_id}.png")
                    if not os.path.exists(template_path):
                        self.storage.download_file(template_url, template_path)
                    
                    # Create icon verification crop with bounding box
                    icon_verification_crop = self._create_icon_verification_crop(
                        image, center, bbox
                    )
                    
                    # Save crops for LLM
                    icon_crop_path = os.path.join(tmp_dir, f"icon_{icon_det.id}_verify.png")
                    cv2.imwrite(icon_crop_path, icon_verification_crop)
                    
                    # Query LLM for template verification
                    template_prompt = self._create_template_verification_prompt()
                    
                    try:
                        verification_result = self.llm._invoke_with_template_comparison(
                            template_prompt, template_path, icon_crop_path, TemplateVerificationResult
                        )
                        stats["api_calls_made"] += 1
                        
                        if not verification_result.is_match:
                            print(f"         ❌ Template verification FAILED: {verification_result.reasoning}")
                            stats["icons_rejected"] += 1
                            continue
                        
                        print(f"         ✅ Template verified (conf={verification_result.confidence})")
                        
                    except Exception as e:
                        print(f"         ⚠️ Template verification error: {e}, proceeding anyway...")
                        stats["api_calls_made"] += 1
                
                # STEP 2: Find matching label
                print(f"         STEP 2: Finding matching label...")
                
                # Create padded crop for label search
                crop, crop_info = self._create_padded_crop(
                    image, center, padding_stats, is_icon=True
                )

                # Save crop
                crop_path = os.path.join(tmp_dir, f"icon_{icon_det.id}_label_search.png")
                cv2.imwrite(crop_path, crop)

                # Create prompt and query LLM
                prompt = self._create_icon_matching_prompt(icon_name, available_tags)

                try:
                    result = self.llm._invoke_with_structured_output(
                        prompt, crop_path, IconMatchResult
                    )
                    stats["api_calls_made"] += 1

                    if not result.match_found or not result.matched_tag:
                        print(f"         No match found: {result.reasoning}")
                        continue

                    matched_tag = result.matched_tag
                    
                    # Validate matched tag is in available tags
                    if matched_tag not in available_tags:
                        print(f"         ⚠️ Invalid tag '{matched_tag}' not in available tags, skipping...")
                        continue
                    
                    # Find closest unassigned tag detection for this tag name
                    remaining_tags = [t for t in unassigned_tags if t.id not in used_tag_ids]
                    closest_result = self._find_closest_tag_detection(
                        matched_tag, center, remaining_tags
                    )
                    
                    distance = 0.0
                    if closest_result:
                        closest_tag, distance = closest_result
                        # Link to the physical tag detection
                        match.label_detection_id = closest_tag.id
                        match.distance = distance
                        used_tag_ids.add(closest_tag.id)
                        print(f"         📍 Linked to tag detection at distance {distance:.1f}px")
                    
                    # Update match with LLM-assigned label
                    match.llm_assigned_label = matched_tag
                    match.match_confidence = 0.85 if result.confidence == "high" else 0.70 if result.confidence == "medium" else 0.55
                    match.match_method = "llm_tag_for_icon"  # Phase 5: Tag matching for icons
                    match.match_status = "matched"
                    self.db.add(match)

                    used_icon_ids.add(icon_det.id)
                    stats["icons_matched"] += 1

                    print(
                        f"         ✅ MATCHED to '{matched_tag}' "
                        f"(conf={result.confidence}, dist={distance:.1f}px)"
                    )

                except Exception as e:
                    print(f"         Error: {e}")
                    stats["api_calls_made"] += 1

        self.db.commit()

        print(f"\n{'='*60}")
        print(f"PHASE 5 COMPLETE")
        print(f"   Icons processed: {stats['total_unmatched_icons']}")
        print(f"   Icons matched: {stats['icons_matched']}")
        print(f"   Icons rejected (template mismatch): {stats['icons_rejected']}")
        print(f"   API calls made: {stats['api_calls_made']}")
        print(f"{'='*60}")

        return stats

    def match_icons_for_unlabeled_tags(
        self,
        project: Project,
        save_crops: bool = False,
    ) -> Dict:
        """
        PHASE 6: Detect icons for unassigned tags using LLM + Template Matching.
        
        This method processes tags that have no assigned icon and:
        1. Verifies tag text is correct using LLM
        2. Detects if icon is present near the tag using LLM
        3. Uses template matching to find precise icon bbox
        4. Creates new IconDetection and IconLabelMatch records
        
        Args:
            project: Project to process
            save_crops: Whether to save crop images for debugging
            
        Returns:
            Statistics about detection results
        """
        print(f"\n{'='*60}")
        print(f"PHASE 6: ICON DETECTION FOR UNASSIGNED TAGS")
        print(f"{'='*60}")

        # Get matching context (fresh data after Phase 5)
        context = self._get_matching_context(project)
        all_matches = context["all_matches"]
        unassigned_tags = context["unassigned_tags"]
        padding_stats = context["padding_stats"]

        print(f"   Unassigned tags: {len(unassigned_tags)}")

        if not unassigned_tags:
            print("   No unlabeled tags to process!")
            return {
                "total_unassigned_tags": 0,
                "tags_verified_incorrect": 0,
                "icons_detected_by_llm": 0,
                "icons_not_found": 0,
                "template_match_success": 0,
                "template_match_failed": 0,
                "tags_matched": 0,
                "api_calls_made": 0,
            }

        # Get matched pairs for white patching
        matched_pairs = [
            m for m in all_matches
            if m.match_status == "matched" and m.label_detection_id
        ]
        print(f"   Matched pairs to white-patch: {len(matched_pairs)}")

        # Get icon templates for template matching
        from models.legend import LegendItem
        legend_table_ids = [lt.id for lt in project.legend_tables]
        icon_templates = (
            self.db.query(IconTemplate)
            .join(IconTemplate.legend_item)
            .filter(LegendItem.legend_table_id.in_(legend_table_ids))
            .options(selectinload(IconTemplate.legend_item))
            .all()
        )
        
        if not icon_templates:
            print("   No icon templates available for matching!")
            return {
                "total_unassigned_tags": len(unassigned_tags),
                "tags_verified_incorrect": 0,
                "icons_detected_by_llm": 0,
                "icons_not_found": 0,
                "template_match_success": 0,
                "template_match_failed": 0,
                "tags_matched": 0,
                "api_calls_made": 0,
            }

        print(f"   Icon templates available: {len(icon_templates)}")

        stats = {
            "total_unassigned_tags": len(unassigned_tags),
            "tags_verified_incorrect": 0,
            "icons_detected_by_llm": 0,
            "icons_not_found": 0,
            "template_match_success": 0,
            "template_match_failed": 0,
            "tags_matched": 0,
            "api_calls_made": 0,
        }

        used_tags = set()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cache for patched page images (color and grayscale)
            patched_pages = {}
            patched_pages_gray = {}
            template_cache = {}  # Cache for downloaded templates
            
            for tag_det in unassigned_tags:
                tag_name = (
                    tag_det.label_template.tag_name
                    if tag_det.label_template and tag_det.label_template.tag_name
                    else (
                        tag_det.label_template.legend_item.label_text
                        if tag_det.label_template and tag_det.label_template.legend_item
                        else "Unknown"
                    )
                )

                if tag_det.id in used_tags:
                    continue

                # Get icon template for this tag's legend item
                icon_template = None
                if tag_det.label_template and tag_det.label_template.legend_item:
                    legend_item_id = tag_det.label_template.legend_item_id
                    for tpl in icon_templates:
                        if tpl.legend_item_id == legend_item_id:
                            icon_template = tpl
                            break
                
                if not icon_template:
                    print(f"\n      Tag '{tag_name}' - No icon template found, skipping")
                    continue

                center = (int(tag_det.center[0]), int(tag_det.center[1]))
                tag_bbox = tag_det.bbox  # [x, y, w, h]
                
                print(f"\n{'='*60}")
                print(f"   Processing Tag '{tag_name}' at {center}")
                print(f"{'='*60}")

                # Download and patch page image (cached per page)
                page_id = str(tag_det.page_id)
                if page_id not in patched_pages:
                    page_path = os.path.join(tmp_dir, f"page_{page_id}.png")
                    if not os.path.exists(page_path):
                        self.storage.download_file(tag_det.page.image_url, page_path)
                    
                    raw_image = cv2.imread(page_path)
                    if raw_image is None:
                        continue
                    
                    # Get matched pairs for this page
                    page_matched_pairs = [
                        m for m in matched_pairs
                        if m.icon_detection and str(m.icon_detection.page_id) == page_id
                    ]
                    
                    # Apply white patching to matched pairs
                    patched_color = self._patch_matched_pairs_with_white(
                        raw_image, page_matched_pairs
                    )
                    patched_pages[page_id] = patched_color
                    patched_pages_gray[page_id] = cv2.cvtColor(patched_color, cv2.COLOR_BGR2GRAY)
                
                image = patched_pages[page_id]
                image_gray = patched_pages_gray[page_id]
                if image is None:
                    continue

                # Download template image (cached)
                template_id = str(icon_template.id)
                if template_id not in template_cache:
                    template_path = os.path.join(tmp_dir, f"template_{template_id}.png")
                    template_url = icon_template.cropped_icon_url
                    self.storage.download_file(template_url, template_path)
                    template_img = cv2.imread(template_path)
                    if template_img is not None:
                        template_cache[template_id] = {
                            'color': template_img,
                            'gray': cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY),
                            'path': template_path,
                        }
                
                if template_id not in template_cache:
                    print(f"         Failed to load template image")
                    continue
                
                template_data = template_cache[template_id]

                # Create search area crop for LLM (with magenta box on tag)
                crop_llm, crop_info = self._create_padded_crop(
                    image, center, padding_stats, is_icon=False
                )
                
                # Draw magenta box around tag bbox in crop
                tag_x_in_crop = int(tag_bbox[0] - crop_info["x1"])
                tag_y_in_crop = int(tag_bbox[1] - crop_info["y1"])
                cv2.rectangle(
                    crop_llm,
                    (tag_x_in_crop, tag_y_in_crop),
                    (tag_x_in_crop + int(tag_bbox[2]), tag_y_in_crop + int(tag_bbox[3])),
                    (255, 0, 255), 3
                )
                
                # Create clean search area for template matching (no overlays)
                crop_clean, _ = self._create_padded_crop(
                    image, center, padding_stats, is_icon=False
                )
                crop_clean_gray = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2GRAY)

                # Save crops
                crop_llm_path = os.path.join(tmp_dir, f"tag_{tag_det.id}_llm.png")
                cv2.imwrite(crop_llm_path, crop_llm)

                # STEP 1: Combined LLM verification (tag + icon detection)
                print(f"         STEP 1: LLM Verification (tag + icon)...")
                
                combined_prompt = self._create_combined_tag_icon_verification_prompt(tag_name)
                
                try:
                    result = self.llm._invoke_with_template_comparison(
                        combined_prompt,
                        template_data['path'],
                        crop_llm_path,
                        CombinedTagIconVerificationResult
                    )
                    stats["api_calls_made"] += 1
                    
                    print(f"            Tag correct: {result.tag_correct} ({result.tag_confidence})")
                    print(f"            Detected text: '{result.detected_text}'")
                    print(f"            Icon detected: {result.icon_detected} ({result.icon_confidence})")
                    
                    if not result.tag_correct:
                        print(f"         ❌ TAG INCORRECT: {result.tag_reasoning}")
                        stats["tags_verified_incorrect"] += 1
                        continue
                    
                    if not result.icon_detected:
                        print(f"         ❌ ICON NOT DETECTED: {result.icon_reasoning}")
                        stats["icons_not_found"] += 1
                        continue
                    
                    print(f"         ✅ Tag verified & icon detected by LLM")
                    stats["icons_detected_by_llm"] += 1
                    
                except Exception as e:
                    print(f"         LLM error: {e}")
                    stats["api_calls_made"] += 1
                    continue

                # STEP 2: Template matching for precise bbox
                print(f"         STEP 2: Template matching for precise bbox...")
                
                # Calculate scales from matched pairs
                scales = self._calculate_template_scales(
                    all_matches, template_data['gray'].shape
                )
                
                # Try template matching with decreasing thresholds
                match_result = None
                for threshold in [0.6, 0.5, 0.4, 0.35]:
                    match_result = self._template_match_with_rotation(
                        crop_clean_gray,
                        template_data['gray'],
                        scales,
                        rotation_step=30,
                        threshold=threshold
                    )
                    if match_result:
                        print(f"            Match found at threshold {threshold}")
                        break
                
                if not match_result:
                    print(f"         ❌ TEMPLATE MATCHING FAILED")
                    stats["template_match_failed"] += 1
                    continue
                
                print(f"         ✅ Template match: score={match_result['score']:.3f}, "
                      f"scale={match_result['scale']:.3f}, rotation={match_result['rotation']}°")
                stats["template_match_success"] += 1

                # Transform bbox to original image coordinates
                bbox_x_crop = match_result['location'][0]
                bbox_y_crop = match_result['location'][1]
                bbox_w = match_result['width']
                bbox_h = match_result['height']
                
                bbox_x_original = crop_info['x1'] + bbox_x_crop
                bbox_y_original = crop_info['y1'] + bbox_y_crop
                
                # Calculate icon center
                icon_center_x = bbox_x_original + bbox_w / 2.0
                icon_center_y = bbox_y_original + bbox_h / 2.0
                
                # Calculate distance
                distance = float(np.sqrt(
                    (icon_center_x - center[0])**2 +
                    (icon_center_y - center[1])**2
                ))
                
                print(f"            Icon bbox: ({bbox_x_original}, {bbox_y_original}, {bbox_w}, {bbox_h})")
                print(f"            Icon center: ({icon_center_x:.1f}, {icon_center_y:.1f})")
                print(f"            Distance from tag: {distance:.1f}px")

                # Create new IconDetection record
                new_icon_detection = IconDetection(
                    project_id=project.id,
                    icon_template_id=icon_template.id,
                    page_id=tag_det.page_id,
                    bbox=[bbox_x_original, bbox_y_original, bbox_w, bbox_h],
                    center=[icon_center_x, icon_center_y],
                    confidence=match_result['score'],
                    scale=match_result['scale'],
                    rotation=match_result['rotation'],
                    verification_status="verified",
                )
                self.db.add(new_icon_detection)
                self.db.flush()  # Get the ID

                # Create IconLabelMatch record
                new_match = IconLabelMatch(
                    icon_detection_id=new_icon_detection.id,
                    label_detection_id=tag_det.id,
                    distance=distance,
                    match_confidence=match_result['score'],
                    match_method="llm_icon_for_tag",  # Phase 6: Icon detection for tags
                    match_status="matched",
                    llm_assigned_label=tag_name,
                )
                self.db.add(new_match)

                used_tags.add(tag_det.id)
                stats["tags_matched"] += 1

                print(f"         ✅ CREATED new IconDetection ID={new_icon_detection.id}")
                print(f"         ✅ MATCHED tag to new icon (dist={distance:.1f}px)")

        self.db.commit()

        print(f"\n{'='*60}")
        print(f"PHASE 6 COMPLETE")
        print(f"   Tags processed: {stats['total_unassigned_tags']}")
        print(f"   Tags verified incorrect: {stats['tags_verified_incorrect']}")
        print(f"   Icons detected by LLM: {stats['icons_detected_by_llm']}")
        print(f"   Icons not found by LLM: {stats['icons_not_found']}")
        print(f"   Template match success: {stats['template_match_success']}")
        print(f"   Template match failed: {stats['template_match_failed']}")
        print(f"   Tags matched (final): {stats['tags_matched']}")
        print(f"   API calls made: {stats['api_calls_made']}")
        print(f"{'='*60}")

        return stats

    def match_unmatched_items(
        self,
        project: Project,
        save_crops: bool = False,
    ) -> Dict:
        """
        Match unmatched icons and tags using LLM (backward compatibility).
        
        This method calls both Phase 5 and Phase 6 sequentially.
        For new implementations, use the separate methods:
        - match_tags_for_unlabeled_icons() for Phase 5
        - match_icons_for_unlabeled_tags() for Phase 6

        Args:
            project: Project to process
            save_crops: Whether to save crop images for debugging

        Returns:
            Combined statistics about matching results
        """
        print(f"\n{'='*60}")
        print(f"LLM-BASED ICON-TAG MATCHING (COMBINED)")
        print(f"{'='*60}")

        # Run Phase 5: Tag matching for unlabeled icons
        phase5_stats = self.match_tags_for_unlabeled_icons(project, save_crops)
        
        # Run Phase 6: Icon matching for unlabeled tags
        phase6_stats = self.match_icons_for_unlabeled_tags(project, save_crops)

        # Combine statistics
        combined_stats = {
            "total_unmatched_icons": phase5_stats["total_unmatched_icons"],
            "total_unassigned_tags": phase6_stats["total_unassigned_tags"],
            "icons_matched": phase5_stats["icons_matched"],
            "icons_rejected": phase5_stats["icons_rejected"],
            "tags_matched": phase6_stats["tags_matched"],
            "icons_detected_by_llm": phase6_stats.get("icons_detected_by_llm", 0),
            "template_match_success": phase6_stats.get("template_match_success", 0),
            "api_calls_made": phase5_stats["api_calls_made"] + phase6_stats["api_calls_made"],
        }

        print(f"\n{'='*60}")
        print(f"LLM MATCHING COMPLETE (COMBINED)")
        print(f"   Phase 5 - Icons matched: {combined_stats['icons_matched']}")
        print(f"   Phase 5 - Icons rejected: {combined_stats['icons_rejected']}")
        print(f"   Phase 6 - Tags matched (new icons created): {combined_stats['tags_matched']}")
        print(f"   Total API calls: {combined_stats['api_calls_made']}")
        print(f"{'='*60}")

        return combined_stats


def get_llm_matcher_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> LLMMatcherService:
    return LLMMatcherService(
        db=db,
        storage_service=storage_service,
        llm_service=llm_service,
    )

