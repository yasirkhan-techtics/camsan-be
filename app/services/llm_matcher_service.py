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
from schemas.llm_schemas import IconMatchResult, TagMatchResult, TemplateVerificationResult


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
        # If available_tags provided, include them as hints
        tags_hint = ""
        if available_tags:
            tags_list = ", ".join(available_tags)
            tags_hint = f"\n**Common label formats in this drawing:** {tags_list}"

        return f"""You are analyzing an electrical drawing to match a symbol with its label.

**TASK:** Find the label/tag that belongs to the ICON marked with a MAGENTA crosshair (+).

**Icon Type:** {icon_name}
{tags_hint}

**Instructions:**
1. Look at the icon marked with the MAGENTA crosshair
2. Search the surrounding area for a text label that identifies THIS specific icon
3. The label should be near the icon (typically within a few icon-widths)
4. Read the EXACT text of the label you see (e.g., "CL1", "S2", "P-101")
5. Labels are typically alphanumeric codes near the symbol

**Response Format:**
You MUST respond with VALID JSON in this exact format:
{{
  "match_found": true or false,
  "matched_tag": "exact_label_text" or null,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}}

**IMPORTANT:**
- Respond ONLY with valid JSON
- Use true/false (lowercase) for match_found
- Return the EXACT text you see for matched_tag
- Use null (not "null" or "None") if no label visible
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

**Note:** Already matched icons are shown with GRAY semi-transparent overlay. Look for icons WITHOUT gray overlay.

**Available Icon Types:**
{icons_list}

**Instructions:**
1. Look at the tag/label marked with the MAGENTA crosshair
2. Search for a nearby icon (symbol) that this label identifies
3. IGNORE icons with gray overlay (already matched)
4. Choose ONLY from the available icon types listed above

**Response Format:**
Respond with JSON in this exact format:
{{
  "match_found": true or false,
  "matched_icon_type": "icon_type" or null,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}}

If NO match found:
{{
  "match_found": false,
  "matched_icon_type": null,
  "confidence": "low",
  "reasoning": "No unmatched icons visible near the tag"
}}"""

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
        all_matches = (
            self.db.query(IconLabelMatch)
            .join(IconDetection)
            .filter(IconDetection.project_id == project.id)
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

        # Get unassigned tags (verified tags not in any match)
        matched_label_ids = {
            m.label_detection_id for m in all_matches if m.label_detection_id
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

        # Get available tags
        available_tags = list(set(
            t.label_template.legend_item.label_text
            for t in unassigned_tags
            if t.label_template and t.label_template.legend_item
        ))

        print(f"   Available tag types: {available_tags}")

        stats = {
            "total_unmatched_icons": len(unmatched_icons),
            "icons_matched": 0,
            "icons_rejected": 0,
            "api_calls_made": 0,
        }

        used_icon_ids = set()

        with tempfile.TemporaryDirectory() as tmp_dir:
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

                # Download page image
                page_path = os.path.join(tmp_dir, f"page_{icon_det.page_id}.png")
                if not os.path.exists(page_path):
                    self.storage.download_file(icon_det.page.image_url, page_path)

                image = cv2.imread(page_path)
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
                    
                    # Update match with LLM-assigned label
                    match.llm_assigned_label = matched_tag
                    match.distance = 0.0
                    match.match_confidence = 0.85 if result.confidence == "high" else 0.70 if result.confidence == "medium" else 0.55
                    match.match_method = "llm_tag_for_icon"
                    match.match_status = "matched"
                    self.db.add(match)

                    used_icon_ids.add(icon_det.id)
                    stats["icons_matched"] += 1

                    print(
                        f"         ✅ MATCHED to '{matched_tag}' "
                        f"(conf={result.confidence}, LLM-assigned)"
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
        PHASE 6: Match unlabeled tags to icons using LLM.
        
        This method processes tags that have no assigned icon and uses LLM
        to find matching icons by analyzing the surrounding area.
        
        Args:
            project: Project to process
            save_crops: Whether to save crop images for debugging
            
        Returns:
            Statistics about matching results
        """
        print(f"\n{'='*60}")
        print(f"PHASE 6: ICON MATCHING FOR UNLABELED TAGS")
        print(f"{'='*60}")

        # Get matching context (fresh data after Phase 5)
        context = self._get_matching_context(project)
        all_matches = context["all_matches"]
        unmatched_icons = context["unmatched_icons"]
        unassigned_tags = context["unassigned_tags"]
        padding_stats = context["padding_stats"]

        print(f"   Unassigned tags: {len(unassigned_tags)}")
        print(f"   Remaining unmatched icons: {len(unmatched_icons)}")

        if not unassigned_tags:
            print("   No unlabeled tags to process!")
            return {
                "total_unassigned_tags": 0,
                "tags_matched": 0,
                "api_calls_made": 0,
            }

        if not unmatched_icons:
            print("   No unmatched icons available to assign!")
            return {
                "total_unassigned_tags": len(unassigned_tags),
                "tags_matched": 0,
                "api_calls_made": 0,
            }

        # Get available icon types
        available_icon_types = list(set(
            m.icon_detection.icon_template.legend_item.description
            for m in unmatched_icons
            if m.icon_detection and m.icon_detection.icon_template and m.icon_detection.icon_template.legend_item
        ))

        print(f"   Available icon types: {available_icon_types}")

        stats = {
            "total_unassigned_tags": len(unassigned_tags),
            "tags_matched": 0,
            "api_calls_made": 0,
        }

        used_tags = set()
        used_icon_ids = set()

        # Get matched icons for overlay
        matched_icons = [
            m.icon_detection for m in all_matches
            if m.match_status == "matched" and m.icon_detection
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for tag_det in unassigned_tags:
                tag_name = (
                    tag_det.label_template.legend_item.label_text
                    if tag_det.label_template and tag_det.label_template.legend_item
                    else "Unknown"
                )

                if tag_name in used_tags:
                    continue

                center = (int(tag_det.center[0]), int(tag_det.center[1]))
                print(f"\n      Tag '{tag_name}' at {center}")

                # Download page image
                page_path = os.path.join(tmp_dir, f"page_{tag_det.page_id}.png")
                if not os.path.exists(page_path):
                    self.storage.download_file(tag_det.page.image_url, page_path)

                image = cv2.imread(page_path)
                if image is None:
                    continue

                # Create padded crop
                crop, crop_info = self._create_padded_crop(
                    image, center, padding_stats, is_icon=False
                )

                # Overlay matched icons
                crop = self._overlay_matched_icons(crop, crop_info, matched_icons)

                # Save crop
                crop_path = os.path.join(tmp_dir, f"tag_{tag_det.id}_crop.png")
                cv2.imwrite(crop_path, crop)

                # Get remaining icon types (excluding already used)
                remaining_icon_types = list(set(
                    m.icon_detection.icon_template.legend_item.description
                    for m in unmatched_icons
                    if m.icon_detection
                    and m.icon_detection.icon_template
                    and m.icon_detection.icon_template.legend_item
                    and m.icon_detection.id not in used_icon_ids
                ))

                if not remaining_icon_types:
                    print(f"         No remaining icon types")
                    continue

                prompt = self._create_tag_matching_prompt(tag_name, remaining_icon_types)

                try:
                    result = self.llm._invoke_with_structured_output(
                        prompt, crop_path, TagMatchResult
                    )
                    stats["api_calls_made"] += 1

                    if not result.match_found or not result.matched_icon_type:
                        print(f"         No match found: {result.reasoning}")
                        continue

                    matched_icon_type = result.matched_icon_type
                    if matched_icon_type not in remaining_icon_types:
                        print(f"         Invalid icon type: {matched_icon_type}")
                        continue

                    # Find closest unmatched icon of this type
                    candidate_matches = [
                        m for m in unmatched_icons
                        if m.icon_detection
                        and m.icon_detection.icon_template
                        and m.icon_detection.icon_template.legend_item
                        and m.icon_detection.icon_template.legend_item.description == matched_icon_type
                        and m.icon_detection.id not in used_icon_ids
                    ]

                    if not candidate_matches:
                        print(f"         No available icons of type '{matched_icon_type}'")
                        continue

                    # Find closest
                    tag_center = np.array(center)
                    closest_match = None
                    closest_dist = float("inf")

                    for m in candidate_matches:
                        icon_center = np.array(m.icon_detection.center)
                        dist = np.sqrt(np.sum((icon_center - tag_center) ** 2))
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_match = m

                    if not closest_match:
                        continue

                    # Update match
                    closest_match.label_detection_id = tag_det.id
                    closest_match.distance = float(closest_dist)
                    closest_match.match_confidence = 0.80
                    closest_match.match_method = "llm_icon_for_tag"
                    closest_match.match_status = "matched"
                    self.db.add(closest_match)

                    used_tags.add(tag_name)
                    used_icon_ids.add(closest_match.icon_detection.id)
                    stats["tags_matched"] += 1

                    print(
                        f"         ✅ MATCHED to icon ID={closest_match.icon_detection.id} "
                        f"(conf={result.confidence}, dist={closest_dist:.1f}px)"
                    )

                except Exception as e:
                    print(f"         Error: {e}")
                    stats["api_calls_made"] += 1

        self.db.commit()

        print(f"\n{'='*60}")
        print(f"PHASE 6 COMPLETE")
        print(f"   Tags processed: {stats['total_unassigned_tags']}")
        print(f"   Tags matched: {stats['tags_matched']}")
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
            "api_calls_made": phase5_stats["api_calls_made"] + phase6_stats["api_calls_made"],
        }

        print(f"\n{'='*60}")
        print(f"LLM MATCHING COMPLETE (COMBINED)")
        print(f"   Icons matched: {combined_stats['icons_matched']}")
        print(f"   Icons rejected: {combined_stats['icons_rejected']}")
        print(f"   Tags matched: {combined_stats['tags_matched']}")
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

