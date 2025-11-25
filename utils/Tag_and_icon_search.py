"""
Electrical Symbol Detection System
Detects and counts electrical symbols in layout drawings with scale and rotation invariance.
Updated to allow separate template selection and search images.
Enhanced to support multiple tags and one icon.
Enhanced to support separate scales and rotations for tags and icons.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Detection:
    """Store detection information"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    scale: float
    rotation: float
    center: Tuple[int, int]
    tag_name: str = None  # NEW: Name of the tag/icon
    template_type: str = None  # NEW: 'tag' or 'icon'


class ElectricalSymbolDetector:
    """
    Detects electrical symbols in layout drawings using multi-scale and rotation-invariant matching.
    """
    
    def __init__(self, 
                 scales: List[float] = None,
                 rotations: List[float] = None,
                 tag_scales: List[float] = None,
                 tag_rotations: List[float] = None,
                 icon_scales: List[float] = None,
                 icon_rotations: List[float] = None,
                 threshold: float = 0.7,
                 tag_threshold: float = None,
                 icon_threshold: float = None,
                 nms_threshold: float = 0.3):
        """
        Initialize the detector.
        
        Args:
            scales: List of scale factors to search (default: [0.8, 0.9, 1.0, 1.1, 1.2]) - used if tag/icon scales not specified
            rotations: List of rotation angles in degrees (default: 0 to 360 in 15-degree steps) - used if tag/icon rotations not specified
            tag_scales: Specific scale factors for tag detection (overrides scales)
            tag_rotations: Specific rotation angles for tag detection (overrides rotations)
            icon_scales: Specific scale factors for icon detection (overrides scales)
            icon_rotations: Specific rotation angles for icon detection (overrides rotations)
            threshold: Matching threshold (0-1, higher = more strict) - used if tag/icon thresholds not specified
            tag_threshold: Specific threshold for tag detection (overrides threshold)
            icon_threshold: Specific threshold for icon detection (overrides threshold)
            nms_threshold: Non-maximum suppression IoU threshold
        """
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.rotations = rotations or list(range(0, 360, 15))
        
        # NEW: Separate scales and rotations for tags and icons
        self.tag_scales = tag_scales if tag_scales is not None else self.scales
        self.tag_rotations = tag_rotations if tag_rotations is not None else self.rotations
        self.icon_scales = icon_scales if icon_scales is not None else self.scales
        self.icon_rotations = icon_rotations if icon_rotations is not None else self.rotations
        
        self.threshold = threshold
        self.tag_threshold = tag_threshold if tag_threshold is not None else threshold
        self.icon_threshold = icon_threshold if icon_threshold is not None else threshold
        self.nms_threshold = nms_threshold
        self.template = None
        self.template_gray = None
        self.template_source_image = None  # Store source image path for reference
        
        # NEW: Store multiple templates
        self.templates = []  # List of (template, template_gray, name, type)
        
    def select_template_roi(self, image_path: str, max_display_size: int = 1200) -> np.ndarray:
        """
        Two-step template selection for better accuracy:
        Step 1: User selects rough area around the symbol
        Step 2: Zoomed view shown, user draws tight bounding box
        
        Args:
            image_path: Path to the input image
            max_display_size: Maximum width or height for display (default: 1200 pixels)
            
        Returns:
            template: The selected template image (tightly cropped)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Store source image path
        self.template_source_image = image_path
        
        original_h, original_w = image.shape[:2]
        
        # ============================================================
        # STEP 1: Select rough area around the symbol
        # ============================================================
        print("\n" + "="*70)
        print("STEP 1: SELECT REFERENCE AREA (Rough Selection)")
        print("="*70)
        print("Instructions:")
        print("  - Draw a bounding box around the general area of the symbol")
        print("  - Don't worry about being precise - just capture the symbol area")
        print("  - Press SPACE or ENTER to confirm")
        print("  - Press 'c' to cancel")
        print("="*70 + "\n")
        
        # Calculate display scale to fit screen
        scale_factor = 1.0
        display_image = image.copy()
        
        # Check if image is too large
        if original_h > max_display_size or original_w > max_display_size:
            scale_factor = min(max_display_size / original_w, max_display_size / original_h)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            display_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Image resized for display: {original_w}x{original_h} -> {new_w}x{new_h}")
            print(f"Scale factor: {scale_factor:.3f}\n")
        
        # Step 1: Select rough ROI
        bbox1 = cv2.selectROI("STEP 1: Select Reference Area - Rough Selection", 
                             display_image, 
                             showCrosshair=True, 
                             fromCenter=False)
        cv2.destroyAllWindows()
        
        x1, y1, w1, h1 = bbox1
        
        if w1 == 0 or h1 == 0:
            raise ValueError("No valid ROI selected in Step 1")
        
        # Scale coordinates back to original image size
        x1_orig = int(x1 / scale_factor)
        y1_orig = int(y1 / scale_factor)
        w1_orig = int(w1 / scale_factor)
        h1_orig = int(h1 / scale_factor)
        
        # Ensure coordinates are within bounds
        x1_orig = max(0, min(x1_orig, original_w - 1))
        y1_orig = max(0, min(y1_orig, original_h - 1))
        w1_orig = min(w1_orig, original_w - x1_orig)
        h1_orig = min(h1_orig, original_h - y1_orig)
        
        # Extract rough region from original image
        rough_region = image[y1_orig:y1_orig+h1_orig, x1_orig:x1_orig+w1_orig]
        
        print(f"✓ Step 1 complete!")
        print(f"  Rough area selected: {w1_orig}x{h1_orig} pixels")
        print(f"  Position: ({x1_orig}, {y1_orig})")
        
        # ============================================================
        # STEP 2: Select tight bounding box on zoomed view
        # ============================================================
        print("\n" + "="*70)
        print("STEP 2: REFINE SELECTION (Tight Crop)")
        print("="*70)
        print("Instructions:")
        print("  - You'll see a ZOOMED view of the area you selected")
        print("  - Draw a TIGHT bounding box around ONLY the symbol")
        print("  - Exclude any extra whitespace or background")
        print("  - This precise crop improves detection accuracy")
        print("  - Press SPACE or ENTER to confirm")
        print("  - Press 'c' to cancel and restart")
        print("="*70 + "\n")
        
        # Create a larger display of the rough region for better visibility
        zoom_display = rough_region.copy()
        zoom_h, zoom_w = zoom_display.shape[:2]
        
        # Optionally enlarge small regions for easier selection
        min_display_size = 400
        if zoom_h < min_display_size or zoom_w < min_display_size:
            zoom_factor = min_display_size / min(zoom_h, zoom_w)
            new_zoom_w = int(zoom_w * zoom_factor)
            new_zoom_h = int(zoom_h * zoom_factor)
            zoom_display = cv2.resize(zoom_display, (new_zoom_w, new_zoom_h), 
                                     interpolation=cv2.INTER_CUBIC)
            print(f"Zoomed view enlarged for better visibility: {zoom_w}x{zoom_h} -> {new_zoom_w}x{new_zoom_h}")
            zoom_scale = zoom_factor
        else:
            zoom_scale = 1.0
        
        print(f"Showing closeup view of selected area...")
        print(f"Original rough selection size: {w1_orig}x{h1_orig}")
        
        # Step 2: Select tight ROI on the zoomed view
        bbox2 = cv2.selectROI("STEP 2: Draw Tight Box Around Symbol (Zoomed View)", 
                             zoom_display, 
                             showCrosshair=True, 
                             fromCenter=False)
        cv2.destroyAllWindows()
        
        x2, y2, w2, h2 = bbox2
        
        if w2 == 0 or h2 == 0:
            raise ValueError("No valid ROI selected in Step 2")
        
        # Scale coordinates back to rough region size
        x2_region = int(x2 / zoom_scale)
        y2_region = int(y2 / zoom_scale)
        w2_region = int(w2 / zoom_scale)
        h2_region = int(h2 / zoom_scale)
        
        # Ensure coordinates are within rough region bounds
        x2_region = max(0, min(x2_region, w1_orig - 1))
        y2_region = max(0, min(y2_region, h1_orig - 1))
        w2_region = min(w2_region, w1_orig - x2_region)
        h2_region = min(h2_region, h1_orig - y2_region)
        
        # Calculate final coordinates in original image
        x_final = x1_orig + x2_region
        y_final = y1_orig + y2_region
        w_final = w2_region
        h_final = h2_region
        
        # Extract final tight template from ORIGINAL high-resolution image
        template = image[y_final:y_final+h_final, x_final:x_final+w_final]
        self.template = template.copy()
        self.template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        print(f"\n" + "="*70)
        print("✓ TEMPLATE SELECTION COMPLETE!")
        print("="*70)
        print(f"  Source image: {image_path}")
        print(f"  Original image size: {original_w}x{original_h}")
        print(f"  Final template size: {w_final}x{h_final} pixels")
        print(f"  Final template position: ({x_final}, {y_final})")
        print(f"  Reduction from Step 1: {w1_orig}x{h1_orig} -> {w_final}x{h_final}")
        print(f"  Size reduction: {((1 - (w_final*h_final)/(w1_orig*h1_orig)) * 100):.1f}% smaller")
        print("="*70 + "\n")
        
        return template
    
    # NEW: Collect multiple templates
    def collect_templates(self, image_path: str):
        """
        Collect multiple tag templates and one icon template interactively.
        User enters 'tag' or 'icon' to define templates.
        """
        print("\n" + "="*70)
        print("TEMPLATE COLLECTION MODE")
        print("="*70)
        print("Enter 'tag' to select a tag template (will ask for name)")
        print("Enter 'icon' to select the icon template (ends collection)")
        print("="*70 + "\n")
        
        while True:
            user_input = input("Enter 'tag' or 'icon': ").strip().lower()
            
            if user_input == 'tag':
                tag_name = input("Enter tag name: ").strip()
                if not tag_name:
                    print("Tag name cannot be empty!")
                    continue
                
                print(f"\nSelecting TAG template: '{tag_name}'")
                template = self.select_template_roi(image_path)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                self.templates.append((template, template_gray, tag_name, 'tag'))
                print(f"✓ Tag '{tag_name}' added. Total tags: {len([t for t in self.templates if t[3] == 'tag'])}\n")
                
            elif user_input == 'icon':
                icon_name = input("Enter icon name (optional, press Enter for 'icon'): ").strip()
                if not icon_name:
                    icon_name = 'icon'
                
                print(f"\nSelecting ICON template: '{icon_name}'")
                template = self.select_template_roi(image_path)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                self.templates.append((template, template_gray, icon_name, 'icon'))
                print(f"✓ Icon '{icon_name}' added.")
                break
            else:
                print("Invalid input. Please enter 'tag' or 'icon'.")
        
        print("\n" + "="*70)
        print("TEMPLATE COLLECTION SUMMARY")
        print("="*70)
        tags = [t for t in self.templates if t[3] == 'tag']
        icons = [t for t in self.templates if t[3] == 'icon']
        print(f"Tags collected: {len(tags)}")
        for t in tags:
            print(f"  - {t[2]}")
        print(f"Icons collected: {len(icons)}")
        for t in icons:
            print(f"  - {t[2]}")
        print("="*70 + "\n")
    
    def rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """
        Rotate image by given angle with proper scaling.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            rotated_image: Rotated image
            scale: Scale factor after rotation
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        return rotated, 1.0
    
    def template_match(self, 
                      image_gray: np.ndarray, 
                      template: np.ndarray,
                      method: int = cv2.TM_CCOEFF_NORMED) -> np.ndarray:
        """
        Perform template matching.
        
        Args:
            image_gray: Grayscale image to search in
            template: Grayscale template to search for
            method: OpenCV template matching method
            
        Returns:
            result: Matching result matrix
        """
        if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            return None
        
        result = cv2.matchTemplate(image_gray, template, method)
        return result
    
    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            filtered_detections: List of Detection objects after NMS
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        
        while len(detections) > 0:
            # Keep the detection with highest confidence
            current = detections[0]
            keep.append(current)
            detections = detections[1:]
            
            # Remove detections with high IoU
            filtered = []
            for det in detections:
                iou = self.calculate_iou(current.bbox, det.bbox)
                if iou < self.nms_threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            iou: Intersection over Union value
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def detect_symbols(self, search_image_path: str, visualize: bool = True) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect all instances of the template symbol in the specified search image.
        
        Args:
            search_image_path: Path to the image to search in (can be different from template source)
            visualize: Whether to create visualization
            
        Returns:
            detections: List of Detection objects
            result_image: Image with drawn detections
        """
        # NEW: Support multiple templates
        if len(self.templates) == 0 and self.template is None:
            raise ValueError("Template not selected. Call select_template_roi() or collect_templates() first.")
        
        # Read search image
        image = cv2.imread(search_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {search_image_path}")
        
        print(f"\n{'='*70}")
        print(f"SEARCHING IN IMAGE")
        print(f"{'='*70}")
        print(f"  Template source: {self.template_source_image}")
        print(f"  Search image: {search_image_path}")
        print(f"  Search image size: {image.shape[1]}x{image.shape[0]}")
        
        # NEW: Handle multiple templates
        if len(self.templates) > 0:
            print(f"  Templates to search: {len(self.templates)}")
            for template, _, name, ttype in self.templates:
                print(f"    - {ttype}: '{name}' ({template.shape[1]}x{template.shape[0]})")
        else:
            print(f"  Template size: {self.template.shape[1]}x{self.template.shape[0]}")
        
        # NEW: Display scale and rotation parameters
        print(f"\n  Tag scales: {self.tag_scales}")
        print(f"  Tag rotations: {self.tag_rotations}")
        print(f"  Icon scales: {self.icon_scales}")
        print(f"  Icon rotations: {self.icon_rotations}")
        print(f"{'='*70}\n")
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_gray.shape
        
        all_detections = []
        
        # NEW: Process multiple templates
        templates_to_process = []
        if len(self.templates) > 0:
            templates_to_process = self.templates
        else:
            templates_to_process = [(self.template, self.template_gray, 'symbol', 'icon')]
        
        for template, template_gray, name, ttype in templates_to_process:
            # NEW: Use specific threshold, scales, and rotations for tag/icon
            current_threshold = self.tag_threshold if ttype == 'tag' else self.icon_threshold
            current_scales = self.tag_scales if ttype == 'tag' else self.icon_scales
            current_rotations = self.tag_rotations if ttype == 'tag' else self.icon_rotations
            
            print(f"Searching for {ttype} '{name}'")
            print(f"  Threshold: {current_threshold:.2f}")
            print(f"  Scales: {current_scales}")
            print(f"  Rotations: {current_rotations}")
            
            template_detections = []
            
            total_iterations = len(current_scales) * len(current_rotations)
            current_iteration = 0
            
            # Search across scales and rotations
            for scale in current_scales:
                for rotation in current_rotations:
                    current_iteration += 1
                    
                    # Progress indicator
                    if current_iteration % 10 == 0:
                        print(f"  Progress: {current_iteration}/{total_iterations}")
                    
                    # Resize and rotate template
                    h_t, w_t = template_gray.shape
                    new_w = int(w_t * scale)
                    new_h = int(h_t * scale)
                    
                    if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
                        continue
                    
                    # Resize template
                    resized_template = cv2.resize(template_gray, (new_w, new_h))
                    
                    # Rotate template
                    rotated_template, _ = self.rotate_image(resized_template, rotation)
                    
                    # Skip if template is too large
                    if rotated_template.shape[0] > h_img or rotated_template.shape[1] > w_img:
                        continue
                    
                    # Perform template matching
                    result = self.template_match(image_gray, rotated_template)
                    
                    if result is None:
                        continue
                    
                    # Find matches above threshold (use specific threshold for tag/icon)
                    locations = np.where(result >= current_threshold)
                    
                    for pt in zip(*locations[::-1]):
                        x, y = pt
                        w, h = rotated_template.shape[1], rotated_template.shape[0]
                        confidence = result[y, x]
                        center = (x + w // 2, y + h // 2)
                        
                        detection = Detection(
                            bbox=(x, y, w, h),
                            confidence=float(confidence),
                            scale=scale,
                            rotation=rotation,
                            center=center,
                            tag_name=name,  # NEW
                            template_type=ttype  # NEW
                        )
                        template_detections.append(detection)
            
            print(f"  Found {len(template_detections)} raw detections for '{name}'")
            
            # Apply NMS per template
            filtered = self.non_max_suppression(template_detections)
            print(f"  After NMS: {len(filtered)} detections for '{name}'")
            
            all_detections.extend(filtered)
        
        print(f"\nTotal detections across all templates: {len(all_detections)}")
        
        # Create visualization
        result_image = image.copy()
        if visualize:
            result_image = self.draw_detections(result_image, all_detections)
        
        return all_detections, result_image
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of Detection objects
            
        Returns:
            image_with_boxes: Image with drawn bounding boxes
        """
        result = image.copy()
        
        # Calculate adaptive line thickness based on image size
        img_height, img_width = image.shape[:2]
        line_thickness = max(2, int((img_height + img_width) / 1000))
        
        # NEW: Separate counters for tags and icons
        tag_counter = {}
        icon_counter = 1
        
        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox
            
            # NEW: Different colors for tags vs icons
            if det.template_type == 'tag':
                color = (255, 0, 0)  # Blue for tags
                if det.tag_name not in tag_counter:
                    tag_counter[det.tag_name] = 0
                tag_counter[det.tag_name] += 1
                label = f"{det.tag_name}-{tag_counter[det.tag_name]} ({det.confidence:.2f})"
            else:
                color = (0, 255, 0)  # Green for icons
                label = f"{det.tag_name or 'icon'}-{icon_counter} ({det.confidence:.2f})"
                icon_counter += 1
            
            overlay = result.copy()
            alpha = 0.35

            # Draw bounding box on overlay
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, line_thickness)

            # Blend overlay with original result
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

            # Prepare dynamic label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = max(0.25, line_thickness // 8)

            # Measure text size
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Position of the text background
            text_x = x
            text_y = y - text_h - baseline - 5

            # Make sure text background doesn't go above image boundary
            if text_y < 0:
                text_y = y + h + text_h + baseline + 5

            # Draw semi-transparent text background
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y),
                (text_x + text_w + 4, text_y + text_h + baseline + 4),
                color,
                -1
            )
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

            # Draw text
            cv2.putText(
                result,
                label,
                (text_x + 2, text_y + text_h + 2),
                font,
                font_scale,
                (0,0,0),
                font_thickness
            )
        
        return result
    
    def save_result_image(self, 
                         search_image_path: str, 
                         result_image: np.ndarray, 
                         output_folder: str = None,
                         output_filename: str = None) -> str:
        """
        Save the result image with bounding boxes to an output folder.
        
        Args:
            search_image_path: Path to the search image (where detections were made)
            result_image: Image with drawn detections
            output_folder: Custom output folder (default: 'output' in same directory)
            output_filename: Custom output filename (default: original_name_detected.ext)
            
        Returns:
            output_path: Path where the image was saved
        """
        import os
        
        # Create output folder
        if output_folder is None:
            image_dir = os.path.dirname(search_image_path) or '.'
            output_folder = os.path.join(image_dir, 'output')
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate output filename
        if output_filename is None:
            image_filename = os.path.basename(search_image_path)
            image_name, image_ext = os.path.splitext(image_filename)
            output_filename = f"{image_name}_detected{image_ext}"
        
        output_path = os.path.join(output_folder, output_filename)
        
        # Ensure result image has same dimensions as original
        original_img = cv2.imread(search_image_path)
        if result_image.shape != original_img.shape:
            print(f"  Resizing result to match original dimensions: {original_img.shape[1]}x{original_img.shape[0]}")
            result_image = cv2.resize(result_image, (original_img.shape[1], original_img.shape[0]))
        
        # Save the result
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"✓ Result image saved successfully!")
            print(f"  Output folder: {output_folder}")
            print(f"  Filename: {output_filename}")
            print(f"  Full path: {output_path}")
            print(f"  Dimensions: {result_image.shape[1]}x{result_image.shape[0]} pixels")
        else:
            print(f"✗ Failed to save image to: {output_path}")
        
        return output_path
    
    def get_detection_summary(self, detections: List[Detection]) -> Dict:
        """
        Get summary of detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            summary: Dictionary with detection statistics
        """
        if len(detections) == 0:
            return {
                'count': 0,
                'bboxes': [],
                'centers': [],
                'confidences': [],
                'scales': [],
                'rotations': [],
                'tags': [],  # NEW
                'template_types': []  # NEW
            }
        
        summary = {
            'count': len(detections),
            'bboxes': [det.bbox for det in detections],
            'centers': [det.center for det in detections],
            'confidences': [det.confidence for det in detections],
            'scales': [det.scale for det in detections],
            'rotations': [det.rotation for det in detections],
            'tags': [det.tag_name for det in detections],  # NEW
            'template_types': [det.template_type for det in detections],  # NEW
            'avg_confidence': np.mean([det.confidence for det in detections]),
            'min_confidence': np.min([det.confidence for det in detections]),
            'max_confidence': np.max([det.confidence for det in detections])
        }
        
        # NEW: Group by tag/icon
        summary['by_tag'] = {}
        for det in detections:
            tag = det.tag_name or 'unknown'
            if tag not in summary['by_tag']:
                summary['by_tag'][tag] = []
            summary['by_tag'][tag].append(det)
        
        return summary


def run_detection_pipeline(template_image_path: str,
                          search_image_path: str = None,
                          scales: List[float] = None,
                          rotations: List[float] = None,
                          threshold: float = 0.7,
                          tag_threshold: float = None,
                          icon_threshold: float = None,
                          tag_scales: List[float] = None,
                          icon_scales: List[float] = None,
                          tag_rotations: List[float] = None,
                          icon_rotations: List[float] = None,
                          nms_threshold: float = 0.3,
                          save_output: bool = True,
                          output_path: str = None,
                          max_display_size: int = 1200) -> Dict:
    """
    Run the complete detection pipeline with separate template and search images.
    
    Args:
        template_image_path: Path to the image for template selection
        search_image_path: Path to the image to search in (if None, user will be prompted)
        scales: List of scale factors to search
        rotations: List of rotation angles in degrees
        threshold: Matching threshold (0-1) - used if tag/icon thresholds not specified
        tag_threshold: Specific threshold for tag detection (default: uses threshold)
        icon_threshold: Specific threshold for icon detection (default: uses threshold)
        nms_threshold: Non-maximum suppression IoU threshold
        save_output: Whether to save the output image
        output_path: Path to save the output image (default: search_image_path_result.jpg)
        max_display_size: Maximum width/height for display window (default: 1200 pixels)
        
    Returns:
        results: Dictionary containing detections and summary
    """
    # Initialize detector
    detector = ElectricalSymbolDetector(
        scales=scales,
        rotations=rotations,
        threshold=threshold,
        tag_threshold = tag_threshold,
        icon_threshold = icon_threshold,
        tag_scales = tag_scales,
        icon_scales = icon_scales,
        tag_rotations=tag_rotations,
        icon_rotations=icon_rotations,
        nms_threshold=nms_threshold
    )
    
    # Step 1: Select template from template image
    print("="*70)
    print("STEP 1: SELECT TEMPLATE SYMBOL")
    print("="*70)
        
    # Collect templates
    detector.collect_templates(template_image_path)
    
    # Determine search image
    if search_image_path is None:
        search_image_path = template_image_path
    
    # Detect all symbols (returns tuple: detections list and result_image)
    all_detections, result_image = detector.detect_symbols(search_image_path)
    
    # Save results if requested
    if save_output:
        detector.save_result_image(search_image_path, result_image, output_folder=output_path)
    
    # Separate detections by type
    tag_detections = [det for det in all_detections if det.template_type == 'tag']
    icon_detections = [det for det in all_detections if det.template_type == 'icon']
    
    # Display summary
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"\nTags detected: {len(tag_detections)}")
    for det in tag_detections:
        print(f"  - {det.tag_name}: confidence {det.confidence:.3f} at {det.center}")
    
    print(f"\nIcons detected: {len(icon_detections)}")
    for det in icon_detections:
        print(f"  - {det.tag_name or 'icon'}: confidence {det.confidence:.3f} at {det.center}")
    print("="*70 + "\n")
    
    return {
        'tag_detections': tag_detections,
        'icon_detections': icon_detections,
        'all_detections': all_detections,
        'result_image': result_image,
        'detector': detector
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Enhanced Electrical Symbol Detector")
        print("="*70)
        print("\nUsage: python Tag_and_icon_search.py <reference_image> [search_image] [options]")
        print("\nArguments:")
        print("  reference_image  Image to select templates from")
        print("  search_image     (Optional) Image to search in")
        print("\nWorkflow:")
        print("  1. You'll be prompted to define TAG templates first")
        print("  2. For each tag, enter a name and crop the symbol")
        print("  3. Enter 'icon' when finished with tags to define icon template")
        print("  4. System searches for all templates and generates report")
        print("\nOptional Arguments:")
        print("  --threshold <value>       Base threshold for both tags and icons (default: 0.7)")
        print("  --tag-threshold <value>   Specific threshold for tags (overrides --threshold)")
        print("  --icon-threshold <value>  Specific threshold for icons (overrides --threshold)")
        print("  --nms-threshold <value>   NMS IoU threshold (default: 0.3)")
        print("\nExamples:")
        print("  python enhanced_detector.py layout.png")
        print("  python enhanced_detector.py reference.png floor_plan.png")
        print("  python enhanced_detector.py ref.png search.png --threshold 0.8")
        print("  python enhanced_detector.py ref.png search.png --tag-threshold 0.7 --icon-threshold 0.8")
        print("="*70)
        sys.exit(1)
    
    template_image_path = sys.argv[1]
    search_image_path = None
    
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        search_image_path = sys.argv[2]
        arg_start = 3
    else:
        arg_start = 2
    
    # Parse optional arguments
    threshold = 0.7
    tag_threshold = None
    icon_threshold = None
    nms_threshold = 0.3
    
    i = arg_start
    while i < len(sys.argv):
        if sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--tag-threshold' and i + 1 < len(sys.argv):
            tag_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--icon-threshold' and i + 1 < len(sys.argv):
            icon_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--nms-threshold' and i + 1 < len(sys.argv):
            nms_threshold = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    print("\n" + "="*70)
    print("ENHANCED ELECTRICAL SYMBOL DETECTOR")
    print("="*70)
    print(f"Reference image: {template_image_path}")
    if search_image_path:
        print(f"Search image: {search_image_path}")
    else:
        print(f"Search image: Same as reference")
    print(f"Base Threshold: {threshold}")
    if tag_threshold is not None:
        print(f"Tag Threshold: {tag_threshold}")
    if icon_threshold is not None:
        print(f"Icon Threshold: {icon_threshold}")
    print(f"NMS Threshold: {nms_threshold}")
    print("="*70)
    
    # Run detection
    results = run_detection_pipeline(
        template_image_path=template_image_path,
        search_image_path=search_image_path,
        threshold=threshold,
        tag_threshold=tag_threshold,
        icon_threshold=icon_threshold,
        nms_threshold=nms_threshold
    )
    
    # Display results
    # Count templates by type
    tag_templates = [t for t in results['detector'].templates if t[3] == 'tag']
    icon_templates = [t for t in results['detector'].templates if t[3] == 'icon']
    num_templates = len(tag_templates) + len(icon_templates)
    
    if num_templates > 0:
        fig, axes = plt.subplots(1, num_templates + 1, 
                                 figsize=(5 * (num_templates + 1), 5))
        axes = axes.flatten() if num_templates > 1 else [axes[0], axes[1]]
        
        # Show tag templates
        idx = 0
        for template, _, name, ttype in tag_templates:
            tag_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(tag_rgb)
            axes[idx].set_title(f"Tag Template: {name}")
            axes[idx].axis('off')
            idx += 1
        
        # Show icon templates
        for template, _, name, ttype in icon_templates:
            icon_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(icon_rgb)
            axes[idx].set_title(f"Icon Template: {name}")
            axes[idx].axis('off')
            idx += 1
        
        # Show result
        result_rgb = cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB)
        axes[-1].imshow(result_rgb)
        axes[-1].set_title(
            f"Detections (Tags: {len(results['tag_detections'])}, Icons: {len(results['icon_detections'])})"
        )
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # If no templates, just show the result
        result_rgb = cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(result_rgb)
        plt.title(f"Detections (Tags: {len(results['tag_detections'])}, Icons: {len(results['icon_detections'])})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    print("\n✓ Detection complete! Check the 'output' folder for detailed results.")