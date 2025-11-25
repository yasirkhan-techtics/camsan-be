"""
Electrical Symbol Detection System
Detects and counts electrical symbols in layout drawings with multiple matching methods.
Uses sliding window approach for feature-based methods to handle small symbols in large images.
Memory-efficient implementation.
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


class ElectricalSymbolDetector:
    """
    Detects electrical symbols in layout drawings using multiple matching methods.
    Methods: template_matching, sift, orb, akaze, brisk
    Uses sliding window for feature-based methods.
    """
    
    def __init__(self, 
                 scales: List[float] = None,
                 rotations: List[float] = None,
                 threshold: float = 0.7,
                 nms_threshold: float = 0.3,
                 method: str = 'template_matching',
                 window_scale: float = 2.5):
        """
        Initialize the detector.
        
        Args:
            scales: List of scale factors (only for template_matching)
            rotations: List of rotation angles (only for template_matching)
            threshold: Matching threshold
            nms_threshold: Non-maximum suppression IoU threshold
            method: Detection method - 'template_matching', 'sift', 'orb', 'akaze', 'brisk'
            window_scale: Sliding window size as multiple of template size (default: 2.5x)
        """
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.rotations = rotations or list(range(0, 360, 15))
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.method = method.lower()
        self.window_scale = window_scale
        self.template = None
        self.template_gray = None
        self.template_source_image = None
        
        # Validate method
        valid_methods = ['template_matching', 'sift', 'orb', 'akaze', 'brisk']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Choose from: {valid_methods}")
        
        # Initialize feature detector based on method
        self.feature_detector = None
        if self.method in ['sift', 'orb', 'akaze', 'brisk']:
            self._initialize_feature_detector()
    
    def _initialize_feature_detector(self):
        """Initialize the feature detector based on selected method"""
        try:
            if self.method == 'sift':
                # Reduce memory usage by limiting features
                self.feature_detector = cv2.SIFT_create(nfeatures=500)
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                print("✓ SIFT detector initialized (memory-efficient mode)")
            
            elif self.method == 'orb':
                # Increase features for better detection
                self.feature_detector = cv2.ORB_create(nfeatures=2000, 
                                                       scaleFactor=1.2,
                                                       nlevels=8,
                                                       edgeThreshold=15,
                                                       patchSize=31)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                print("✓ ORB detector initialized")
            
            elif self.method == 'akaze':
                self.feature_detector = cv2.AKAZE_create(threshold=0.001)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                print("✓ AKAZE detector initialized")
            
            elif self.method == 'brisk':
                self.feature_detector = cv2.BRISK_create(thresh=30, octaves=3)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                print("✓ BRISK detector initialized")
                
        except Exception as e:
            raise ValueError(f"Error initializing {self.method.upper()} detector: {str(e)}")
    
    def select_template_roi(self, image_path: str, max_display_size: int = 1200) -> np.ndarray:
        """Two-step template selection for better accuracy."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.template_source_image = image_path
        original_h, original_w = image.shape[:2]
        
        # STEP 1: Select rough area
        print("\n" + "="*70)
        print("STEP 1: SELECT REFERENCE AREA (Rough Selection)")
        print("="*70)
        print("Instructions:")
        print("  - Draw a bounding box around the general area of the symbol")
        print("  - Press SPACE or ENTER to confirm")
        print("="*70 + "\n")
        
        scale_factor = 1.0
        display_image = image.copy()
        
        if original_h > max_display_size or original_w > max_display_size:
            scale_factor = min(max_display_size / original_w, max_display_size / original_h)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            display_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        bbox1 = cv2.selectROI("STEP 1: Select Reference Area", 
                             display_image, 
                             showCrosshair=True, 
                             fromCenter=False)
        cv2.destroyAllWindows()
        
        x1, y1, w1, h1 = bbox1
        if w1 == 0 or h1 == 0:
            raise ValueError("No valid ROI selected in Step 1")
        
        x1_orig = int(x1 / scale_factor)
        y1_orig = int(y1 / scale_factor)
        w1_orig = int(w1 / scale_factor)
        h1_orig = int(h1 / scale_factor)
        
        rough_region = image[y1_orig:y1_orig+h1_orig, x1_orig:x1_orig+w1_orig]
        
        # STEP 2: Select tight bounding box
        print("\n" + "="*70)
        print("STEP 2: REFINE SELECTION (Tight Crop)")
        print("="*70)
        print("Instructions:")
        print("  - Draw a TIGHT bounding box around ONLY the symbol")
        print("  - Press SPACE or ENTER to confirm")
        print("="*70 + "\n")
        
        zoom_display = rough_region.copy()
        zoom_h, zoom_w = zoom_display.shape[:2]
        
        min_display_size = 400
        if zoom_h < min_display_size or zoom_w < min_display_size:
            zoom_factor = min_display_size / min(zoom_h, zoom_w)
            new_zoom_w = int(zoom_w * zoom_factor)
            new_zoom_h = int(zoom_h * zoom_factor)
            zoom_display = cv2.resize(zoom_display, (new_zoom_w, new_zoom_h), 
                                     interpolation=cv2.INTER_CUBIC)
            zoom_scale = zoom_factor
        else:
            zoom_scale = 1.0
        
        bbox2 = cv2.selectROI("STEP 2: Draw Tight Box Around Symbol", 
                             zoom_display, 
                             showCrosshair=True, 
                             fromCenter=False)
        cv2.destroyAllWindows()
        
        x2, y2, w2, h2 = bbox2
        if w2 == 0 or h2 == 0:
            raise ValueError("No valid ROI selected in Step 2")
        
        x2_region = int(x2 / zoom_scale)
        y2_region = int(y2 / zoom_scale)
        w2_region = int(w2 / zoom_scale)
        h2_region = int(h2 / zoom_scale)
        
        x_final = x1_orig + x2_region
        y_final = y1_orig + y2_region
        w_final = w2_region
        h_final = h2_region
        
        template = image[y_final:y_final+h_final, x_final:x_final+w_final]
        self.template = template.copy()
        self.template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Extract features from template for feature-based methods
        if self.method in ['sift', 'orb', 'akaze', 'brisk']:
            self.template_keypoints, self.template_descriptors = \
                self.feature_detector.detectAndCompute(self.template_gray, None)
            
            if self.template_descriptors is None or len(self.template_keypoints) < 4:
                print(f"⚠️  WARNING: Only {len(self.template_keypoints) if self.template_keypoints else 0} keypoints found in template!")
                print(f"   {self.method.upper()} may not work well. Consider using template_matching instead.")
            else:
                print(f"✓ Extracted {len(self.template_keypoints)} keypoints from template")
        
        print(f"\n{'='*70}")
        print("✓ TEMPLATE SELECTION COMPLETE!")
        print("="*70)
        print(f"  Method: {self.method.upper()}")
        print(f"  Template size: {w_final}x{h_final} pixels")
        print("="*70 + "\n")
        
        return template
    
    def rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return rotated, 1.0
    
    def template_match(self, image_gray: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Perform template matching using TM_CCOEFF_NORMED"""
        if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            return None
        result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        return result
    
    def feature_match_window(self, window_gray: np.ndarray, window_offset: Tuple[int, int]) -> List[Detection]:
        """
        Perform feature-based matching in a sliding window.
        Returns ALL detections without filtering (filtering done later with NMS).
        
        Args:
            window_gray: Grayscale window to search in
            window_offset: (x_offset, y_offset) position of window in full image
            
        Returns:
            detections: List of Detection objects (unfiltered)
        """
        detections = []
        
        # Check if window is too small
        if window_gray.shape[0] < self.template_gray.shape[0] or \
           window_gray.shape[1] < self.template_gray.shape[1]:
            return detections
        
        # Detect features in window
        try:
            keypoints, descriptors = self.feature_detector.detectAndCompute(window_gray, None)
        except Exception as e:
            return detections
        
        if descriptors is None or len(keypoints) < 4:
            return detections
        
        # Limit descriptors to prevent BRISK error
        max_descriptors = 50000  # Prevent overflow
        if len(descriptors) > max_descriptors:
            # Sample descriptors uniformly
            indices = np.linspace(0, len(descriptors)-1, max_descriptors, dtype=int)
            descriptors = descriptors[indices]
            keypoints = [keypoints[i] for i in indices]
        
        # Match features
        try:
            matches = self.matcher.knnMatch(self.template_descriptors, descriptors, k=2)
        except cv2.error as e:
            # Handle BRISK/descriptor overflow
            return detections
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Need at least 4 matches for homography
        if len(good_matches) < 4:
            return detections
        
        # Extract matched keypoints
        src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return detections
        
        # Calculate inlier ratio
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)
        
        # No minimum threshold here - return all detections for later filtering
        
        # Get template corners
        h, w = self.template_gray.shape
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to window coordinates
        try:
            transformed_corners = cv2.perspectiveTransform(template_corners, M)
            
            # Calculate bounding box in window coordinates
            x_coords = transformed_corners[:, 0, 0]
            y_coords = transformed_corners[:, 0, 1]
            
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            # Basic validation only (no strict filtering)
            if bbox_w < 5 or bbox_h < 5 or bbox_w > window_gray.shape[1] * 2 or bbox_h > window_gray.shape[0] * 2:
                return detections
            
            # Convert to full image coordinates
            x_offset, y_offset = window_offset
            x_min_global = x_min + x_offset
            y_min_global = y_min + y_offset
            
            # Calculate confidence (combination of inlier ratio and number of matches)
            confidence = inlier_ratio * min(1.0, len(good_matches) / 20.0)
            
            # Calculate scale and rotation
            scale = np.sqrt(bbox_w * bbox_h) / np.sqrt(w * h)
            rotation = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            
            center = (x_min_global + bbox_w // 2, y_min_global + bbox_h // 2)
            
            detection = Detection(
                bbox=(x_min_global, y_min_global, bbox_w, bbox_h),
                confidence=float(confidence),
                scale=float(scale),
                rotation=float(rotation),
                center=center
            )
            detections.append(detection)
            
        except cv2.error:
            pass
        
        return detections
    
    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        while len(detections) > 0:
            current = detections[0]
            keep.append(current)
            detections = detections[1:]
            
            filtered = []
            for det in detections:
                iou = self.calculate_iou(current.bbox, det.bbox)
                if iou < self.nms_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def detect_symbols(self, search_image_path: str, visualize: bool = True) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect all instances of the template symbol using selected method.
        Uses sliding window for feature-based methods.
        """
        if self.template is None:
            raise ValueError("Template not selected. Call select_template_roi() first.")
        
        image = cv2.imread(search_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {search_image_path}")
        
        print(f"\n{'='*70}")
        print(f"SEARCHING USING {self.method.upper()}")
        print(f"{'='*70}")
        print(f"  Search image: {search_image_path}")
        print(f"  Search image size: {image.shape[1]}x{image.shape[0]}")
        print(f"  Template size: {self.template.shape[1]}x{self.template.shape[0]}")
        print(f"{'='*70}\n")
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_gray.shape
        
        all_detections = []
        
        if self.method == 'template_matching':
            # Traditional template matching with scale and rotation
            print("Searching across scales and rotations...")
            total_iterations = len(self.scales) * len(self.rotations)
            current_iteration = 0
            
            for scale in self.scales:
                for rotation in self.rotations:
                    current_iteration += 1
                    
                    if current_iteration % 10 == 0:
                        print(f"Progress: {current_iteration}/{total_iterations}")
                    
                    h_t, w_t = self.template_gray.shape
                    new_w = int(w_t * scale)
                    new_h = int(h_t * scale)
                    
                    if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
                        continue
                    
                    resized_template = cv2.resize(self.template_gray, (new_w, new_h))
                    rotated_template, _ = self.rotate_image(resized_template, rotation)
                    
                    if rotated_template.shape[0] > h_img or rotated_template.shape[1] > w_img:
                        continue
                    
                    result = self.template_match(image_gray, rotated_template)
                    
                    if result is None:
                        continue
                    
                    locations = np.where(result >= self.threshold)
                    
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
                            center=center
                        )
                        all_detections.append(detection)
        
        else:
            # Feature-based matching with sliding window
            print(f"Performing {self.method.upper()} feature matching with sliding window...")
            
            # Calculate window size (larger than template to allow for scale/rotation)
            h_t, w_t = self.template_gray.shape
            window_h = int(h_t * self.window_scale)
            window_w = int(w_t * self.window_scale)
            
            # Calculate stride (50% overlap)
            stride_h = window_h // 2
            stride_w = window_w // 2
            
            print(f"  Window size: {window_w}x{window_h}")
            print(f"  Stride: {stride_w}x{stride_h}")
            
            # Calculate total windows
            num_windows_h = (h_img - window_h) // stride_h + 1
            num_windows_w = (w_img - window_w) // stride_w + 1
            total_windows = num_windows_h * num_windows_w
            
            print(f"  Total windows to process: {total_windows}")
            
            window_count = 0
            
            # Slide window across image
            for y in range(0, h_img - window_h + 1, stride_h):
                for x in range(0, w_img - window_w + 1, stride_w):
                    window_count += 1
                    
                    if window_count % 50 == 0:
                        print(f"  Progress: {window_count}/{total_windows} windows ({100*window_count/total_windows:.1f}%)")
                    
                    # Extract window
                    window = image_gray[y:y+window_h, x:x+window_w]
                    
                    # Perform feature matching in window
                    window_detections = self.feature_match_window(window, (x, y))
                    
                    all_detections.extend(window_detections)
            
            print(f"  Completed: {window_count}/{total_windows} windows")
        
        print(f"Found {len(all_detections)} raw detections")
        
        # Filter by confidence threshold BEFORE NMS (for all methods)
        print(f"Applying confidence threshold: {self.threshold}")
        threshold_filtered = [det for det in all_detections if det.confidence >= self.threshold]
        print(f"After threshold filter: {len(threshold_filtered)} detections")
        
        # Apply NMS with IoU filtering (for all methods)
        filtered_detections = self.non_max_suppression(threshold_filtered)
        print(f"After NMS (IoU < {self.nms_threshold}): {len(filtered_detections)} detections")
        
        # Create visualization
        result_image = image.copy()
        if visualize:
            result_image = self.draw_detections(result_image, filtered_detections)
        
        return filtered_detections, result_image
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on image"""
        result = image.copy()
        img_height, img_width = image.shape[:2]
        line_thickness = max(2, int((img_height + img_width) / 1000))
        
        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox
            
            overlay = result.copy()
            color = (0, 255, 0)
            alpha = 0.35
            
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, line_thickness)
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            label = f"#{idx+1} ({det.confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = max(1, line_thickness // 2)
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            while text_w > w - 6 and font_scale > 0.3:
                font_scale -= 0.02
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            text_x = x
            text_y = y - text_h - baseline - 5
            
            if text_y < 0:
                text_y = y + h + text_h + baseline + 5
            
            overlay = result.copy()
            cv2.rectangle(overlay, (text_x, text_y),
                         (text_x + text_w + 4, text_y + text_h + baseline + 4),
                         color, -1)
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            cv2.putText(result, label, (text_x + 2, text_y + text_h + 2),
                       font, font_scale, (0, 0, 0), font_thickness)
        
        return result
    
    def save_result_image(self, search_image_path: str, result_image: np.ndarray, 
                         output_folder: str = None, output_filename: str = None) -> str:
        """Save the result image"""
        import os
        
        if output_folder is None:
            image_dir = os.path.dirname(search_image_path) or '.'
            output_folder = os.path.join(image_dir, 'output')
        
        os.makedirs(output_folder, exist_ok=True)
        
        if output_filename is None:
            image_filename = os.path.basename(search_image_path)
            image_name, image_ext = os.path.splitext(image_filename)
            output_filename = f"{image_name}_detected_{self.method}{image_ext}"
        
        output_path = os.path.join(output_folder, output_filename)
        
        original_img = cv2.imread(search_image_path)
        if result_image.shape != original_img.shape:
            result_image = cv2.resize(result_image, (original_img.shape[1], original_img.shape[0]))
        
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"✓ Result saved: {output_path}")
        else:
            print(f"✗ Failed to save: {output_path}")
        
        return output_path
    
    def get_detection_summary(self, detections: List[Detection]) -> Dict:
        """Get summary of detections"""
        if len(detections) == 0:
            return {
                'count': 0,
                'bboxes': [],
                'centers': [],
                'confidences': [],
                'scales': [],
                'rotations': []
            }
        
        summary = {
            'count': len(detections),
            'bboxes': [det.bbox for det in detections],
            'centers': [det.center for det in detections],
            'confidences': [det.confidence for det in detections],
            'scales': [det.scale for det in detections],
            'rotations': [det.rotation for det in detections],
            'avg_confidence': np.mean([det.confidence for det in detections]),
            'min_confidence': np.min([det.confidence for det in detections]),
            'max_confidence': np.max([det.confidence for det in detections])
        }
        
        return summary


def run_detection_pipeline(template_image_path: str,
                          search_image_path: str = None,
                          method: str = 'template_matching',
                          scales: List[float] = None,
                          rotations: List[float] = None,
                          threshold: float = 0.7,
                          nms_threshold: float = 0.3,
                          window_scale: float = 2.5,
                          save_output: bool = True,
                          output_path: str = None,
                          max_display_size: int = 1200) -> Dict:
    """
    Run the complete detection pipeline with selected matching method.
    
    Args:
        template_image_path: Path to the image for template selection
        search_image_path: Path to the image to search in
        method: Matching method - 'template_matching', 'sift', 'orb', 'akaze', 'brisk'
        scales: List of scale factors (only for template_matching)
        rotations: List of rotation angles (only for template_matching)
        threshold: Matching threshold (0-1 for template_matching, 0.3-0.7 for features)
        nms_threshold: Non-maximum suppression IoU threshold
        window_scale: Window size multiplier for feature methods (default: 2.5x template)
        save_output: Whether to save the output image
        output_path: Path to save the output image
        max_display_size: Maximum width/height for display window
        
    Returns:
        results: Dictionary containing detections and summary
    
    Examples:
        # Template matching
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='template_matching', threshold=0.8)
        
        # ORB with sliding window (good for small symbols)
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='orb', threshold=0.4, window_scale=3.0)
        
        # AKAZE with custom window size
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='akaze', threshold=0.5, window_scale=2.5)
    """
    # Initialize detector with selected method
    detector = ElectricalSymbolDetector(
        scales=scales,
        rotations=rotations,
        threshold=threshold,
        nms_threshold=nms_threshold,
        method=method,
        window_scale=window_scale
    )
    
    # Step 1: Select template
    print("="*70)
    print(f"DETECTION METHOD: {method.upper()}")
    if method != 'template_matching':
        print(f"Using sliding window approach (window_scale={window_scale}x)")
    print("="*70)
    template = detector.select_template_roi(template_image_path, max_display_size=max_display_size)
    
    # Step 2: Get search image
    if search_image_path is None:
        print("\n" + "="*70)
        print("STEP 2: SPECIFY SEARCH IMAGE")
        print("="*70)
        search_image_path = input("Search image path (Enter for same): ").strip()
        if not search_image_path:
            search_image_path = template_image_path
    
    # Step 3: Detect symbols
    print("\n" + "="*70)
    print("STEP 3: DETECTING SYMBOLS")
    print("="*70)
    detections, result_image = detector.detect_symbols(search_image_path, visualize=True)
    
    # Get summary
    summary = detector.get_detection_summary(detections)
    
    print(f"\n{'='*70}")
    print(f"DETECTION RESULTS")
    print(f"{'='*70}")
    print(f"Method: {method.upper()}")
    print(f"Total symbols found: {summary['count']}")
    if summary['count'] > 0:
        print(f"Average confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"Confidence range: {summary.get('min_confidence', 0):.3f} - {summary.get('max_confidence', 0):.3f}")
        print(f"\nDetection details:")
        for idx, det in enumerate(detections):
            print(f"  #{idx+1}: pos={det.center}, conf={det.confidence:.3f}, " +
                  f"scale={det.scale:.2f}x, rot={det.rotation:.1f}°")
    
    # Save output
    if save_output:
        import os
        image_dir = os.path.dirname(search_image_path) or '.'
        output_folder = os.path.join(image_dir, 'output')
        
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print("="*70)
        
        if output_path is None:
            saved_path = detector.save_result_image(search_image_path, result_image, output_folder)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result_image)
            saved_path = output_path
            print(f"✓ Result saved to: {output_path}")
        
        template_path = os.path.join(output_folder, f'template_{method}.png')
        cv2.imwrite(template_path, template)
        print(f"✓ Template saved: {template_path}")
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    plt.imshow(template_rgb)
    plt.title(f'Template ({method.upper()})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.imshow(result_rgb)
    plt.title(f'Detections: {summary["count"]} ({method.upper()})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'detections': detections,
        'summary': summary,
        'result_image': result_image,
        'template': template,
        'template_source': template_image_path,
        'search_source': search_image_path,
        'method': method
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python electrical_symbol_detector.py <template_image_path> [search_image_path]")
        print("\nArguments:")
        print("  template_image_path    Image to select template from")
        print("  search_image_path      (Optional) Image to search in")
        print("\nOptional arguments:")
        print("  --method <name>          Detection method (default: template_matching)")
        print("                           Options: template_matching, sift, orb, akaze, brisk")
        print("  --threshold <value>      Matching threshold (default: 0.7)")
        print("                           Recommended: 0.7-0.8 for template_matching")
        print("                                       0.3-0.5 for feature methods")
        print("  --nms-threshold <value>  NMS threshold (default: 0.3)")
        print("  --window-scale <value>   Window size multiplier for features (default: 2.5)")
        print("\nExamples:")
        print("  # Template matching (searches all scales/rotations):")
        print("  python electrical_symbol_detector.py layout.png --threshold 0.8")
        print("")
        print("  # ORB with sliding window (memory-efficient, good for small symbols):")
        print("  python electrical_symbol_detector.py ref.png search.png --method orb --threshold 0.4")
        print("")
        print("  # AKAZE with larger search window:")
        print("  python electrical_symbol_detector.py ref.png --method akaze --threshold 0.5 --window-scale 3.0")
        print("")
        print("  # BRISK (fastest feature method):")
        print("  python electrical_symbol_detector.py ref.png --method brisk --threshold 0.4")
        print("")
        print("Method Comparison:")
        print("  template_matching: Multi-scale search, thorough but slower")
        print("  orb:              Fast, handles rotation, memory-efficient (RECOMMENDED)")
        print("  akaze:            Good balance, handles rotation/scale")
        print("  brisk:            Fastest feature method")
        print("  sift:             Most accurate but memory-intensive (use with caution)")
        sys.exit(1)
    
    template_image_path = sys.argv[1]
    search_image_path = None
    
    # Check if second positional argument is provided (search image)
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        search_image_path = sys.argv[2]
        arg_start = 3
    else:
        arg_start = 2
    
    # Parse optional arguments
    method = 'template_matching'
    threshold = 0.7
    nms_threshold = 0.3
    window_scale = 2.5
    
    i = arg_start
    while i < len(sys.argv):
        if sys.argv[i] == '--method' and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--nms-threshold' and i + 1 < len(sys.argv):
            nms_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--window-scale' and i + 1 < len(sys.argv):
            window_scale = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Run detection
    results = run_detection_pipeline(
        template_image_path=template_image_path,
        search_image_path=search_image_path,
        method=method,
        threshold=threshold,
        nms_threshold=nms_threshold,
        window_scale=window_scale
    )