"""
Electrical Symbol Detection System
Detects and counts electrical symbols in layout drawings with multiple matching methods.
Methods: template_matching, edge_matching, shape_matching, frequency_matching, cnn_features
All methods use confidence threshold + NMS filtering.
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
    Methods: template_matching, edge_matching, shape_matching, frequency_matching, cnn_features
    """
    
    def __init__(self, 
                 scales: List[float] = None,
                 rotations: List[float] = None,
                 threshold: float = 0.7,
                 nms_threshold: float = 0.3,
                 method: str = 'template_matching'):
        """
        Initialize the detector.
        
        Args:
            scales: List of scale factors
            rotations: List of rotation angles in degrees
            threshold: Matching threshold
            nms_threshold: Non-maximum suppression IoU threshold
            method: Detection method - 'template_matching', 'edge_matching', 
                   'shape_matching', 'frequency_matching', 'cnn_features'
        """
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.rotations = rotations or list(range(0, 360, 15))
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.method = method.lower()
        self.template = None
        self.template_gray = None
        self.template_edges = None
        self.template_contour = None
        self.template_cnn_features = None
        self.template_source_image = None
        
        # Validate method
        valid_methods = ['template_matching', 'edge_matching', 'shape_matching', 
                        'frequency_matching', 'cnn_features']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Choose from: {valid_methods}")
        
        # Initialize CNN model if needed
        if self.method == 'cnn_features':
            self._initialize_cnn()
    
    def _initialize_cnn(self):
        """Initialize pre-trained CNN for feature extraction (VGG16)"""
        try:
            # Load pre-trained VGG16 model (built into OpenCV DNN)
            # Note: You need to download the model files first
            print("Initializing CNN (VGG16) for feature extraction...")
            print("Note: VGG16 model files must be downloaded separately")
            print("Download from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API")
            
            # Placeholder - will be loaded if files exist
            self.cnn_model = None
            print("✓ CNN placeholder initialized (download model files to use)")
            
        except Exception as e:
            print(f"⚠️  CNN initialization failed: {e}")
            self.cnn_model = None
    
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
        
        # Preprocess template based on method
        if self.method == 'edge_matching':
            self.template_edges = cv2.Canny(self.template_gray, 50, 150)
            print(f"✓ Extracted edge map from template")
        
        elif self.method == 'shape_matching':
            # Extract contour from template
            _, template_bin = cv2.threshold(self.template_gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # Get largest contour
                self.template_contour = max(contours, key=cv2.contourArea)
                print(f"✓ Extracted shape contour from template ({len(self.template_contour)} points)")
            else:
                print("⚠️  Warning: No contour found in template")
        
        elif self.method == 'cnn_features':
            # Extract CNN features (placeholder - requires model files)
            print("✓ Template ready for CNN feature extraction")
            # self.template_cnn_features = self._extract_cnn_features(self.template_gray)
        
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
    
    def template_match_standard(self, image_gray: np.ndarray) -> List[Detection]:
        """
        Standard template matching with multi-scale and rotation.
        """
        detections = []
        h_img, w_img = image_gray.shape
        
        print("Searching across scales and rotations...")
        total_iterations = len(self.scales) * len(self.rotations)
        current_iteration = 0
        
        for scale in self.scales:
            for rotation in self.rotations:
                current_iteration += 1
                
                if current_iteration % 10 == 0:
                    print(f"  Progress: {current_iteration}/{total_iterations}")
                
                h_t, w_t = self.template_gray.shape
                new_w = int(w_t * scale)
                new_h = int(h_t * scale)
                
                if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
                    continue
                
                resized_template = cv2.resize(self.template_gray, (new_w, new_h))
                rotated_template, _ = self.rotate_image(resized_template, rotation)
                
                if rotated_template.shape[0] > h_img or rotated_template.shape[1] > w_img:
                    continue
                
                # Template matching
                result = cv2.matchTemplate(image_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
                
                # Find all matches (no threshold here, will filter later)
                locations = np.where(result >= 0.5)  # Very low threshold to get all candidates
                
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
                    detections.append(detection)
        
        return detections
    
    def template_match_edges(self, image_gray: np.ndarray) -> List[Detection]:
        """
        Edge-based template matching (robust to noise and lighting).
        """
        detections = []
        h_img, w_img = image_gray.shape
        
        # Extract edges from search image
        image_edges = cv2.Canny(image_gray, 50, 150)
        
        print("Searching with edge-based matching...")
        total_iterations = len(self.scales) * len(self.rotations)
        current_iteration = 0
        
        for scale in self.scales:
            for rotation in self.rotations:
                current_iteration += 1
                
                if current_iteration % 10 == 0:
                    print(f"  Progress: {current_iteration}/{total_iterations}")
                
                h_t, w_t = self.template_edges.shape
                new_w = int(w_t * scale)
                new_h = int(h_t * scale)
                
                if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
                    continue
                
                resized_template = cv2.resize(self.template_edges, (new_w, new_h))
                rotated_template, _ = self.rotate_image(resized_template, rotation)
                
                if rotated_template.shape[0] > h_img or rotated_template.shape[1] > w_img:
                    continue
                
                # Edge-based template matching
                result = cv2.matchTemplate(image_edges, rotated_template, cv2.TM_CCOEFF_NORMED)
                
                locations = np.where(result >= 0.5)
                
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
                    detections.append(detection)
        
        return detections
    
    def shape_match(self, image_gray: np.ndarray) -> List[Detection]:
        """
        Shape-based matching using contours (handles partial occlusion).
        """
        detections = []
        
        if self.template_contour is None:
            print("⚠️  No template contour available")
            return detections
        
        print("Searching with shape matching...")
        
        # Convert image to binary
        _, image_bin = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in image
        contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Found {len(contours)} contours in image")
        
        # Get template size for scale estimation
        template_area = cv2.contourArea(self.template_contour)
        
        for idx, contour in enumerate(contours):
            if idx % 100 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(contours)} contours")
            
            # Skip very small contours
            contour_area = cv2.contourArea(contour)
            if contour_area < 100:
                continue
            
            # Shape matching (rotation invariant)
            match_score = cv2.matchShapes(self.template_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
            
            # Convert match score to confidence (lower is better, invert it)
            confidence = 1.0 / (1.0 + match_score)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            
            # Estimate scale
            scale = np.sqrt(contour_area / template_area) if template_area > 0 else 1.0
            
            detection = Detection(
                bbox=(x, y, w, h),
                confidence=float(confidence),
                scale=float(scale),
                rotation=0.0,  # Shape matching is rotation invariant
                center=center
            )
            detections.append(detection)
        
        return detections
    
    def frequency_match(self, image_gray: np.ndarray) -> List[Detection]:
        """
        Frequency domain matching using phase correlation (fast for large images).
        """
        detections = []
        h_img, w_img = image_gray.shape
        
        print("Searching with frequency domain matching...")
        
        # Use sliding window approach for frequency matching
        h_t, w_t = self.template_gray.shape
        window_h = int(h_t * 2)
        window_w = int(w_t * 2)
        stride = min(h_t, w_t) // 2
        
        total_windows = ((h_img - window_h) // stride + 1) * ((w_img - window_w) // stride + 1)
        window_count = 0
        
        for y in range(0, h_img - window_h + 1, stride):
            for x in range(0, w_img - window_w + 1, stride):
                window_count += 1
                
                if window_count % 50 == 0:
                    print(f"  Progress: {window_count}/{total_windows} windows")
                
                window = image_gray[y:y+window_h, x:x+window_w]
                
                # Resize template to match window for comparison
                template_resized = cv2.resize(self.template_gray, (window_w, window_h))
                
                # Compute normalized cross-correlation using FFT
                # This is faster than spatial domain for large images
                result = cv2.matchTemplate(window, template_resized, cv2.TM_CCOEFF_NORMED)
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.5:  # Low threshold, will filter later
                    # Convert to global coordinates
                    x_global = x + max_loc[0]
                    y_global = y + max_loc[1]
                    
                    center = (x_global + w_t // 2, y_global + h_t // 2)
                    
                    detection = Detection(
                        bbox=(x_global, y_global, w_t, h_t),
                        confidence=float(max_val),
                        scale=1.0,
                        rotation=0.0,
                        center=center
                    )
                    detections.append(detection)
        
        return detections
    
    def cnn_feature_match(self, image_gray: np.ndarray) -> List[Detection]:
        """
        CNN feature-based matching (one-shot learning approach).
        Note: Requires pre-trained CNN model files.
        """
        detections = []
        
        print("CNN feature matching...")
        print("⚠️  This method requires VGG16 model files to be downloaded")
        print("    Falling back to standard template matching")
        
        # Fallback to standard template matching
        detections = self.template_match_standard(image_gray)
        
        return detections
    
    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression with IoU filtering"""
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
        All raw outputs pass through confidence threshold + NMS filtering.
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
        
        # Select matching method
        if self.method == 'template_matching':
            all_detections = self.template_match_standard(image_gray)
        elif self.method == 'edge_matching':
            all_detections = self.template_match_edges(image_gray)
        elif self.method == 'shape_matching':
            all_detections = self.shape_match(image_gray)
        elif self.method == 'frequency_matching':
            all_detections = self.frequency_match(image_gray)
        elif self.method == 'cnn_features':
            all_detections = self.cnn_feature_match(image_gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        print(f"\nFound {len(all_detections)} raw detections")
        
        # Filter by confidence threshold (for ALL methods)
        print(f"Applying confidence threshold: {self.threshold}")
        threshold_filtered = [det for det in all_detections if det.confidence >= self.threshold]
        print(f"After threshold filter: {len(threshold_filtered)} detections")
        
        # Apply NMS with IoU filtering (for ALL methods)
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
            font_scale = max(0.1, line_thickness * 0.05)
            font_thickness = max(0.25, line_thickness // 8)
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            
            # Draw text
            cv2.putText(result, label, (x, y - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            
            # label = f"#{idx+1} ({det.confidence:.2f})"
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 0.6
            # font_thickness = max(1, line_thickness // 2)
            
            # (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # while text_w > w - 6 and font_scale > 0.3:
            #     font_scale -= 0.02
            #     (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # text_x = x
            # text_y = y - text_h - baseline - 5
            
            # if text_y < 0:
            #     text_y = y + h + text_h + baseline + 5
            
            # overlay = result.copy()
            # cv2.rectangle(overlay, (text_x, text_y),
            #              (text_x + text_w + 4, text_y + text_h + baseline + 4),
            #              color, -1)
            # cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            # cv2.putText(result, label, (text_x + 2, text_y + text_h + 2),
            #            font, font_scale, (0, 0, 0), font_thickness)
        
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
                          save_output: bool = True,
                          output_path: str = None,
                          max_display_size: int = 1200) -> Dict:
    """
    Run the complete detection pipeline with selected matching method.
    All methods use confidence threshold + NMS filtering.
    
    Args:
        template_image_path: Path to the image for template selection
        search_image_path: Path to the image to search in
        method: Matching method - 'template_matching', 'edge_matching', 
               'shape_matching', 'frequency_matching', 'cnn_features'
        scales: List of scale factors (for template/edge/frequency methods)
        rotations: List of rotation angles (for template/edge methods)
        threshold: Matching threshold (0-1)
        nms_threshold: Non-maximum suppression IoU threshold
        save_output: Whether to save the output image
        output_path: Path to save the output image
        max_display_size: Maximum width/height for display window
        
    Returns:
        results: Dictionary containing detections and summary
    
    Examples:
        # Standard template matching
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='template_matching', threshold=0.8)
        
        # Edge-based matching (robust to noise)
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='edge_matching', threshold=0.75)
        
        # Shape matching (handles occlusion)
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='shape_matching', threshold=0.6)
        
        # Frequency matching (fast for large images)
        results = run_detection_pipeline('ref.png', 'search.png', 
                                        method='frequency_matching', threshold=0.7)
    """
    # Initialize detector with selected method
    detector = ElectricalSymbolDetector(
        scales=scales,
        rotations=rotations,
        threshold=threshold,
        nms_threshold=nms_threshold,
        method=method
    )
    
    # Step 1: Select template
    print("="*70)
    print(f"DETECTION METHOD: {method.upper()}")
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
        print("                           Options:")
        print("                             - template_matching: Standard multi-scale matching")
        print("                             - edge_matching: Edge-based (robust to noise)")
        print("                             - shape_matching: Contour-based (handles occlusion)")
        print("                             - frequency_matching: FFT-based (fast for large images)")
        print("                             - cnn_features: CNN features (requires model files)")
        print("  --threshold <value>      Matching threshold (default: 0.7)")
        print("  --nms-threshold <value>  NMS IoU threshold (default: 0.3)")
        print("\nExamples:")
        print("  # Standard template matching:")
        print("  python electrical_symbol_detector.py layout.png --threshold 0.8")
        print("")
        print("  # Edge-based matching (best for noisy images):")
        print("  python electrical_symbol_detector.py ref.png search.png --method edge_matching")
        print("")
        print("  # Shape matching (best for overlapping symbols):")
        print("  python electrical_symbol_detector.py ref.png --method shape_matching --threshold 0.6")
        print("")
        print("  # Frequency matching (fastest for large images):")
        print("  python electrical_symbol_detector.py ref.png --method frequency_matching")
        print("")
        print("Method Comparison:")
        print("  template_matching:  Best accuracy, comprehensive search")
        print("  edge_matching:      Robust to lighting/noise variations")
        print("  shape_matching:     Handles partial occlusion, rotation invariant")
        print("  frequency_matching: Fast for large images, good for similar sizes")
        print("  cnn_features:       Requires VGG16 model files (experimental)")
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
        else:
            i += 1
    
    # Run detection
    results = run_detection_pipeline(
        template_image_path=template_image_path,
        search_image_path=search_image_path,
        method=method,
        threshold=threshold,
        nms_threshold=nms_threshold
    )