"""
Electrical Symbol Detection System
Detects and counts electrical symbols in layout drawings with scale and rotation invariance.
Uses a separate reference image for the template symbol.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os


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
    Detects electrical symbols in layout drawings using multi-scale and rotation-invariant matching.
    """
    
    def __init__(self, 
                 scales: List[float] = None,
                 rotations: List[float] = None,
                 threshold: float = 0.7,
                 nms_threshold: float = 0.3):
        """
        Initialize the detector.
        
        Args:
            scales: List of scale factors to search (default: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
            rotations: List of rotation angles in degrees (default: 0 to 360 in 15-degree steps)
            threshold: Matching threshold (0-1, higher = more strict)
            nms_threshold: Non-maximum suppression IoU threshold
        """
        # self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        # self.rotations = rotations or list(range(0, 360, 45))
        self.scales = scales or [0.9, 1.0, 1.1, 1.11]
        self.rotations = rotations or list(range(0, 360, 90))
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.template = None
        self.template_gray = None
        
    def load_reference_image(self, reference_path: str) -> np.ndarray:
        """
        Load a reference image to use as the template.
        
        Args:
            reference_path: Path to the reference image
            
        Returns:
            template: The loaded template image
        """
        # Read reference image
        template = cv2.imread(reference_path)
        if template is None:
            raise ValueError(f"Could not load reference image from {reference_path}")
        
        # Store template
        self.template = template.copy()
        self.template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        h, w = template.shape[:2]
        
        print("\n" + "="*70)
        print("REFERENCE IMAGE LOADED SUCCESSFULLY!")
        print("="*70)
        print(f"  Reference image: {os.path.basename(reference_path)}")
        print(f"  Template size: {w}x{h} pixels")
        print("="*70 + "\n")
        
        return template
    
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
    
    def detect_symbols(self, image_path: str, visualize: bool = True) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect all instances of the template symbol in the image.
        
        Args:
            image_path: Path to the image to search in
            visualize: Whether to create visualization
            
        Returns:
            detections: List of Detection objects
            result_image: Image with drawn detections
        """
        if self.template is None:
            raise ValueError("Template not loaded. Call load_reference_image() first.")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_gray.shape
        
        all_detections = []
        
        print("Searching for symbols...")
        total_iterations = len(self.scales) * len(self.rotations)
        current_iteration = 0
        
        # Search across scales and rotations
        for scale in self.scales:
            for rotation in self.rotations:
                current_iteration += 1
                
                # Progress indicator
                if current_iteration % 10 == 0:
                    print(f"Progress: {current_iteration}/{total_iterations}")
                
                # Resize and rotate template
                h_t, w_t = self.template_gray.shape
                new_w = int(w_t * scale)
                new_h = int(h_t * scale)
                
                if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
                    continue
                
                # Resize template
                resized_template = cv2.resize(self.template_gray, (new_w, new_h))
                
                # Rotate template
                rotated_template, _ = self.rotate_image(resized_template, rotation)
                
                # Skip if template is too large
                if rotated_template.shape[0] > h_img or rotated_template.shape[1] > w_img:
                    continue
                
                # Perform template matching
                result = self.template_match(image_gray, rotated_template)
                
                if result is None:
                    continue
                
                # Find matches above threshold
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
        
        print(f"Found {len(all_detections)} raw detections")
        
        # Apply non-maximum suppression
        filtered_detections = self.non_max_suppression(all_detections)
        filtered_detections = all_detections
        
        print(f"After NMS: {len(filtered_detections)} detections")
        
        # Create visualization
        result_image = image.copy()
        if visualize:
            result_image = self.draw_detections(result_image, filtered_detections)
        
        return filtered_detections, result_image
    
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
        
        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox
            
            # Draw rectangle with bright green color for better visibility
            color = (0, 255, 0)  # Green (BGR)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)
            
            # # Draw center point
            # point_size = max(1, line_thickness)
            # cv2.circle(result, det.center, point_size, (0, 0, 255), -1)  # Red center
            
            # Add label with background for better readability
            label = f"#{idx+1} ({det.confidence:.2f})"
            font_scale = max(0.4, line_thickness * 0.2)
            font_thickness = max(1, line_thickness // 2)
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(result, 
                         (x, y - text_h - baseline - 5), 
                         (x + text_w, y), 
                         color, -1)
            
            # Draw text
            cv2.putText(result, label, (x, y - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        return result
    
    def save_result_image(self, 
                         image_path: str, 
                         result_image: np.ndarray, 
                         output_folder: str = None,
                         output_filename: str = None) -> str:
        """
        Save the result image with bounding boxes to an output folder.
        
        Args:
            image_path: Original image path
            result_image: Image with drawn detections
            output_folder: Custom output folder (default: 'output' in same directory)
            output_filename: Custom output filename (default: original_name_detected.ext)
            
        Returns:
            output_path: Path where the image was saved
        """
        # Create output folder
        if output_folder is None:
            image_dir = os.path.dirname(image_path) or '.'
            output_folder = os.path.join(image_dir, 'output')
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate output filename
        if output_filename is None:
            image_filename = os.path.basename(image_path)
            image_name, image_ext = os.path.splitext(image_filename)
            output_filename = f"{image_name}_detected{image_ext}"
        
        output_path = os.path.join(output_folder, output_filename)
        
        # Ensure result image has same dimensions as original
        original_img = cv2.imread(image_path)
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


def run_detection_pipeline(image_path: str,
                          reference_path: str,
                          scales: List[float] = None,
                          rotations: List[float] = None,
                          threshold: float = 0.7,
                          nms_threshold: float = 0.3,
                          save_output: bool = True,
                          output_path: str = None) -> Dict:
    """
    Run the complete detection pipeline.
    
    Args:
        image_path: Path to the input image
        reference_path: Path to the reference symbol image
        scales: List of scale factors to search
        rotations: List of rotation angles in degrees
        threshold: Matching threshold (0-1)
        nms_threshold: Non-maximum suppression IoU threshold
        save_output: Whether to save the output image
        output_path: Path to save the output image (default: input_path_result.jpg)
        
    Returns:
        results: Dictionary containing detections and summary
    """
    # Initialize detector
    detector = ElectricalSymbolDetector(
        scales=scales,
        rotations=rotations,
        threshold=threshold,
        nms_threshold=nms_threshold
    )
    
    # Load reference image
    print("Step 1: Loading reference symbol image")
    template = detector.load_reference_image(reference_path)
    
    # Detect symbols
    print("\nStep 2: Detecting symbols in image")
    detections, result_image = detector.detect_symbols(image_path, visualize=True)
    
    # Get summary
    summary = detector.get_detection_summary(detections)
    
    print(f"\n{'='*50}")
    print(f"DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total symbols found: {summary['count']}")
    if summary['count'] > 0:
        print(f"Average confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"Min confidence: {summary.get('min_confidence', 0):.3f}")
        print(f"Max confidence: {summary.get('max_confidence', 0):.3f}")
        print(f"\nDetection details:")
        for idx, det in enumerate(detections):
            print(f"  Symbol #{idx+1}:")
            print(f"    - Position: {det.center}")
            print(f"    - BBox: {det.bbox}")
            print(f"    - Confidence: {det.confidence:.3f}")
            print(f"    - Scale: {det.scale:.2f}x")
            print(f"    - Rotation: {det.rotation}°")
    
    # Save output
    if save_output:
        print(f"\n{'='*50}")
        print(f"SAVING RESULTS")
        print(f"{'='*50}")
        
        # Create output folder
        image_dir = os.path.dirname(image_path) or '.'
        output_folder = os.path.join(image_dir, 'output')
        
        # Save result image with detections
        if output_path is None:
            saved_path = detector.save_result_image(image_path, result_image, output_folder)
        else:
            # Use custom path if provided
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            original_img = cv2.imread(image_path)
            if result_image.shape != original_img.shape:
                result_image = cv2.resize(result_image, (original_img.shape[1], original_img.shape[0]))
            cv2.imwrite(output_path, result_image)
            saved_path = output_path
            print(f"✓ Result saved to custom path: {output_path}")
        
        # Also save a copy of the reference template for documentation
        template_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_reference.png'
        template_path = os.path.join(output_folder, template_filename)
        cv2.imwrite(template_path, template)
        print(f"\n✓ Reference template saved: {template_path}")
    
    return {
        'detections': detections,
        'summary': summary,
        'result_image': result_image,
        'template': template
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python electrical_symbol_detector.py <image_path> <reference_path>")
        print("\nArguments:")
        print("  image_path       Path to the image where symbols need to be detected")
        print("  reference_path   Path to the reference symbol image")
        print("\nOptional arguments:")
        print("  --threshold <value>      Matching threshold (default: 0.7)")
        print("  --nms-threshold <value>  NMS threshold (default: 0.3)")
        print("\nExample:")
        print("  python electrical_symbol_detector.py layout.png symbol_reference.png")
        print("  python electrical_symbol_detector.py layout.png symbol.png --threshold 0.8")
        sys.exit(1)
    
    image_path = sys.argv[1]
    reference_path = sys.argv[2]
    
    # Parse optional arguments
    threshold = 0.7
    nms_threshold = 0.3
    
    for i in range(3, len(sys.argv), 2):
        if sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
        elif sys.argv[i] == '--nms-threshold' and i + 1 < len(sys.argv):
            nms_threshold = float(sys.argv[i + 1])
    
    # Validate file paths
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(reference_path):
        print(f"Error: Reference image file not found: {reference_path}")
        sys.exit(1)
    
    # Run detection
    results = run_detection_pipeline(
        image_path=image_path,
        reference_path=reference_path,
        threshold=threshold,
        nms_threshold=nms_threshold
    )