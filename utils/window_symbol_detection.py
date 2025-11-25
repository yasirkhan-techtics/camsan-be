"""
Electrical Symbol Detection System - Multi-Template Version
Detects and counts electrical symbols using bounding box coordinates as templates.
Supports multiple reference symbols with text labels.
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
    text_label: str  # Added text label from reference


@dataclass
class TemplateInfo:
    """Store template information"""
    text: str
    template_gray: np.ndarray
    original_bbox: Tuple[int, int, int, int]


class MultiTemplateSymbolDetector:
    """
    Detects electrical symbols in layout drawings using bounding boxes as templates.
    Supports multiple templates with text labels.
    """
    
    def __init__(self, 
                 scales: List[float] = None,
                 rotations: List[float] = None,
                 threshold: float = 0.7,
                 nms_threshold: float = 0.3):
        """
        Initialize the detector.
        
        Args:
            scales: List of scale factors to search
            rotations: List of rotation angles in degrees
            threshold: Matching threshold (0-1, higher = more strict)
            nms_threshold: Non-maximum suppression IoU threshold
        """
        self.scales = scales or [1]
        self.rotations = rotations or list(range(0, 360, 90))
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.templates = []  # List of TemplateInfo objects
        
    def load_templates_from_bboxes(self, 
                                   image_path: str,
                                   bbox_list: List[Dict]) -> List[TemplateInfo]:
        """
        Load templates from bounding box coordinates in the image.
        
        Args:
            image_path: Path to the image containing reference symbols
            bbox_list: List of dictionaries with 'Text', 'x', 'y', 'Width', 'Height'
            
        Returns:
            templates: List of TemplateInfo objects
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.templates = []
        
        print("\n" + "="*70)
        print("LOADING REFERENCE TEMPLATES FROM BOUNDING BOXES")
        print("="*70)
        
        for idx, bbox_dict in enumerate(bbox_list):
            text = bbox_dict.get('Text', f'Symbol_{idx}')
            x = bbox_dict['x']
            y = bbox_dict['y']
            w = bbox_dict['Width']
            h = bbox_dict['Height']
            
            # Extract template from image
            template_gray = image_gray[y:y+h, x:x+w].copy()
            
            if template_gray.size == 0:
                print(f"  Warning: Empty template for '{text}' - skipping")
                continue
            
            template_info = TemplateInfo(
                text=text,
                template_gray=template_gray,
                original_bbox=(x, y, w, h)
            )
            
            self.templates.append(template_info)
            
            print(f"  [{idx+1}] Loaded template '{text}'")
            print(f"      Position: ({x}, {y}), Size: {w}x{h} pixels")
        
        print("="*70)
        print(f"Total templates loaded: {len(self.templates)}")
        print("="*70 + "\n")
        
        return self.templates
    
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
        Detect all instances of the template symbols in the image.
        
        Args:
            image_path: Path to the image to search in
            visualize: Whether to create visualization
            
        Returns:
            detections: List of Detection objects
            result_image: Image with drawn detections
        """
        if len(self.templates) == 0:
            raise ValueError("No templates loaded. Call load_templates_from_bboxes() first.")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_gray.shape
        
        all_detections = []
        
        print("Searching for symbols...")
        
        # Search for each template
        for template_info in self.templates:
            print(f"\nSearching for template: '{template_info.text}'")
            
            template_detections = []
            total_iterations = len(self.scales) * len(self.rotations)
            current_iteration = 0
            
            # Search across scales and rotations
            for scale in self.scales:
                for rotation in self.rotations:
                    current_iteration += 1
                    
                    # Progress indicator
                    if current_iteration % 10 == 0:
                        print(f"  Progress: {current_iteration}/{total_iterations}")
                    
                    # Resize and rotate template
                    h_t, w_t = template_info.template_gray.shape
                    new_w = int(w_t * scale)
                    new_h = int(h_t * scale)
                    
                    if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
                        continue
                    
                    # Resize template
                    resized_template = cv2.resize(template_info.template_gray, (new_w, new_h))
                    
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
                            center=center,
                            text_label=template_info.text
                        )
                        template_detections.append(detection)
            
            print(f"  Found {len(template_detections)} raw detections for '{template_info.text}'")
            all_detections.extend(template_detections)
        
        print(f"\nTotal raw detections: {len(all_detections)}")
        
        # Apply non-maximum suppression
        filtered_detections = self.non_max_suppression(all_detections)
        
        print(f"After NMS: {len(filtered_detections)} detections")
        
        # Create visualization
        result_image = image.copy()
        if visualize:
            result_image = self.draw_detections(result_image, filtered_detections)
        
        return filtered_detections, result_image
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes on image with text labels.
        
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
        
        # Group detections by text label for color consistency
        label_colors = {}
        base_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 128, 0),  # Orange
        ]
        
        for idx, det in enumerate(detections):
            # Assign color based on text label
            if det.text_label not in label_colors:
                color_idx = len(label_colors) % len(base_colors)
                label_colors[det.text_label] = base_colors[color_idx]
            
            color = label_colors[det.text_label]
            
            x, y, w, h = det.bbox
            
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)
            
            # Add label with text and confidence
            label = f"{det.text_label} ({det.confidence:.2f})"
            font_scale = max(0.5, line_thickness * 0.25)
            font_thickness = max(1, line_thickness // 2)
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(result, 
                         (x, y - text_h - baseline - 5), 
                         (x + text_w + 4, y), 
                         color, -1)
            
            # Draw text in black for contrast
            cv2.putText(result, label, (x + 2, y - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        return result
    
    def get_detection_summary(self, detections: List[Detection]) -> Dict:
        """
        Get summary of detections grouped by text label.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            summary: Dictionary with detection statistics
        """
        if len(detections) == 0:
            return {
                'total_count': 0,
                'by_label': {}
            }
        
        # Group by label
        by_label = {}
        for det in detections:
            if det.text_label not in by_label:
                by_label[det.text_label] = []
            by_label[det.text_label].append(det)
        
        # Create summary
        summary = {
            'total_count': len(detections),
            'by_label': {}
        }
        
        for label, dets in by_label.items():
            summary['by_label'][label] = {
                'count': len(dets),
                'detections': [
                    {
                        'bbox': det.bbox,
                        'center': det.center,
                        'confidence': det.confidence,
                        'scale': det.scale,
                        'rotation': det.rotation
                    }
                    for det in dets
                ],
                'avg_confidence': np.mean([d.confidence for d in dets]),
                'min_confidence': np.min([d.confidence for d in dets]),
                'max_confidence': np.max([d.confidence for d in dets])
            }
        
        return summary
    
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
            output_filename: Custom output filename
            
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
        
        # Save the result
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"✓ Result image saved: {output_path}")
        else:
            print(f"✗ Failed to save image to: {output_path}")
        
        return output_path


def run_multi_template_detection(image_path: str,
                                 bbox_list: List[Dict],
                                 scales: List[float] = None,
                                 rotations: List[float] = None,
                                 threshold: float = 0.7,
                                 nms_threshold: float = 0.3,
                                 save_output: bool = True,
                                 output_folder: str = None) -> Dict:
    """
    Run the complete detection pipeline with multiple templates.
    
    Args:
        image_path: Path to the input image
        bbox_list: List of dicts with 'Text', 'x', 'y', 'Width', 'Height'
        scales: List of scale factors to search
        rotations: List of rotation angles in degrees
        threshold: Matching threshold (0-1)
        nms_threshold: Non-maximum suppression IoU threshold
        save_output: Whether to save the output image
        output_folder: Custom output folder
        
    Returns:
        results: Dictionary containing detections and summary
    """
    # Initialize detector
    detector = MultiTemplateSymbolDetector(
        scales=scales,
        rotations=rotations,
        threshold=threshold,
        nms_threshold=nms_threshold
    )
    
    # Load templates from bounding boxes
    print("Step 1: Loading templates from bounding boxes")
    detector.load_templates_from_bboxes(image_path, bbox_list)
    
    # Detect symbols
    print("\nStep 2: Detecting symbols in image")
    detections, result_image = detector.detect_symbols(image_path, visualize=True)
    
    # Get summary
    summary = detector.get_detection_summary(detections)
    
    print(f"\n{'='*70}")
    print(f"DETECTION RESULTS")
    print(f"{'='*70}")
    print(f"Total symbols found: {summary['total_count']}")
    print(f"\nBreakdown by template:")
    
    for label, info in summary['by_label'].items():
        print(f"\n  Template '{label}': {info['count']} detections")
        print(f"    Avg confidence: {info['avg_confidence']:.3f}")
        print(f"    Range: [{info['min_confidence']:.3f}, {info['max_confidence']:.3f}]")
        
        for idx, det_info in enumerate(info['detections']):
            print(f"    Detection #{idx+1}:")
            print(f"      - Center: {det_info['center']}")
            print(f"      - BBox: {det_info['bbox']}")
            print(f"      - Confidence: {det_info['confidence']:.3f}")
    
    # Save output
    if save_output:
        print(f"\n{'='*70}")
        print(f"SAVING RESULTS")
        print(f"{'='*70}")
        detector.save_result_image(image_path, result_image, output_folder)
    
    return {
        'detections': detections,
        'summary': summary,
        'result_image': result_image
    }


def display_results_jupyter(results: Dict, figsize=(15, 10)):
    """
    Display detection results in Jupyter notebook.
    
    Args:
        results: Results dictionary from run_multi_template_detection
        figsize: Figure size for matplotlib
    """
    result_image = results['result_image']
    summary = results['summary']
    
    # Convert BGR to RGB for matplotlib
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Display image
    plt.figure(figsize=figsize)
    plt.imshow(result_image_rgb)
    plt.axis('off')
    plt.title(f"Detection Results - Total: {summary['total_count']} symbols found", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total symbols found: {summary['total_count']}\n")
    
    for label, info in summary['by_label'].items():
        print(f"Template '{label}': {info['count']} detections")
        print(f"  Avg confidence: {info['avg_confidence']:.3f}")
        print(f"  Range: [{info['min_confidence']:.3f}, {info['max_confidence']:.3f}]\n")


if __name__ == "__main__":
    # Example usage when running as script
    
    # Example bounding box list
    bbox_list = [
        {'Text': 'CF1', 'x': 6049, 'y': 3676, 'Width': 60, 'Height': 30},
        {'Text': 'BF3', 'x': 5060, 'y': 6177, 'Width': 66, 'Height': 29}
    ]
    
    # Image path (the same image contains both reference symbols and target symbols)
    image_path = "layout_drawing.png"
    
    # Run detection
    results = run_multi_template_detection(
        image_path=image_path,
        bbox_list=bbox_list,
        threshold=0.7,
        nms_threshold=0.3,
        save_output=True
    )
    
    print("\n" + "="*70)
    print("DETECTION COMPLETE!")
    print("="*70)