"""
Electrical Symbol Detection System
Detects and counts electrical symbols in layout drawings with scale and rotation invariance.
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
    Detects electrical symbols in layout drawings using multi-scale and rotation-invariant matching.
    """

    def __init__(self,
                 scales: List[float] = None,
                 rotations: List[float] = None,
                 threshold: float = 0.7,
                 nms_threshold: float = 0.2):
        """
        Initialize the detector.

        Args:
            scales: List of scale factors to search (default: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
            rotations: List of rotation angles in degrees (default: 0 to 360 in 15-degree steps)
            threshold: Matching threshold (0-1, higher = more strict)
            nms_threshold: Non-maximum suppression IoU threshold
        """
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.rotations = rotations or [0, 90, 180, 270]  # Only check 90-degree rotations for speed
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.template = None
        self.template_gray = None

    def rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """
        Rotate image by given angle with proper scaling.
        """
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

    def template_match(self,
                       image_gray: np.ndarray,
                       template: np.ndarray,
                       method: int = cv2.TM_CCOEFF_NORMED) -> np.ndarray:
        """
        Perform template matching.
        """
        if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            return None

        result = cv2.matchTemplate(image_gray, template, method)
        return result

    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        """
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
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        """
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

    def detect_symbols(self, image_path: str, visualize: bool = True) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect all instances of the template symbol in the image.
        """
        if self.template is None:
            raise ValueError("Template not selected.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_gray.shape

        all_detections = []
        
        total_iterations = len(self.scales) * len(self.rotations)
        current_iteration = 0
        last_progress = 0
        
        print(f"         Starting template matching: {len(self.scales)} scales × {len(self.rotations)} rotations = {total_iterations} iterations")

        for scale_idx, scale in enumerate(self.scales):
            for rotation_idx, rotation in enumerate(self.rotations):
                current_iteration += 1
                
                # Print progress every 10% or every 10 iterations
                progress_pct = int((current_iteration / total_iterations) * 100)
                if progress_pct >= last_progress + 10 or current_iteration % 10 == 0:
                    print(f"         Progress: {current_iteration}/{total_iterations} ({progress_pct}%) - Scale {scale:.2f}, Rotation {rotation}°")
                    last_progress = progress_pct
                
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
                
                matches_found = len(locations[0])
                
                # Sanity check: if too many matches, the threshold is likely too low or template is bad
                if matches_found > 10000:
                    print(f"         ⚠️ WARNING: Found {matches_found} matches at scale {scale:.2f}, rotation {rotation}° - TOO MANY! Skipping this scale/rotation.")
                    print(f"            This usually means the template is too generic or threshold is too low.")
                    continue
                
                if matches_found > 0:
                    print(f"         ⭐ Found {matches_found} raw match(es) at scale {scale:.2f}, rotation {rotation}°")

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
        
        print(f"         ✅ Template matching complete: {len(all_detections)} raw detections")
        print(f"         🔄 Applying non-maximum suppression...")

        filtered_detections = self.non_max_suppression(all_detections)
        
        print(f"         ✅ NMS complete: {len(filtered_detections)} final detections")

        result_image = image.copy()
        if visualize:
            result_image = self.draw_detections(result_image, filtered_detections)

        return filtered_detections, result_image

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes on image.
        """
        result = image.copy()

        img_height, img_width = image.shape[:2]
        line_thickness = max(2, int((img_height + img_width) / 1000))

        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox

            color = (0, 255, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)

            label = f"#{idx+1} ({det.confidence:.2f})"
            font_scale = max(0.1, line_thickness * 0.05)
            font_thickness = max(1, line_thickness // 4)

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            cv2.rectangle(
                result,
                (x, y - text_h - baseline - 5),
                (x + text_w, y),
                color,
                -1,
            )

            cv2.putText(
                result,
                label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        return result


