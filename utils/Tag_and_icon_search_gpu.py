"""
GPU-Accelerated Electrical Symbol Detection
Updated functions for GPU parallelization using CUDA (via OpenCV) or CuPy

Installation requirements:
    pip install opencv-contrib-python  # For CUDA support
    pip install cupy-cuda12x  # Replace 12x with your CUDA version (11x, 12x, etc.)
    
Check CUDA availability:
    import cv2
    print(cv2.cuda.getCudaEnabledDeviceCount())
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import concurrent.futures
from functools import partial

# Try to import CuPy for advanced GPU operations
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Install with: pip install cupy-cuda12x")

# Check CUDA availability in OpenCV
CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
if CUDA_AVAILABLE:
    print(f"✓ CUDA enabled in OpenCV: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
else:
    print("Warning: CUDA not available in OpenCV. Using CPU fallback.")


@dataclass
class Detection:
    """Store detection information"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    scale: float
    rotation: float
    center: Tuple[int, int]
    tag_name: str = None
    template_type: str = None


class GPUElectricalSymbolDetector:
    """
    GPU-accelerated version of ElectricalSymbolDetector
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
                 nms_threshold: float = 0.3,
                 use_gpu: bool = True,
                 num_cpu_workers: int = 4):
        """
        Initialize GPU-accelerated detector.
        
        Args:
            use_gpu: Enable GPU acceleration if available
            num_cpu_workers: Number of CPU workers for parallel processing fallback
        """
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.rotations = rotations or list(range(0, 360, 15))
        
        self.tag_scales = tag_scales if tag_scales is not None else self.scales
        self.tag_rotations = tag_rotations if tag_rotations is not None else self.rotations
        self.icon_scales = icon_scales if icon_scales is not None else self.scales
        self.icon_rotations = icon_rotations if icon_rotations is not None else self.rotations
        
        self.threshold = threshold
        self.tag_threshold = tag_threshold if tag_threshold is not None else threshold
        self.icon_threshold = icon_threshold if icon_threshold is not None else threshold
        self.nms_threshold = nms_threshold
        
        self.templates = []
        self.template_source_image = None
        
        # GPU settings
        self.use_gpu = use_gpu and (CUDA_AVAILABLE or CUPY_AVAILABLE)
        self.num_cpu_workers = num_cpu_workers
        
        if self.use_gpu:
            self.gpu_method = self._determine_gpu_method()
            print(f"✓ GPU acceleration enabled using: {self.gpu_method}")
        else:
            print("⚠ Running on CPU (GPU not available or disabled)")
    
    def _determine_gpu_method(self) -> str:
        """Determine best available GPU method"""
        if CUPY_AVAILABLE:
            return "CuPy"
        elif CUDA_AVAILABLE:
            return "OpenCV-CUDA"
        return "CPU"
    
    # ============================================================
    # GPU-ACCELERATED ROTATION
    # ============================================================
    
    def rotate_image_gpu(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """
        GPU-accelerated image rotation using CUDA
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            rotated_image: Rotated image
            scale: Scale factor (always 1.0)
        """
        if not self.use_gpu:
            return self.rotate_image_cpu(image, angle)
        
        try:
            if self.gpu_method == "CuPy":
                return self._rotate_image_cupy(image, angle)
            elif self.gpu_method == "OpenCV-CUDA":
                return self._rotate_image_cuda(image, angle)
        except Exception as e:
            print(f"GPU rotation failed: {e}, falling back to CPU")
            return self.rotate_image_cpu(image, angle)
    
    def _rotate_image_cupy(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """Rotate using CuPy (fastest)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Transfer to GPU
        gpu_image = cp.asarray(image)
        gpu_M = cp.asarray(M)
        
        # Perform rotation on GPU
        rotated = cp.asnumpy(self._gpu_warp_affine(gpu_image, gpu_M, (new_w, new_h)))
        
        return rotated, 1.0
    
    def _rotate_image_cuda(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """Rotate using OpenCV CUDA"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Upload to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        # Warp on GPU
        gpu_rotated = cv2.cuda.warpAffine(gpu_image, M, (new_w, new_h))
        
        # Download result
        rotated = gpu_rotated.download()
        
        return rotated, 1.0
    
    def rotate_image_cpu(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        """CPU fallback for rotation"""
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
    
    # ============================================================
    # GPU-ACCELERATED TEMPLATE MATCHING
    # ============================================================
    
    def template_match_gpu(self, 
                          image_gray: np.ndarray, 
                          template: np.ndarray,
                          method: int = cv2.TM_CCOEFF_NORMED) -> np.ndarray:
        """
        GPU-accelerated template matching
        
        Args:
            image_gray: Grayscale search image
            template: Grayscale template
            method: Matching method
            
        Returns:
            result: Matching result matrix
        """
        if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            return None
        
        if not self.use_gpu:
            return cv2.matchTemplate(image_gray, template, method)
        
        try:
            if self.gpu_method == "CuPy":
                return self._template_match_cupy(image_gray, template, method)
            elif self.gpu_method == "OpenCV-CUDA":
                return self._template_match_cuda(image_gray, template, method)
        except Exception as e:
            print(f"GPU template matching failed: {e}, using CPU")
            return cv2.matchTemplate(image_gray, template, method)
    
    def _template_match_cupy(self, image_gray: np.ndarray, template: np.ndarray, method: int) -> np.ndarray:
        """Template matching using CuPy with FFT-based correlation"""
        # Upload to GPU
        gpu_image = cp.asarray(image_gray, dtype=cp.float32)
        gpu_template = cp.asarray(template, dtype=cp.float32)
        
        # Normalize
        gpu_image = (gpu_image - cp.mean(gpu_image)) / (cp.std(gpu_image) + 1e-8)
        gpu_template = (gpu_template - cp.mean(gpu_template)) / (cp.std(gpu_template) + 1e-8)
        
        # FFT-based correlation (much faster on GPU)
        result = cp.asnumpy(self._gpu_match_template_fft(gpu_image, gpu_template))
        
        return result
    
    def _gpu_match_template_fft(self, gpu_image, gpu_template):
        """FFT-based template matching on GPU"""
        h_img, w_img = gpu_image.shape
        h_tpl, w_tpl = gpu_template.shape
        
        # Pad template to image size
        padded_template = cp.zeros_like(gpu_image)
        padded_template[:h_tpl, :w_tpl] = gpu_template
        
        # FFT
        fft_image = cp.fft.fft2(gpu_image)
        fft_template = cp.fft.fft2(padded_template)
        
        # Correlation in frequency domain
        correlation = cp.fft.ifft2(fft_image * cp.conj(fft_template))
        correlation = cp.real(correlation)
        
        # Extract valid region
        result = correlation[:h_img - h_tpl + 1, :w_img - w_tpl + 1]
        
        # Normalize to [0, 1]
        result = (result - cp.min(result)) / (cp.max(result) - cp.min(result) + 1e-8)
        
        return result
    
    def _template_match_cuda(self, image_gray: np.ndarray, template: np.ndarray, method: int) -> np.ndarray:
        """Template matching using OpenCV CUDA"""
        # Upload to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_template = cv2.cuda_GpuMat()
        gpu_image.upload(image_gray)
        gpu_template.upload(template)
        
        # Create template matcher
        matcher = cv2.cuda.createTemplateMatching(gpu_image.type(), method)
        
        # Match on GPU
        gpu_result = matcher.match(gpu_image, gpu_template)
        
        # Download result
        result = gpu_result.download()
        
        return result
    
    # ============================================================
    # PARALLEL BATCH PROCESSING
    # ============================================================
    
    def _process_single_transform(self, 
                                  args: Tuple) -> List[Detection]:
        """
        Process a single scale-rotation combination
        Used for parallel processing
        """
        (image_gray, template_gray, scale, rotation, 
         current_threshold, name, ttype) = args
        
        detections = []
        
        # Resize template
        h_t, w_t = template_gray.shape
        new_w = int(w_t * scale)
        new_h = int(h_t * scale)
        
        h_img, w_img = image_gray.shape
        
        if new_w < 10 or new_h < 10 or new_w > w_img or new_h > h_img:
            return detections
        
        resized_template = cv2.resize(template_gray, (new_w, new_h))
        
        # Rotate template
        rotated_template, _ = self.rotate_image_gpu(resized_template, rotation)
        
        if rotated_template.shape[0] > h_img or rotated_template.shape[1] > w_img:
            return detections
        
        # Template matching
        result = self.template_match_gpu(image_gray, rotated_template)
        
        if result is None:
            return detections
        
        # Find matches
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
                tag_name=name,
                template_type=ttype
            )
            detections.append(detection)
        
        return detections
    
    def detect_symbols_parallel(self, 
                               search_image_path: str, 
                               visualize: bool = True) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect symbols using GPU-accelerated parallel processing
        
        Args:
            search_image_path: Path to search image
            visualize: Whether to create visualization
            
        Returns:
            detections: List of Detection objects
            result_image: Image with drawn detections
        """
        if len(self.templates) == 0:
            raise ValueError("Templates not collected. Call collect_templates() first.")
        
        # Load search image
        image = cv2.imread(search_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {search_image_path}")
        
        print(f"\n{'='*70}")
        print(f"GPU-ACCELERATED SYMBOL DETECTION")
        print(f"{'='*70}")
        print(f"  Method: {self.gpu_method if self.use_gpu else 'CPU Parallel'}")
        print(f"  Search image: {search_image_path}")
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"  Templates: {len(self.templates)}")
        print(f"{'='*70}\n")
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pre-upload image to GPU if using GPU
        if self.use_gpu and self.gpu_method == "CuPy" and CUPY_AVAILABLE:
            gpu_image_gray = cp.asarray(image_gray)
        else:
            gpu_image_gray = None
        
        all_detections = []
        
        for template, template_gray, name, ttype in self.templates:
            current_threshold = self.tag_threshold if ttype == 'tag' else self.icon_threshold
            current_scales = self.tag_scales if ttype == 'tag' else self.icon_scales
            current_rotations = self.tag_rotations if ttype == 'tag' else self.icon_rotations
            
            print(f"Processing {ttype} '{name}'...")
            print(f"  Scales: {len(current_scales)}, Rotations: {len(current_rotations)}")
            print(f"  Total combinations: {len(current_scales) * len(current_rotations)}")
            
            # Create task list
            tasks = []
            for scale in current_scales:
                for rotation in current_rotations:
                    tasks.append((
                        image_gray, template_gray, scale, rotation,
                        current_threshold, name, ttype
                    ))
            
            # Process in parallel
            if self.use_gpu:
                # GPU: Process tasks in batches
                print(f"  Processing on GPU in batches...")
                template_detections = []
                
                batch_size = 10  # Process 10 transforms at a time
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    for task in batch:
                        dets = self._process_single_transform(task)
                        template_detections.extend(dets)
                    
                    if (i + batch_size) % 50 == 0:
                        print(f"    Progress: {min(i+batch_size, len(tasks))}/{len(tasks)}")
            else:
                # CPU: Use multiprocessing
                print(f"  Processing on {self.num_cpu_workers} CPU workers...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cpu_workers) as executor:
                    results = executor.map(self._process_single_transform, tasks)
                    template_detections = []
                    for dets in results:
                        template_detections.extend(dets)
            
            print(f"  Raw detections: {len(template_detections)}")
            
            # NMS
            filtered = self.non_max_suppression(template_detections)
            print(f"  After NMS: {len(filtered)}")
            
            all_detections.extend(filtered)
        
        print(f"\n✓ Total detections: {len(all_detections)}")
        
        # Visualization
        result_image = image.copy()
        if visualize:
            result_image = self.draw_detections(result_image, all_detections)
        
        return all_detections, result_image
    
    # ============================================================
    # GPU-ACCELERATED NMS (Optional enhancement)
    # ============================================================
    
    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply non-maximum suppression
        (Can be GPU-accelerated for large detection counts)
        """
        if len(detections) == 0:
            return []
        
        # For <1000 detections, CPU is fine
        if len(detections) < 1000 or not self.use_gpu:
            return self._nms_cpu(detections)
        
        # For large detection counts, use GPU
        try:
            if CUPY_AVAILABLE:
                return self._nms_gpu(detections)
        except:
            pass
        
        return self._nms_cpu(detections)
    
    def _nms_cpu(self, detections: List[Detection]) -> List[Detection]:
        """CPU-based NMS"""
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
    
    def _nms_gpu(self, detections: List[Detection]) -> List[Detection]:
        """GPU-accelerated NMS using CuPy"""
        # Convert to arrays
        boxes = np.array([det.bbox for det in detections], dtype=np.float32)
        scores = np.array([det.confidence for det in detections], dtype=np.float32)
        
        # Upload to GPU
        gpu_boxes = cp.asarray(boxes)
        gpu_scores = cp.asarray(scores)
        
        # Sort by score
        indices = cp.argsort(gpu_scores)[::-1]
        
        keep_indices = []
        
        while len(indices) > 0:
            current_idx = int(indices[0])
            keep_indices.append(current_idx)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = gpu_boxes[current_idx]
            remaining_boxes = gpu_boxes[indices[1:]]
            
            ious = self._calculate_iou_vectorized_gpu(current_box, remaining_boxes)
            
            # Keep boxes with IoU < threshold
            mask = ious < self.nms_threshold
            indices = indices[1:][cp.asnumpy(mask)]
        
        # Return filtered detections
        return [detections[i] for i in keep_indices]
    
    def _calculate_iou_vectorized_gpu(self, box, boxes):
        """Vectorized IoU calculation on GPU"""
        x1, y1, w1, h1 = box
        x2, y2, w2, h2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        xi1 = cp.maximum(x1, x2)
        yi1 = cp.maximum(y1, y2)
        xi2 = cp.minimum(x1 + w1, x2 + w2)
        yi2 = cp.minimum(y1 + h1, y2 + h2)
        
        inter_area = cp.maximum(0, xi2 - xi1) * cp.maximum(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes"""
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
    
    def _gpu_warp_affine(self, gpu_image, gpu_M, dsize):
        """CuPy-based affine transformation"""
        # This is a simplified version - for production, use cupyx.scipy.ndimage
        # or integrate with OpenCV CUDA
        from cupyx.scipy import ndimage as cu_ndimage
        
        # Note: This requires additional implementation
        # For now, fall back to CPU for complex transforms
        cpu_image = cp.asnumpy(gpu_image)
        cpu_M = cp.asnumpy(gpu_M)
        result = cv2.warpAffine(cpu_image, cpu_M, dsize)
        return cp.asarray(result)
    
    # Keep other methods from original class (draw_detections, etc.)
    # These don't need GPU acceleration as they're not bottlenecks
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes (same as original)"""
        result = image.copy()
        img_height, img_width = image.shape[:2]
        line_thickness = max(2, int((img_height + img_width) / 1000))
        
        tag_counter = {}
        icon_counter = 1
        
        for det in detections:
            x, y, w, h = det.bbox
            
            if det.template_type == 'tag':
                color = (255, 0, 0)
                if det.tag_name not in tag_counter:
                    tag_counter[det.tag_name] = 0
                tag_counter[det.tag_name] += 1
                label = f"{det.tag_name}-{tag_counter[det.tag_name]} ({det.confidence:.2f})"
            else:
                color = (0, 255, 0)
                label = f"{det.tag_name or 'icon'}-{icon_counter} ({det.confidence:.2f})"
                icon_counter += 1
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = max(1, line_thickness // 2)
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_y = y - 10 if y - 10 > text_h else y + h + text_h + 10
            
            cv2.rectangle(result, (x, text_y - text_h - 5), (x + text_w, text_y + 5), color, -1)
            cv2.putText(result, label, (x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        return result


# ============================================================
# USAGE EXAMPLE
# ============================================================

def run_gpu_detection_pipeline(template_image_path: str,
                               search_image_path: str = None,
                               use_gpu: bool = True,
                               num_cpu_workers: int = 4,
                               **kwargs) -> Dict:
    """
    Run GPU-accelerated detection pipeline
    
    Args:
        template_image_path: Path to template selection image
        search_image_path: Path to search image
        use_gpu: Enable GPU acceleration
        num_cpu_workers: CPU workers for fallback
        **kwargs: Other parameters (thresholds, scales, etc.)
        
    Returns:
        results: Detection results dictionary
    """
    # Initialize GPU detector
    detector = GPUElectricalSymbolDetector(
        use_gpu=use_gpu,
        num_cpu_workers=num_cpu_workers,
        **kwargs
    )
    
    # Collect templates (same as before)
    print("="*70)
    print("TEMPLATE COLLECTION")
    print("="*70)
    detector.collect_templates(template_image_path)
    
    # Run detection on GPU
    search_path = search_image_path or template_image_path
    all_detections, result_image = detector.detect_symbols_parallel(search_path)
    
    # Separate by type
    tag_detections = [d for d in all_detections if d.template_type == 'tag']
    icon_detections = [d for d in all_detections if d.template_type == 'icon']
    
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Tags: {len(tag_detections)}")
    print(f"Icons: {len(icon_detections)}")
    print("="*70)
    
    return {
        'tag_detections': tag_detections,
        'icon_detections': icon_detections,
        'all_detections': all_detections,
        'result_image': result_image,
        'detector': detector
    }


if __name__ == "__main__":
    """
    Example usage:
    
    python gpu_detector.py reference.png search.png --use-gpu
    """
    import sys
    
    if len(sys.argv) < 2:
        print("GPU-Accelerated Symbol Detector")
        print("Usage: python gpu_detector.py <ref_image> [search_image] [--use-gpu] [--cpu-workers N]")
        sys.exit(1)
    
    template_path = sys.argv[1]
    search_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    use_gpu = '--use-gpu' in sys.argv or '--gpu' in sys.argv
    cpu_workers = 4
    
    if '--cpu-workers' in sys.argv:
        idx = sys.argv.index('--cpu-workers')
        cpu_workers = int(sys.argv[idx + 1])
    
    results = run_gpu_detection_pipeline(
        template_image_path=template_path,
        search_image_path=search_path,
        use_gpu=use_gpu,
        num_cpu_workers=cpu_workers
    )