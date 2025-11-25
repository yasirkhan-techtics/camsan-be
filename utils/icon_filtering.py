"""
Filter Detection Results by Scale and Confidence
Use this after running the detection pipeline to refine results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================================
# STEP 1: Filter detections based on scale and confidence
# =============================================================================

def filter_detections(detections, 
                     min_confidence=None, 
                     max_confidence=None,
                     min_scale=None, 
                     max_scale=None,
                     allowed_scales=None):
    """
    Filter detections based on confidence and scale criteria.
    
    Args:
        detections: List of Detection objects from pipeline
        min_confidence: Minimum confidence threshold (0-1)
        max_confidence: Maximum confidence threshold (0-1)
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        allowed_scales: List of specific scales to keep (e.g., [1.0, 1.1])
        
    Returns:
        filtered_detections: List of filtered Detection objects
    """
    filtered = []
    
    for det in detections:
        # Check confidence bounds
        if min_confidence is not None and det.confidence < min_confidence:
            continue
        if max_confidence is not None and det.confidence > max_confidence:
            continue
        
        # Check scale bounds
        if min_scale is not None and det.scale < min_scale:
            continue
        if max_scale is not None and det.scale > max_scale:
            continue
        
        # Check specific allowed scales
        if allowed_scales is not None and det.scale not in allowed_scales:
            continue
        
        filtered.append(det)
    
    return filtered


def draw_filtered_detections(image, filtered_detections):
    """
    Draw bounding boxes for filtered detections.
    
    Args:
        image: Original search image (BGR format)
        filtered_detections: List of filtered Detection objects
        
    Returns:
        result_image: Image with drawn bounding boxes
    """
    result = image.copy()
    
    # Calculate adaptive line thickness
    img_height, img_width = image.shape[:2]
    line_thickness = max(2, int((img_height + img_width) / 1000))
    
    for idx, det in enumerate(filtered_detections):
        x, y, w, h = det.bbox
        
        # Draw rectangle - Green color
        color = (0, 255, 0)  # BGR
        cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)
        
        # Draw center point - Red color
        # point_size = max(1, line_thickness)
        # cv2.circle(result, det.center, point_size, (0, 0, 255), -1)
        
        # Add label with confidence and scale
        label = f"#{idx+1} C:{det.confidence:.2f} S:{det.scale:.2f}x"
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
        
        # Draw text in black
        cv2.putText(result, label, (x, y - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    return result


def analyze_detection_distribution(detections):
    """
    Analyze the distribution of scales and confidences in detections.
    
    Args:
        detections: List of Detection objects
        
    Returns:
        stats: Dictionary with distribution statistics
    """
    if len(detections) == 0:
        return {
            'scales': [],
            'confidences': [],
            'scale_counts': {},
            'confidence_range': (0, 0)
        }
    
    scales = [det.scale for det in detections]
    confidences = [det.confidence for det in detections]
    
    # Count detections per scale
    scale_counts = {}
    for scale in scales:
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
    
    stats = {
        'scales': scales,
        'confidences': confidences,
        'scale_counts': scale_counts,
        'unique_scales': sorted(set(scales)),
        'confidence_range': (min(confidences), max(confidences)),
        'avg_confidence': np.mean(confidences),
        'scale_range': (min(scales), max(scales))
    }
    
    return stats


def visualize_detection_stats(detections, title="Detection Statistics"):
    """
    Visualize detection statistics with histograms.
    
    Args:
        detections: List of Detection objects
        title: Plot title
    """
    stats = analyze_detection_distribution(detections)
    
    if len(detections) == 0:
        print("No detections to visualize")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Scale distribution
    scales = stats['scales']
    axes[0].hist(scales, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Scale Factor')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Scale Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence distribution
    confidences = stats['confidences']
    axes[1].hist(confidences, bins=20, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scale vs Confidence scatter
    axes[2].scatter(scales, confidences, alpha=0.6, c='coral', edgecolors='black')
    axes[2].set_xlabel('Scale Factor')
    axes[2].set_ylabel('Confidence')
    axes[2].set_title('Scale vs Confidence')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total detections: {len(detections)}")
    print(f"Confidence range: {stats['confidence_range'][0]:.3f} - {stats['confidence_range'][1]:.3f}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Scale range: {stats['scale_range'][0]:.2f} - {stats['scale_range'][1]:.2f}")
    print(f"\nDetections per scale:")
    for scale in sorted(stats['scale_counts'].keys()):
        count = stats['scale_counts'][scale]
        print(f"  Scale {scale:.2f}x: {count} detections")
    print(f"{'='*60}\n")


# =============================================================================
# EXAMPLE USAGE IN JUPYTER NOTEBOOK
# =============================================================================

"""

# -------------------------------------------------------------------
# STEP 1: Analyze original detections
# -------------------------------------------------------------------
print("Original Detection Results:")
visualize_detection_stats(detections, title="Original Detection Statistics")

# -------------------------------------------------------------------
# STEP 2: Filter detections based on your criteria
# -------------------------------------------------------------------

# Example 1: Filter by confidence only
filtered_detections_1 = filter_detections(
    detections,
    min_confidence=0.85,  # Keep only high confidence detections
    max_confidence=None
)

print(f"After confidence filter (>0.85): {len(filtered_detections_1)} detections")

# -------------------------------------------------------------------

# Example 2: Filter by scale only
filtered_detections_2 = filter_detections(
    detections,
    min_scale=0.9,
    max_scale=1.1
)

print(f"After scale filter (0.9-1.1): {len(filtered_detections_2)} detections")

# -------------------------------------------------------------------

# Example 3: Filter by both confidence AND scale
filtered_detections_3 = filter_detections(
    detections,
    min_confidence=0.80,
    max_confidence=None,
    min_scale=0.9,
    max_scale=1.2
)

print(f"After combined filter: {len(filtered_detections_3)} detections")

# -------------------------------------------------------------------

# Example 4: Keep only specific scales
filtered_detections_4 = filter_detections(
    detections,
    allowed_scales=[1.0, 1.1],  # Only keep these exact scales
    min_confidence=0.75
)

print(f"After specific scale filter: {len(filtered_detections_4)} detections")

# -------------------------------------------------------------------
# STEP 3: Visualize filtered results
# -------------------------------------------------------------------

# Choose which filtered set to visualize (e.g., filtered_detections_3)
final_filtered = filtered_detections_3

# Analyze filtered detections
print("\nFiltered Detection Results:")
visualize_detection_stats(final_filtered, title="Filtered Detection Statistics")

# -------------------------------------------------------------------
# STEP 4: Draw filtered detections on original image
# -------------------------------------------------------------------

# Load the original search image
search_image_path = results['search_source']
original_image = cv2.imread(search_image_path)

# Draw filtered detections
filtered_result_image = draw_filtered_detections(original_image, final_filtered)

# Display comparison
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Original results (all detections)
axes[0].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Original Results ({len(detections)} detections)', fontsize=14)
axes[0].axis('off')

# Filtered results
axes[1].imshow(cv2.cvtColor(filtered_result_image, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Filtered Results ({len(final_filtered)} detections)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# STEP 5: Save filtered results
# -------------------------------------------------------------------

import os

# Create output path for filtered results
output_dir = os.path.dirname(search_image_path) or '.'
output_folder = os.path.join(output_dir, 'output')
os.makedirs(output_folder, exist_ok=True)

# Save filtered result
filtered_output_filename = os.path.basename(search_image_path).rsplit('.', 1)[0] + '_filtered.jpg'
filtered_output_path = os.path.join(output_folder, filtered_output_filename)
cv2.imwrite(filtered_output_path, filtered_result_image)

print(f"\n✓ Filtered results saved to: {filtered_output_path}")

# -------------------------------------------------------------------
# STEP 6: Print detailed filtered detection info
# -------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"FILTERED DETECTION DETAILS")
print(f"{'='*60}")
print(f"Total filtered detections: {len(final_filtered)}")
print()

for idx, det in enumerate(final_filtered):
    print(f"Detection #{idx+1}:")
    print(f"  Position: {det.center}")
    print(f"  Bounding Box: {det.bbox}")
    print(f"  Confidence: {det.confidence:.3f}")
    print(f"  Scale: {det.scale:.2f}x")
    print(f"  Rotation: {det.rotation}°")
    print()

print(f"{'='*60}\n")
"""