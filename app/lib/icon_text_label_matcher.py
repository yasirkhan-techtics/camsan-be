import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    scale: float
    rotation: int
    center: Tuple[int, int]
    text_label: str = None


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def assign_labels_to_icons_dynamic(
    icon_detections: List[Detection],
    text_detections: List[Detection],
    image_shape: Tuple[int, int]
) -> Tuple[List[str], Dict]:
    """
    Assign text labels to icons using FULLY DYNAMIC one-to-one matching.
    
    STRICT CONSTRAINT: Each text label can only be assigned to ONE icon.
    This ensures len(assigned_labels) <= len(text_detections)
    
    Strategy:
    1. Build distance matrix between all icons and texts
    2. Use statistical analysis (median, IQR) to learn thresholds
    3. Apply greedy one-to-one matching with learned constraints
    4. Use ratio test dynamically based on distance distribution
    
    Args:
        icon_detections: List of icon Detection objects
        text_detections: List of text Detection objects
        image_shape: (height, width) of the image
    
    Returns:
        icon_labels: List of labels for each icon ("unknown" if no match)
        debug_info: Dictionary with matching statistics
    """
    icon_labels = []
    debug_info = {
        'assignments': [],
        'distances': [],
        'rejected_reason': [],
        'learned_threshold': None,
        'learned_ratio': None,
        'all_candidates': [],
    }
    
    img_height, img_width = image_shape
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    
    print(f"[MATCHER] Image shape: {img_width}x{img_height}, diagonal: {img_diagonal:.1f}px")
    print(f"[MATCHER] Icons: {len(icon_detections)}, Texts: {len(text_detections)}")
    print(f"[MATCHER] ONE-TO-ONE MATCHING: Maximum {len(text_detections)} icons can be labeled")
    print("-" * 70)
    
    if len(text_detections) == 0:
        print("[MATCHER] No text detections - all icons marked as unknown")
        return ["unknown"] * len(icon_detections), debug_info
    
    # STEP 1: Build complete distance matrix
    print("\n[MATCHER] STEP 1: Building distance matrix...")
    distance_matrix = np.zeros((len(icon_detections), len(text_detections)))
    all_distances = []
    
    for i, icon in enumerate(icon_detections):
        for j, text in enumerate(text_detections):
            dist = calculate_distance(icon.center, text.center)
            distance_matrix[i, j] = dist
            all_distances.append(dist)
    
    all_distances = np.array(all_distances)
    
    # Store candidates for debug
    candidates_for_debug = []
    for i in range(len(icon_detections)):
        for j in range(len(text_detections)):
            candidates_for_debug.append({
                "icon_idx": i,
                "text_idx": j,
                "distance": float(distance_matrix[i, j]),
            })
    candidates_for_debug.sort(key=lambda x: x["distance"])
    debug_info["all_candidates"] = candidates_for_debug[:50]
    
    # STEP 2: Learn threshold using statistical analysis
    print("\n[MATCHER] STEP 2: Learning distance threshold from data...")
    min_distances = np.min(distance_matrix, axis=1)  # Closest text for each icon
    
    # Use median + IQR for robust threshold (resistant to outliers)
    q1 = np.percentile(min_distances, 25)
    q3 = np.percentile(min_distances, 75)
    iqr = q3 - q1
    median = np.median(min_distances)
    
    # Dynamic threshold: median + 1.5*IQR (standard outlier detection)
    learned_threshold = median + 1.5 * iqr
    
    # Also consider image scale (as fallback constraint)
    scale_based_max = 0.15 * img_diagonal  # 15% of diagonal as absolute max
    learned_threshold = min(learned_threshold, scale_based_max)
    
    debug_info['learned_threshold'] = float(learned_threshold)
    
    print(f"[MATCHER] Distance statistics (closest text per icon):")
    print(f"[MATCHER]   Min: {np.min(min_distances):.1f}px")
    print(f"[MATCHER]   Q1: {q1:.1f}px, Median: {median:.1f}px, Q3: {q3:.1f}px")
    print(f"[MATCHER]   Max: {np.max(min_distances):.1f}px")
    print(f"[MATCHER]   IQR: {iqr:.1f}px")
    print(f"[MATCHER] → LEARNED THRESHOLD: {learned_threshold:.1f}px")
    
    # STEP 3: Learn uniqueness ratio from data
    print("\n[MATCHER] STEP 3: Learning uniqueness ratio from data...")
    ratios = []
    for i in range(len(icon_detections)):
        sorted_dists = np.sort(distance_matrix[i, :])
        if len(sorted_dists) >= 2 and sorted_dists[1] > 0:
            ratio = sorted_dists[0] / sorted_dists[1]
            ratios.append(ratio)
    
    if len(ratios) > 0:
        # Use median ratio + buffer as threshold
        median_ratio = np.median(ratios)
        learned_ratio = min(median_ratio + 0.15, 0.75)  # Add buffer, cap at 0.75
    else:
        learned_ratio = 0.7  # Fallback if only one text
    
    debug_info['learned_ratio'] = float(learned_ratio)
    print(f"[MATCHER] Uniqueness ratios: median={np.median(ratios) if ratios else 'N/A':.3f}")
    print(f"[MATCHER] → LEARNED RATIO THRESHOLD: {learned_ratio:.3f}")
    
    # STEP 4: Greedy bipartite matching with learned constraints (ONE-TO-ONE)
    print("\n[MATCHER] STEP 4: Performing ONE-TO-ONE greedy matching...")
    print("-" * 70)
    
    used_texts = set()
    assigned_icons = set()
    assignments = []
    
    # Create list of (distance, icon_idx, text_idx) and sort
    candidates = []
    for i in range(len(icon_detections)):
        for j in range(len(text_detections)):
            candidates.append((distance_matrix[i, j], i, j))
    candidates.sort(key=lambda x: x[0])  # Sort by distance (closest first)
    
    # Greedy matching: assign closest valid pairs first
    # CRITICAL: Each text can only be used ONCE
    for dist, icon_idx, text_idx in candidates:
        if icon_idx in assigned_icons or text_idx in used_texts:
            continue  # Already assigned
        
        # Check distance constraint
        if dist > learned_threshold:
            debug_info['rejected_reason'].append(
                f"icon {icon_idx} -> label {text_idx}: distance {dist:.1f} > threshold {learned_threshold:.1f}"
            )
            continue  # Too far
        
        # Check uniqueness (is this significantly better than other available options?)
        available_texts = [k for k in range(len(text_detections)) 
                          if k not in used_texts and k != text_idx]
        
        if available_texts:
            other_dists = [distance_matrix[icon_idx, k] for k in available_texts]
            second_best = min(other_dists)
            ratio = dist / second_best if second_best > 0 else 0
            if ratio > learned_ratio:
                debug_info['rejected_reason'].append(
                    f"icon {icon_idx} -> label {text_idx}: ratio {ratio:.3f} > threshold {learned_ratio:.3f} (not unique enough)"
                )
                continue  # Not unique enough
        
        # Valid assignment - mark both icon and text as used
        assignments.append((icon_idx, text_idx, dist))
        assigned_icons.add(icon_idx)
        used_texts.add(text_idx)
        
        # Stop if all texts are assigned
        if len(used_texts) == len(text_detections):
            break
    
    # STEP 5: Build final labels
    print(f"\n[MATCHER] STEP 5: Finalizing assignments (matched {len(assignments)} pairs)...")
    icon_labels = ["unknown"] * len(icon_detections)
    
    for icon_idx, text_idx, dist in assignments:
        label = text_detections[text_idx].text_label
        icon_labels[icon_idx] = label if label else f"label_{text_idx}"
        
        debug_info['assignments'].append({
            'icon_idx': icon_idx,
            'text_idx': text_idx,
            'distance': float(dist),
            'label': label
        })
        debug_info['distances'].append(float(dist))
        
        icon = icon_detections[icon_idx]
        print(f"[MATCHER] ✓ Icon #{icon_idx+1} at {icon.center} → '{label}' (dist: {dist:.1f}px)")
    
    # Print unassigned icons with reasons
    print()
    for icon_idx, label in enumerate(icon_labels):
        if label == "unknown":
            icon = icon_detections[icon_idx]
            
            # Find why it wasn't assigned
            available_texts_at_decision = []
            for j in range(len(text_detections)):
                # Check if this text was still available when we could have assigned it
                was_available = True
                for assigned_icon, assigned_text, _ in assignments:
                    if assigned_text == j and assigned_icon in assigned_icons:
                        # This text was assigned to another icon
                        was_available = False
                        break
                if was_available:
                    available_texts_at_decision.append((distance_matrix[icon_idx, j], j))
            
            if not available_texts_at_decision:
                reason = "all texts already assigned to closer icons"
            else:
                closest_dist, closest_text_idx = min(available_texts_at_decision)
                if closest_dist > learned_threshold:
                    reason = f"closest available text too far ({closest_dist:.1f}px > {learned_threshold:.1f}px)"
                else:
                    reason = f"not unique enough (ambiguous assignment)"
            
            debug_info['rejected_reason'].append(f"icon {icon_idx}: {reason}")
            print(f"[MATCHER] ✗ Icon #{icon_idx+1} at {icon.center} → unknown ({reason})")
    
    # Print summary statistics
    print("-" * 70)
    if debug_info['distances']:
        print(f"[MATCHER] Successfully assigned: {len(debug_info['distances'])}/{len(icon_detections)}")
        print(f"[MATCHER] Average assignment distance: {np.mean(debug_info['distances']):.1f}px")
        print(f"[MATCHER] Distance std dev: {np.std(debug_info['distances']):.1f}px")
    else:
        print("[MATCHER] No successful assignments")
    
    print(f"[MATCHER] Unknown icons: {icon_labels.count('unknown')}/{len(icon_detections)}")
    
    return icon_labels, debug_info
