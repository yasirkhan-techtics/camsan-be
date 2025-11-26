import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    scale: float
    rotation: int
    center: Tuple[int, int]
    text_label: str = None


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def assign_labels_to_icons_dynamic(
    icon_detections: List[Detection],
    text_detections: List[Detection],
    image_shape: Tuple[int, int],
    max_distance_ratio: float = 0.25,  # Max distance as ratio of image diagonal
) -> Tuple[List[str], Dict]:
    """
    Match icons to labels using nearest-neighbor assignment with greedy matching.
    
    Args:
        icon_detections: List of detected icons
        text_detections: List of detected labels/tags
        image_shape: (height, width) of the page image
        max_distance_ratio: Maximum allowed distance as a ratio of image diagonal
    
    Returns:
        Tuple of (list of assigned labels, debug info dict)
    """
    debug_info = {
        "assignments": [],
        "distances": [],
        "rejected_reason": [],
        "learned_threshold": None,
        "learned_ratio": None,
        "all_candidates": [],
    }

    img_height, img_width = image_shape
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    
    print(f"[MATCHER] Image shape: {image_shape}, diagonal: {img_diagonal:.1f}")
    print(f"[MATCHER] Icons: {len(icon_detections)}, Labels: {len(text_detections)}")

    if len(text_detections) == 0:
        print("[MATCHER] No text detections - returning all unknown")
        return ["unknown"] * len(icon_detections), debug_info

    # Build distance matrix
    distance_matrix = np.zeros((len(icon_detections), len(text_detections)))

    for i, icon in enumerate(icon_detections):
        for j, text in enumerate(text_detections):
            dist = calculate_distance(icon.center, text.center)
            distance_matrix[i, j] = dist

    # Use a more generous threshold - 25% of image diagonal
    # This allows matching icons and labels that are reasonably close
    max_threshold = max_distance_ratio * img_diagonal
    
    # Also compute a data-driven threshold if we have enough data
    min_distances = np.min(distance_matrix, axis=1)
    if len(min_distances) > 2:
        q3 = np.percentile(min_distances, 75)
        iqr = np.percentile(min_distances, 75) - np.percentile(min_distances, 25)
        data_threshold = q3 + 1.5 * iqr
        # Use the larger of data-driven or fixed threshold
        learned_threshold = max(data_threshold, max_threshold * 0.5)
        # But cap at max_threshold
        learned_threshold = min(learned_threshold, max_threshold)
    else:
        learned_threshold = max_threshold

    debug_info["learned_threshold"] = float(learned_threshold)
    print(f"[MATCHER] Using threshold: {learned_threshold:.1f} (max allowed: {max_threshold:.1f})")

    # Greedy matching: sort all (icon, label) pairs by distance and assign greedily
    used_texts = set()
    assigned_icons = set()
    assignments = []

    # Create all candidates
    candidates = []
    for i in range(len(icon_detections)):
        for j in range(len(text_detections)):
            dist = distance_matrix[i, j]
            candidates.append({
                "icon_idx": i,
                "text_idx": j,
                "distance": float(dist),
                "within_threshold": dist <= learned_threshold,
            })
    
    # Sort by distance (closest first)
    candidates.sort(key=lambda x: x["distance"])
    debug_info["all_candidates"] = candidates[:50]  # Store first 50 for debugging

    # Greedy assignment
    for cand in candidates:
        icon_idx = cand["icon_idx"]
        text_idx = cand["text_idx"]
        dist = cand["distance"]
        
        if icon_idx in assigned_icons or text_idx in used_texts:
            continue

        if dist > learned_threshold:
            debug_info["rejected_reason"].append(
                f"icon {icon_idx} -> label {text_idx}: distance {dist:.1f} > threshold {learned_threshold:.1f}"
            )
            continue

        # Accept this match
        assignments.append((icon_idx, text_idx, dist))
        assigned_icons.add(icon_idx)
        used_texts.add(text_idx)
        print(f"[MATCHER] Matched icon {icon_idx} -> label {text_idx} (distance: {dist:.1f})")

    # Build result
    icon_labels = ["unknown"] * len(icon_detections)

    for icon_idx, text_idx, dist in assignments:
        label = text_detections[text_idx].text_label
        icon_labels[icon_idx] = label if label else f"label_{text_idx}"

        debug_info["assignments"].append(
            {
                "icon_idx": icon_idx,
                "text_idx": text_idx,
                "distance": float(dist),
                "label": label,
            }
        )
        debug_info["distances"].append(float(dist))

    # Record unmatched icons
    for icon_idx, label in enumerate(icon_labels):
        if label == "unknown":
            min_dist = np.min(distance_matrix[icon_idx, :])
            debug_info["rejected_reason"].append(
                f"icon {icon_idx}: unmatched (closest label at distance {min_dist:.1f})"
            )

    print(f"[MATCHER] Total matches: {len(assignments)}/{len(icon_detections)} icons")
    return icon_labels, debug_info


