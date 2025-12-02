import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    scale: float
    rotation: int
    center: Tuple[int, int]
    text_label: str = None


class AdvancedIconTagMatcher:
    """
    Multi-technique voting system for icon-tag matching.
    Minimizes false positives through ensemble methods.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        high_density_threshold: float = 0.70,
        min_votes_required: int = 4,
        enable_debug: bool = True
    ):
        """
        Args:
            confidence_threshold: Minimum vote score for assignment (0-1)
            high_density_threshold: Stricter threshold for dense regions
            min_votes_required: Minimum number of techniques that must agree
            enable_debug: Print detailed matching information
        """
        self.confidence_threshold = confidence_threshold
        self.high_density_threshold = high_density_threshold
        self.min_votes_required = min_votes_required
        self.enable_debug = enable_debug

        # Technique weights (tune these based on your data)
        self.weights = {
            'distance': 1.5,
            'bbox_proximity': 1.3,
            'exclusion_zone': 1.4,
            'alignment': 1.0,
            'density_context': 1.1,
            'directional': 0.8,  # Lower weight since tags can be anywhere
        }

    def _calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _bbox_proximity_score(self, icon: Detection, tag: Detection) -> float:
        """
        Score based on minimum edge-to-edge distance between bounding boxes.
        Tags close to icon edges score higher.
        """
        ix, iy, iw, ih = icon.bbox
        tx, ty, tw, th = tag.bbox

        # Calculate edge-to-edge distances
        # Horizontal gap
        if tx + tw < ix:  # Tag left of icon
            h_gap = ix - (tx + tw)
        elif tx > ix + iw:  # Tag right of icon
            h_gap = tx - (ix + iw)
        else:  # Overlapping horizontally
            h_gap = 0

        # Vertical gap
        if ty + th < iy:  # Tag above icon
            v_gap = iy - (ty + th)
        elif ty > iy + ih:  # Tag below icon
            v_gap = ty - (iy + ih)
        else:  # Overlapping vertically
            v_gap = 0

        # Minimum gap distance
        min_gap = min(h_gap, v_gap) if h_gap > 0 and v_gap > 0 else max(h_gap, v_gap)

        # Score: closer edge = higher score
        # Exponential decay with distance
        return np.exp(-min_gap / 80.0)

    def _exclusion_zone_score(
        self,
        icon: Detection,
        tag: Detection,
        all_icons: List[Detection]
    ) -> float:
        """
        Penalize if another icon lies between this icon and tag.
        """
        ix, iy = icon.center
        tx, ty = tag.center

        # Vector from icon to tag
        dx = tx - ix
        dy = ty - iy
        dist = self._calculate_distance(icon.center, tag.center)

        if dist < 1:
            return 1.0

        # Check each other icon
        for other in all_icons:
            if other == icon:
                continue

            ox, oy = other.center

            # Vector from icon to other
            to_other_x = ox - ix
            to_other_y = oy - iy

            # Project other onto icon-tag line
            # dot product / squared distance
            t = (to_other_x * dx + to_other_y * dy) / (dist * dist)

            # Check if projection is between icon and tag
            if 0.2 < t < 0.8:  # Other icon is "between" them
                # Calculate perpendicular distance
                proj_x = ix + t * dx
                proj_y = iy + t * dy
                perp_dist = self._calculate_distance((ox, oy), (proj_x, proj_y))

                # If other icon is very close to the line, heavy penalty
                if perp_dist < 40:
                    return 0.2

        return 1.0

    def _alignment_score(self, icon: Detection, tag: Detection, tolerance: int = 15) -> float:
        """
        Check if tag and icon are roughly aligned horizontally or vertically.
        """
        dx = abs(tag.center[0] - icon.center[0])
        dy = abs(tag.center[1] - icon.center[1])

        # Perfect alignment
        if dx < tolerance or dy < tolerance:
            return 1.0

        # Partial alignment - exponential decay
        min_offset = min(dx, dy)
        return np.exp(-min_offset / (tolerance * 3))

    def _local_density_score(
        self,
        icon: Detection,
        tag: Detection,
        all_icons: List[Detection],
        all_tags: List[Detection],
        radius: int = 150
    ) -> float:
        """
        In high-density regions, be more conservative.
        Returns a multiplier that adjusts threshold requirements.
        """
        # Count nearby icons
        nearby_icons = sum(
            1 for other in all_icons
            if other != icon and self._calculate_distance(icon.center, other.center) < radius
        )

        # Count nearby tags
        nearby_tags = sum(
            1 for other in all_tags
            if other != tag and self._calculate_distance(tag.center, other.center) < radius
        )

        density = (nearby_icons + nearby_tags) / 2.0

        # Higher density = require stronger evidence
        if density > 5:
            return 0.65
        elif density > 3:
            return 0.80
        else:
            return 1.0

    def _directional_score(self, icon: Detection, tag: Detection) -> float:
        """
        Score based on direction from icon to tag.
        Since tags can be anywhere, use a lenient scoring.
        """
        dx = tag.center[0] - icon.center[0]
        dy = tag.center[1] - icon.center[1]
        dist = self._calculate_distance(icon.center, tag.center)

        if dist < 1:
            return 1.0

        # Just check if tag is not too far in any direction
        # Give reasonable score to all directions
        # Penalize only extreme distances
        max_offset = max(abs(dx), abs(dy))

        # Gentle penalty for very far tags
        return np.exp(-max_offset / 500.0)

    def _compute_vote_matrix(
        self,
        icons: List[Detection],
        tags: List[Detection],
        img_diagonal: float
    ) -> np.ndarray:
        """
        Compute voting matrix using all techniques.
        Returns: (num_icons, num_tags) array of vote scores
        """
        n_icons = len(icons)
        n_tags = len(tags)

        if n_icons == 0 or n_tags == 0:
            return np.zeros((n_icons, n_tags))

        votes = np.zeros((n_icons, n_tags))

        # Pre-compute all distances
        icon_centers = np.array([icon.center for icon in icons])
        tag_centers = np.array([tag.center for tag in tags])
        dist_matrix = cdist(icon_centers, tag_centers, metric='euclidean')

        # Normalize distance scores
        max_reasonable_dist = img_diagonal * 0.15

        for i, icon in enumerate(icons):
            for j, tag in enumerate(tags):
                scores = {}

                # Technique 1: Distance (normalized)
                dist = dist_matrix[i, j]
                scores['distance'] = np.exp(-dist / (max_reasonable_dist / 2))

                # Technique 2: BBox proximity
                scores['bbox_proximity'] = self._bbox_proximity_score(icon, tag)

                # Technique 3: Exclusion zone
                scores['exclusion_zone'] = self._exclusion_zone_score(icon, tag, icons)

                # Technique 4: Alignment
                scores['alignment'] = self._alignment_score(icon, tag)

                # Technique 5: Density context (returns multiplier, not direct score)
                density_mult = self._local_density_score(icon, tag, icons, tags)

                # Technique 6: Directional
                scores['directional'] = self._directional_score(icon, tag)

                # Weighted voting
                total_vote = sum(scores[k] * self.weights[k] for k in scores)
                total_weight = sum(self.weights.values())

                # Apply density multiplier
                final_vote = (total_vote / total_weight) * density_mult

                votes[i, j] = final_vote

                # Count techniques that "voted yes" (score > 0.5)
                techniques_agree = sum(1 for s in scores.values() if s > 0.5)

                # Require minimum agreement
                if techniques_agree < self.min_votes_required:
                    votes[i, j] *= 0.5  # Penalize weak consensus

        return votes

    def match(
        self,
        icons: List[Detection],
        tags: List[Detection],
        image_shape: Tuple[int, int]
    ) -> Tuple[List[str], Dict]:
        """
        Match icons to tags using multi-technique voting.
        
        Args:
            icons: List of icon Detection objects
            tags: List of tag Detection objects
            image_shape: (height, width) of the image
            
        Returns:
            icon_labels: List of labels for each icon ("unknown" if no match)
            debug_info: Dictionary with matching statistics
        """
        debug_info = {
            'assignments': [],
            'distances': [],
            'vote_scores': [],
            'rejected_reason': [],
        }

        img_height, img_width = image_shape
        img_diagonal = np.sqrt(img_height ** 2 + img_width ** 2)

        if self.enable_debug:
            print(f"[MATCHER] Image shape: {img_width}x{img_height}, diagonal: {img_diagonal:.1f}px")
            print(f"[MATCHER] Icons: {len(icons)}, Tags: {len(tags)}")
            print(f"[MATCHER] Confidence threshold: {self.confidence_threshold}")
            print(f"[MATCHER] Min votes required: {self.min_votes_required}")
            print("-" * 70)

        if len(tags) == 0:
            if self.enable_debug:
                print("[MATCHER] No tag detections - all icons marked as unknown")
            return ["unknown"] * len(icons), debug_info

        # Compute vote matrix
        if self.enable_debug:
            print("\n[MATCHER] Computing vote matrix using 6 techniques...")
        votes = self._compute_vote_matrix(icons, tags, img_diagonal)

        # Greedy one-to-one matching based on vote scores
        if self.enable_debug:
            print("\n[MATCHER] Performing greedy matching...")

        icon_labels = ["unknown"] * len(icons)
        used_tags = set()
        assigned_icons = set()

        # Create list of (vote_score, icon_idx, tag_idx) and sort by score descending
        candidates = []
        for i in range(len(icons)):
            for j in range(len(tags)):
                candidates.append((votes[i, j], i, j))
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Greedy matching: assign highest scoring pairs first
        for vote_score, icon_idx, tag_idx in candidates:
            if icon_idx in assigned_icons or tag_idx in used_tags:
                continue

            # Check confidence threshold
            if vote_score < self.confidence_threshold:
                debug_info['rejected_reason'].append(
                    f"icon {icon_idx} -> tag {tag_idx}: vote score {vote_score:.3f} < threshold {self.confidence_threshold}"
                )
                continue

            # Valid assignment
            tag = tags[tag_idx]
            label = tag.text_label if tag.text_label else f"tag_{tag_idx}"
            icon_labels[icon_idx] = label

            assigned_icons.add(icon_idx)
            used_tags.add(tag_idx)

            dist = self._calculate_distance(icons[icon_idx].center, tag.center)
            debug_info['assignments'].append({
                'icon_idx': icon_idx,
                'text_idx': tag_idx,  # Use text_idx for backward compatibility
                'tag_idx': tag_idx,
                'vote_score': float(vote_score),
                'distance': float(dist),
                'label': label
            })
            debug_info['distances'].append(float(dist))
            debug_info['vote_scores'].append(float(vote_score))

            if self.enable_debug:
                icon = icons[icon_idx]
                print(f"[MATCHER] ✓ Icon #{icon_idx+1} at {icon.center} → '{label}' (vote: {vote_score:.3f}, dist: {dist:.1f}px)")

            # Stop if all tags are assigned
            if len(used_tags) == len(tags):
                break

        # Print unassigned icons
        if self.enable_debug:
            print()
            for icon_idx, label in enumerate(icon_labels):
                if label == "unknown":
                    icon = icons[icon_idx]
                    # Find best available tag
                    best_score = 0
                    best_tag_idx = -1
                    for j in range(len(tags)):
                        if j not in used_tags and votes[icon_idx, j] > best_score:
                            best_score = votes[icon_idx, j]
                            best_tag_idx = j

                    if best_tag_idx >= 0:
                        reason = f"best available score {best_score:.3f} < threshold {self.confidence_threshold}"
                    else:
                        reason = "all tags already assigned to higher-scoring icons"

                    debug_info['rejected_reason'].append(f"icon {icon_idx}: {reason}")
                    print(f"[MATCHER] ✗ Icon #{icon_idx+1} at {icon.center} → unknown ({reason})")

            # Print summary
            print("-" * 70)
            if debug_info['distances']:
                print(f"[MATCHER] Successfully assigned: {len(debug_info['distances'])}/{len(icons)}")
                print(f"[MATCHER] Average vote score: {np.mean(debug_info['vote_scores']):.3f}")
                print(f"[MATCHER] Average distance: {np.mean(debug_info['distances']):.1f}px")
            else:
                print("[MATCHER] No successful assignments")
            print(f"[MATCHER] Unknown icons: {icon_labels.count('unknown')}/{len(icons)}")

        return icon_labels, debug_info


# Backward-compatible wrapper function
def assign_labels_to_icons_dynamic(
    icon_detections: List[Detection],
    text_detections: List[Detection],
    image_shape: Tuple[int, int]
) -> Tuple[List[str], Dict]:
    """
    Assign text labels to icons using advanced multi-technique voting.
    
    This is a backward-compatible wrapper for AdvancedIconTagMatcher.
    
    Args:
        icon_detections: List of icon Detection objects
        text_detections: List of text Detection objects
        image_shape: (height, width) of the image
    
    Returns:
        icon_labels: List of labels for each icon ("unknown" if no match)
        debug_info: Dictionary with matching statistics
    """
    matcher = AdvancedIconTagMatcher(
        confidence_threshold=0.60,
        high_density_threshold=0.70,
        min_votes_required=4,
        enable_debug=True
    )
    return matcher.match(icon_detections, text_detections, image_shape)
