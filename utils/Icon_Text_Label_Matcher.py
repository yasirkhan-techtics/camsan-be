"""
Advanced Icon-Tag Matching System for Electrical Drawings
Uses multi-technique voting to minimize false positives
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    scale: float
    rotation: int
    center: Tuple[int, int]
    tag_name: str
    template_type: str  # "icon" or "tag"


@dataclass
class LabeledIcon:
    """Output format for labeled icons"""
    bbox: Tuple[int, int, int, int]
    bbox_confidence: float
    assigned_label: str
    label_confidence: float  # Confidence in the label assignment
    center: Tuple[int, int]
    original_detection: Detection


@dataclass
class UnassignedTag:
    """Output format for unassigned tags"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    tag_name: str
    center: Tuple[int, int]
    reason: str  # Why it wasn't assigned


class AdvancedIconTagMatcher:
    """
    Multi-technique voting system for icon-tag matching
    Minimizes false positives through ensemble methods
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
    
    def match_icons_to_tags(
        self,
        all_detections: List[Detection],
        image_shape: Tuple[int, int]
    ) -> Tuple[List[LabeledIcon], List[UnassignedTag], Dict]:
        """
        Main matching function.
        
        Args:
            all_detections: Combined list of icon and tag detections
            image_shape: (height, width) of the image
        
        Returns:
            labeled_icons: Icons with their assigned labels
            unassigned_tags: Tags that weren't matched to any icon
            debug_info: Detailed matching information
        """
        # Separate icons and tags
        icons = [d for d in all_detections if d.template_type == "icon"]
        tags = [d for d in all_detections if d.template_type == "tag"]
        
        if self.enable_debug:
            print("=" * 80)
            print("ADVANCED ICON-TAG MATCHING")
            print("=" * 80)
            print(f"Icons: {len(icons)}, Tags: {len(tags)}")
            print(f"Image shape: {image_shape[1]}x{image_shape[0]}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print(f"Minimum votes required: {self.min_votes_required}/{len(self.weights)}")
            print("-" * 80)
        
        labeled_icons = []
        unassigned_tags = []
        debug_info = {
            'total_icons': len(icons),
            'total_tags': len(tags),
            'assignments': [],
            'vote_matrix': None,
            'thresholds_used': {}
        }
        
        if len(tags) == 0:
            if self.enable_debug:
                print("⚠ No tags detected - all icons marked as unknown")
            
            for icon in icons:
                labeled_icons.append(LabeledIcon(
                    bbox=icon.bbox,
                    bbox_confidence=icon.confidence,
                    assigned_label="unknown",
                    label_confidence=0.0,
                    center=icon.center,
                    original_detection=icon
                ))
            return labeled_icons, unassigned_tags, debug_info
        
        # Calculate image diagonal for normalization
        img_h, img_w = image_shape
        img_diagonal = np.sqrt(img_h ** 2 + img_w ** 2)
        
        # Compute vote matrix
        if self.enable_debug:
            print("\n🔍 Computing multi-technique votes...")
        
        vote_matrix = self._compute_vote_matrix(icons, tags, img_diagonal)
        debug_info['vote_matrix'] = vote_matrix
        
        # Greedy bipartite matching
        if self.enable_debug:
            print("\n🎯 Performing one-to-one matching...")
        
        used_tags = set()
        assigned_icons = set()
        
        # Create candidate list: (vote_score, icon_idx, tag_idx)
        candidates = []
        for i in range(len(icons)):
            for j in range(len(tags)):
                if vote_matrix[i, j] > 0:  # Only consider non-zero votes
                    candidates.append((vote_matrix[i, j], i, j))
        
        # Sort by vote score (highest first)
        candidates.sort(reverse=True)
        
        # Greedy assignment
        matches = []
        for vote_score, icon_idx, tag_idx in candidates:
            # Skip if already assigned
            if icon_idx in assigned_icons or tag_idx in used_tags:
                continue
            
            # Determine threshold based on local density
            icon = icons[icon_idx]
            tag = tags[tag_idx]
            density_mult = self._local_density_score(icon, tag, icons, tags)
            
            if density_mult < 0.7:
                threshold = self.high_density_threshold
            else:
                threshold = self.confidence_threshold
            
            # Check if vote meets threshold
            if vote_score >= threshold:
                matches.append((icon_idx, tag_idx, vote_score))
                assigned_icons.add(icon_idx)
                used_tags.add(tag_idx)
                
                debug_info['assignments'].append({
                    'icon_idx': icon_idx,
                    'tag_idx': tag_idx,
                    'vote_score': vote_score,
                    'threshold_used': threshold,
                    'label': tag.tag_name
                })
        
        if self.enable_debug:
            print(f"\n✅ Matched {len(matches)} icon(s) to tags")
            print(f"❓ {len(icons) - len(matches)} icon(s) remain unknown")
            print(f"🏷️  {len(tags) - len(used_tags)} tag(s) unassigned")
            print("\n" + "=" * 80)
            print("ASSIGNMENT DETAILS:")
            print("=" * 80)
        
        # Build output for labeled icons
        for icon_idx, tag_idx, vote_score in matches:
            icon = icons[icon_idx]
            tag = tags[tag_idx]
            
            labeled_icons.append(LabeledIcon(
                bbox=icon.bbox,
                bbox_confidence=icon.confidence,
                assigned_label=tag.tag_name,
                label_confidence=vote_score,
                center=icon.center,
                original_detection=icon
            ))
            
            if self.enable_debug:
                dist = self._calculate_distance(icon.center, tag.center)
                print(f"✓ Icon at {icon.center} → '{tag.tag_name}' "
                      f"(vote={vote_score:.3f}, dist={dist:.1f}px)")
        
        # Add unknown icons
        for icon_idx, icon in enumerate(icons):
            if icon_idx not in assigned_icons:
                labeled_icons.append(LabeledIcon(
                    bbox=icon.bbox,
                    bbox_confidence=icon.confidence,
                    assigned_label="unknown",
                    label_confidence=0.0,
                    center=icon.center,
                    original_detection=icon
                ))
                
                if self.enable_debug:
                    # Find why it wasn't matched
                    if len(tags) > 0:
                        best_vote = np.max(vote_matrix[icon_idx, :])
                        best_tag_idx = np.argmax(vote_matrix[icon_idx, :])
                        
                        if best_tag_idx in used_tags:
                            reason = "best tag already assigned"
                        else:
                            reason = f"insufficient vote ({best_vote:.3f})"
                    else:
                        reason = "no tags available"
                    
                    print(f"✗ Icon at {icon.center} → unknown ({reason})")
        
        # Build output for unassigned tags
        for tag_idx, tag in enumerate(tags):
            if tag_idx not in used_tags:
                # Find best icon (even though not assigned)
                if len(icons) > 0:
                    best_vote = np.max(vote_matrix[:, tag_idx])
                    best_icon_idx = np.argmax(vote_matrix[:, tag_idx])
                    
                    if best_icon_idx in assigned_icons:
                        reason = "best icon already assigned to different tag"
                    else:
                        reason = f"insufficient vote with best icon ({best_vote:.3f})"
                else:
                    reason = "no icons available"
                
                unassigned_tags.append(UnassignedTag(
                    bbox=tag.bbox,
                    confidence=tag.confidence,
                    tag_name=tag.tag_name,
                    center=tag.center,
                    reason=reason
                ))
                
                if self.enable_debug:
                    print(f"🏷️  Tag '{tag.tag_name}' at {tag.center} unassigned ({reason})")
        
        if self.enable_debug:
            print("=" * 80)
        
        return labeled_icons, unassigned_tags, debug_info


# ============================================================================
# CONVENIENCE FUNCTION FOR EASY USAGE
# ============================================================================

def match_icons_to_tags(
    all_detections: List[Detection],
    image_shape: Tuple[int, int],
    confidence_threshold: float = 0.60,
    enable_debug: bool = True
) -> Tuple[List[LabeledIcon], List[UnassignedTag], Dict]:
    """
    Convenience function for icon-tag matching.
    
    Args:
        all_detections: List of Detection objects (both icons and tags)
        image_shape: (height, width) tuple
        confidence_threshold: Minimum confidence for assignment (0-1)
        enable_debug: Print detailed information
    
    Returns:
        labeled_icons: List of LabeledIcon objects
        unassigned_tags: List of UnassignedTag objects  
        debug_info: Dictionary with matching details
    
    Example:
        >>> labeled_icons, unassigned_tags, debug = match_icons_to_tags(
        ...     all_detections=detections,
        ...     image_shape=(7200, 10800),
        ...     confidence_threshold=0.60
        ... )
    """
    matcher = AdvancedIconTagMatcher(
        confidence_threshold=confidence_threshold,
        high_density_threshold=confidence_threshold + 0.10,
        min_votes_required=4,
        enable_debug=enable_debug
    )
    
    return matcher.match_icons_to_tags(all_detections, image_shape)


# ============================================================================
# OUTPUT FORMATTING HELPERS
# ============================================================================

def print_summary(labeled_icons: List[LabeledIcon], unassigned_tags: List[UnassignedTag]):
    """Print a summary of matching results"""
    print("\n" + "=" * 80)
    print("MATCHING SUMMARY")
    print("=" * 80)
    
    # Count by label
    label_counts = {}
    for icon in labeled_icons:
        label = icon.assigned_label
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nTotal icons: {len(labeled_icons)}")
    for label, count in sorted(label_counts.items()):
        avg_conf = np.mean([i.label_confidence for i in labeled_icons if i.assigned_label == label])
        if label == "unknown":
            print(f"  • {label}: {count}")
        else:
            print(f"  • {label}: {count} (avg confidence: {avg_conf:.3f})")
    
    print(f"\nUnassigned tags: {len(unassigned_tags)}")
    if unassigned_tags:
        for tag in unassigned_tags:
            print(f"  • {tag.tag_name} - {tag.reason}")
    
    print("=" * 80)


def export_to_dict(labeled_icons: List[LabeledIcon]) -> List[Dict]:
    """Convert labeled icons to dictionary format"""
    return [
        {
            'bbox': icon.bbox,
            'bbox_confidence': float(icon.bbox_confidence),
            'label': icon.assigned_label,
            'label_confidence': float(icon.label_confidence),
            'center': icon.center,
        }
        for icon in labeled_icons
    ]


def filter_by_confidence(
    labeled_icons: List[LabeledIcon],
    min_confidence: float = 0.5
) -> Tuple[List[LabeledIcon], List[LabeledIcon]]:
    """
    Split icons into high-confidence and low-confidence groups.
    
    Returns:
        high_confidence: Icons with label_confidence >= min_confidence
        low_confidence: Icons with label_confidence < min_confidence (excluding unknown)
    """
    high_conf = []
    low_conf = []
    
    for icon in labeled_icons:
        if icon.assigned_label == "unknown":
            low_conf.append(icon)
        elif icon.label_confidence >= min_confidence:
            high_conf.append(icon)
        else:
            low_conf.append(icon)
    
    return high_conf, low_conf


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def draw_labeled_results(
    image_path: str,
    labeled_icons: List[LabeledIcon],
    unassigned_tags: List[UnassignedTag],
    output_path: str,
    show_confidence: bool = True,
    show_unassigned_tags: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes with labels and confidence scores on the image.
    
    Args:
        image_path: Path to the input image
        labeled_icons: List of LabeledIcon objects
        unassigned_tags: List of UnassignedTag objects
        output_path: Path to save the annotated image
        show_confidence: Whether to show confidence scores
        show_unassigned_tags: Whether to draw unassigned tags
    
    Returns:
        Annotated image as numpy array
    
    Example:
        >>> result_img = draw_labeled_results(
        ...     image_path="input.png",
        ...     labeled_icons=labeled_icons,
        ...     unassigned_tags=unassigned_tags,
        ...     output_path="result.png"
        ... )
    """
    import cv2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    result = image.copy()
    img_h, img_w = image.shape[:2]
    
    # Calculate adaptive sizes based on image resolution
    line_thickness = max(1, int((img_h + img_w) / 3000))
    dot_radius = max(3, int((img_h + img_w) / 4000))
    font_scale = max(0.2, line_thickness * 0.125)
    font_thickness = max(1, line_thickness // 2)
    
    # Define color palette
    color_palette = {
        'unknown': (0, 0, 255),       # Red for unknown
        'default': (0, 255, 0),       # Green default
        'high_conf': (0, 255, 0),     # Green for high confidence
        'medium_conf': (0, 165, 255), # Orange for medium confidence
        'low_conf': (0, 100, 255),    # Dark orange for low confidence
        'unassigned_tag': (255, 0, 255),  # Magenta for unassigned tags
    }
    
    # Build unique label colors
    unique_labels = set([icon.assigned_label for icon in labeled_icons if icon.assigned_label != "unknown"])
    label_specific_colors = [
        (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (255, 255, 0), (128, 255, 0),
    ]
    
    label_colors = {}
    for i, label in enumerate(sorted(unique_labels)):
        label_colors[label] = label_specific_colors[i % len(label_specific_colors)]
    
    # Draw icons
    for icon in labeled_icons:
        x, y, w, h = icon.bbox
        label = icon.assigned_label
        bbox_conf = icon.bbox_confidence
        label_conf = icon.label_confidence
        
        # Choose color based on label and confidence
        if label == "unknown":
            color = color_palette['unknown']
        else:
            # Use label-specific color, but modify brightness based on confidence
            if label_conf >= 0.7:
                color = label_colors[label]
            elif label_conf >= 0.5:
                # Slightly darker for medium confidence
                base_color = label_colors[label]
                color = tuple(int(c * 0.8) for c in base_color)
            else:
                # Much darker for low confidence
                base_color = label_colors[label]
                color = tuple(int(c * 0.6) for c in base_color)
        
        # Draw bounding box
        alpha1 = 0.2
        overlay = result.copy()

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha1, result, 1 - alpha1, 0, result)
        # cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)

        # Draw center point
        cv2.circle(result, icon.center, dot_radius, color, -1)
        cv2.circle(result, icon.center, dot_radius + 1, (0, 0, 0), 1)
        
        # Prepare label text
        if label == "unknown":
            text = "?"
            if show_confidence:
                text = f"? ({bbox_conf:.2f})"
        else:
            text = label
            if show_confidence:
                text = f"{label} ({label_conf:.2f})"
        
        # Calculate text size and position
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Position text above bbox, or below if too close to top
        text_x = x
        text_y = y - baseline - 5
        if text_y - text_h < 0:
            text_y = y + h + text_h + baseline + 5
        
        # Draw semi-transparent background for text
        # overlay = result.copy()
        # padding = 4
        # cv2.rectangle(
        #     overlay,
        #     (text_x - 2, text_y - text_h - baseline - 2),
        #     (text_x + text_w + padding, text_y + baseline + 2),
        #     color,
        #     -1
        # )
        # cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        # Draw text
        cv2.putText(
            overlay,
            text,
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA
        )

        # Blend with transparency
        alpha = 0.5   # 0 = fully transparent, 1 = solid
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Draw unassigned tags (if enabled)
    if show_unassigned_tags:
        for tag in unassigned_tags:
            x, y, w, h = tag.bbox
            color = color_palette['unassigned_tag']
            
            # Draw with dashed style (smaller thickness)
            dash_length = line_thickness * 3
            
            # Top edge (dashed)
            for i in range(x, x + w, dash_length * 2):
                cv2.line(result, (i, y), (min(i + dash_length, x + w), y), color, line_thickness)
            
            # Bottom edge (dashed)
            for i in range(x, x + w, dash_length * 2):
                cv2.line(result, (i, y + h), (min(i + dash_length, x + w), y + h), color, line_thickness)
            
            # Left edge (dashed)
            for i in range(y, y + h, dash_length * 2):
                cv2.line(result, (x, i), (x, min(i + dash_length, y + h)), color, line_thickness)
            
            # Right edge (dashed)
            for i in range(y, y + h, dash_length * 2):
                cv2.line(result, (x + w, i), (x + w, min(i + dash_length, y + h)), color, line_thickness)
            
            # Draw center point
            cv2.circle(result, tag.center, dot_radius, color, -1)
            cv2.circle(result, tag.center, dot_radius + 1, (0, 0, 0), 1)
            
            # Draw tag name
            text = f"[{tag.tag_name}]"
            if show_confidence:
                text = f"[{tag.tag_name}] ({tag.confidence:.2f})"
            
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            text_x = x
            text_y = y - baseline - 5
            if text_y - text_h < 0:
                text_y = y + h + text_h + baseline + 5
            
            # Background
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (text_x - 2, text_y - text_h - baseline - 2),
                (text_x + text_w + 4, text_y + baseline + 2),
                color,
                -1
            )
            cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
            
            # Text
            cv2.putText(
                result,
                text,
                (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
    
    # Add legend in top-left corner
    legend_y = 30
    legend_x = 20
    legend_font_scale = font_scale * 1.2
    legend_thickness = font_thickness + 1
    
    # Draw semi-transparent background for legend
    legend_items = []
    if any(icon.assigned_label != "unknown" for icon in labeled_icons):
        legend_items.append(("Labeled Icons:", (255, 255, 255)))
        
        for label in sorted(unique_labels):
            count = sum(1 for icon in labeled_icons if icon.assigned_label == label)
            legend_items.append((f"  {label}: {count}", label_colors[label]))
    
    unknown_count = sum(1 for icon in labeled_icons if icon.assigned_label == "unknown")
    if unknown_count > 0:
        legend_items.append((f"Unknown: {unknown_count}", color_palette['unknown']))
    
    if show_unassigned_tags and unassigned_tags:
        legend_items.append((f"Unassigned Tags: {len(unassigned_tags)}", color_palette['unassigned_tag']))
    
    if legend_items:
        max_text_w = 0
        total_h = 0
        for text, _ in legend_items:
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, legend_thickness
            )
            max_text_w = max(max_text_w, text_w)
            total_h += text_h + 10
        
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (legend_x - 10, legend_y - 20),
            (legend_x + max_text_w + 20, legend_y + total_h + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        current_y = legend_y
        for text, color in legend_items:
            cv2.putText(
                result,
                text,
                (legend_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                legend_font_scale,
                color,
                legend_thickness,
                cv2.LINE_AA
            )
            current_y += int((legend_font_scale * 30) + 10)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"\n✅ Annotated image saved: {output_path}")
    print(f"   Resolution: {img_w}x{img_h}")
    print(f"   Labeled icons: {len(labeled_icons)}")
    print(f"   Unassigned tags: {len(unassigned_tags)}")
    
    return result