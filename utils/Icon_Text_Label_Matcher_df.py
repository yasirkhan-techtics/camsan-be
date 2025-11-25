"""
Advanced Icon-Tag Matching System for Electrical Drawings
Uses multi-technique voting to minimize false positives
Updated to accept pandas DataFrame as input
"""

import numpy as np
import pandas as pd
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
    symbol_id: int  # Added to track original row


@dataclass
class LabeledIcon:
    """Output format for labeled icons"""
    symbol_id: int
    bbox: Tuple[int, int, int, int]
    bbox_confidence: float
    assigned_label: str
    label_confidence: float  # Confidence in the label assignment
    center: Tuple[int, int]
    original_detection: Detection
    matched_tag: Optional[Detection] = None  # Store the matched tag info


@dataclass
class UnassignedTag:
    """Output format for unassigned tags"""
    symbol_id: int
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
                    symbol_id=icon.symbol_id,
                    bbox=icon.bbox,
                    bbox_confidence=icon.confidence,
                    assigned_label="unknown",
                    label_confidence=0.0,
                    center=icon.center,
                    original_detection=icon,
                    matched_tag=None
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
                symbol_id=icon.symbol_id,
                bbox=icon.bbox,
                bbox_confidence=icon.confidence,
                assigned_label=tag.tag_name,
                label_confidence=vote_score,
                center=icon.center,
                original_detection=icon,
                matched_tag=tag  # Store matched tag info
            ))
            
            if self.enable_debug:
                dist = self._calculate_distance(icon.center, tag.center)
                print(f"✓ Icon ID={icon.symbol_id} at {icon.center} → '{tag.tag_name}' "
                      f"(vote={vote_score:.3f}, dist={dist:.1f}px)")
        
        # Add unknown icons
        for icon_idx, icon in enumerate(icons):
            if icon_idx not in assigned_icons:
                labeled_icons.append(LabeledIcon(
                    symbol_id=icon.symbol_id,
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
                    
                    print(f"✗ Icon ID={icon.symbol_id} at {icon.center} → unknown ({reason})")
        
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
                    symbol_id=tag.symbol_id,
                    bbox=tag.bbox,
                    confidence=tag.confidence,
                    tag_name=tag.tag_name,
                    center=tag.center,
                    reason=reason
                ))
                
                if self.enable_debug:
                    print(f"🏷️  Tag ID={tag.symbol_id} '{tag.tag_name}' at {tag.center} unassigned ({reason})")
        
        if self.enable_debug:
            print("=" * 80)
        
        return labeled_icons, unassigned_tags, debug_info


# ============================================================================
# DATAFRAME INPUT FUNCTIONS
# ============================================================================

def df_to_detections(df: pd.DataFrame) -> List[Detection]:
    """
    Convert pandas DataFrame to list of Detection objects.
    
    Expected DataFrame columns:
        - Symbol_ID: int
        - BBox_X, BBox_Y, BBox_Width, BBox_Height: int
        - Center_X, Center_Y: int
        - Confidence: float
        - Scale: float
        - Rotation: int
        - Tag_Name: str
        - Template_Type: str ('icon' or 'tag')
    
    Args:
        df: pandas DataFrame with detection data
    
    Returns:
        List of Detection objects
    """
    detections = []
    
    for _, row in df.iterrows():
        detection = Detection(
            bbox=(
                int(row['BBox_X']),
                int(row['BBox_Y']),
                int(row['BBox_Width']),
                int(row['BBox_Height'])
            ),
            confidence=float(row['Confidence']),
            scale=float(row['Scale']),
            rotation=int(row['Rotation']),
            center=(int(row['Center_X']), int(row['Center_Y'])),
            tag_name=str(row['Tag_Name']),
            template_type=str(row['Template_Type']).lower(),
            symbol_id=int(row['Symbol_ID'])
        )
        detections.append(detection)
    
    return detections


def labeled_icons_to_df(labeled_icons: List[LabeledIcon]) -> pd.DataFrame:
    """
    Convert list of LabeledIcon objects back to pandas DataFrame.
    Includes both icon AND tag bbox information in the same row.
    
    Args:
        labeled_icons: List of LabeledIcon objects
    
    Returns:
        pandas DataFrame with labeled icon data including tag bbox info
    """
    data = []
    
    for icon in labeled_icons:
        x, y, w, h = icon.bbox
        cx, cy = icon.center
        
        row = {
            # Icon Information
            'Symbol_ID': icon.symbol_id,
            'Icon_BBox_X': x,
            'Icon_BBox_Y': y,
            'Icon_BBox_Width': w,
            'Icon_BBox_Height': h,
            'Icon_Center_X': cx,
            'Icon_Center_Y': cy,
            'Icon_Confidence': icon.bbox_confidence,
            'Icon_Original_Name': icon.original_detection.tag_name,
            'Icon_Scale': icon.original_detection.scale,
            'Icon_Rotation': icon.original_detection.rotation,
            
            # Matching Results
            'Assigned_Label': icon.assigned_label,
            'Match_Confidence': icon.label_confidence,
            'Match_Status': 'matched' if icon.assigned_label != 'unknown' else 'unmatched',
        }
        
        # Tag Information (if matched)
        if icon.matched_tag is not None:
            tag = icon.matched_tag
            tx, ty, tw, th = tag.bbox
            tcx, tcy = tag.center
            
            row.update({
                'Tag_Symbol_ID': tag.symbol_id,
                'Tag_BBox_X': tx,
                'Tag_BBox_Y': ty,
                'Tag_BBox_Width': tw,
                'Tag_BBox_Height': th,
                'Tag_Center_X': tcx,
                'Tag_Center_Y': tcy,
                'Tag_Confidence': tag.confidence,
                'Tag_Name': tag.tag_name,
                'Tag_Scale': tag.scale,
                'Tag_Rotation': tag.rotation,
                'Icon_Tag_Distance': np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            })
        else:
            # No matched tag - fill with None
            row.update({
                'Tag_Symbol_ID': None,
                'Tag_BBox_X': None,
                'Tag_BBox_Y': None,
                'Tag_BBox_Width': None,
                'Tag_BBox_Height': None,
                'Tag_Center_X': None,
                'Tag_Center_Y': None,
                'Tag_Confidence': None,
                'Tag_Name': None,
                'Tag_Scale': None,
                'Tag_Rotation': None,
                'Icon_Tag_Distance': None
            })
        
        data.append(row)
    
    return pd.DataFrame(data)


def unassigned_tags_to_df(unassigned_tags: List[UnassignedTag]) -> pd.DataFrame:
    """
    Convert list of UnassignedTag objects to pandas DataFrame.
    These are tags that couldn't be matched to any icon.
    
    Args:
        unassigned_tags: List of UnassignedTag objects
    
    Returns:
        pandas DataFrame with unassigned tag data
    """
    data = []
    
    for tag in unassigned_tags:
        x, y, w, h = tag.bbox
        cx, cy = tag.center
        
        data.append({
            'Symbol_ID': tag.symbol_id,
            'Tag_BBox_X': x,
            'Tag_BBox_Y': y,
            'Tag_BBox_Width': w,
            'Tag_BBox_Height': h,
            'Tag_Center_X': cx,
            'Tag_Center_Y': cy,
            'Tag_Confidence': tag.confidence,
            'Tag_Name': tag.tag_name,
            'Match_Status': 'unassigned_tag',
            'Unassigned_Reason': tag.reason,
        })
    
    return pd.DataFrame(data)


# ============================================================================
# CONVENIENCE FUNCTION FOR DATAFRAME INPUT
# ============================================================================

def create_comprehensive_output_df(
    labeled_icons: List[LabeledIcon],
    unassigned_tags: List[UnassignedTag]
) -> pd.DataFrame:
    """
    Create a comprehensive DataFrame combining all icons and unassigned tags.
    Each row represents either:
    - A matched icon-tag pair (with both icon and tag bbox info)
    - An unmatched icon (with icon bbox info only)
    - An unassigned tag (with tag bbox info only)
    
    Args:
        labeled_icons: List of LabeledIcon objects
        unassigned_tags: List of UnassignedTag objects
    
    Returns:
        Comprehensive DataFrame with all detection information
    """
    all_rows = []
    
    # Add all icons (matched and unmatched)
    for icon in labeled_icons:
        x, y, w, h = icon.bbox
        cx, cy = icon.center
        
        row = {
            'Symbol_ID': icon.symbol_id,
            'Detection_Type': 'icon',
            'Match_Status': 'matched' if icon.assigned_label != 'unknown' else 'unmatched_icon',
            
            # Icon Information
            'Icon_BBox_X': x,
            'Icon_BBox_Y': y,
            'Icon_BBox_Width': w,
            'Icon_BBox_Height': h,
            'Icon_Center_X': cx,
            'Icon_Center_Y': cy,
            'Icon_Confidence': icon.bbox_confidence,
            'Icon_Original_Name': icon.original_detection.tag_name,
            'Icon_Scale': icon.original_detection.scale,
            'Icon_Rotation': icon.original_detection.rotation,
            
            # Matching Results
            'Assigned_Label': icon.assigned_label,
            'Match_Confidence': icon.label_confidence,
        }
        
        # Tag Information (if matched)
        if icon.matched_tag is not None:
            tag = icon.matched_tag
            tx, ty, tw, th = tag.bbox
            tcx, tcy = tag.center
            
            row.update({
                'Tag_Symbol_ID': tag.symbol_id,
                'Tag_BBox_X': tx,
                'Tag_BBox_Y': ty,
                'Tag_BBox_Width': tw,
                'Tag_BBox_Height': th,
                'Tag_Center_X': tcx,
                'Tag_Center_Y': tcy,
                'Tag_Confidence': tag.confidence,
                'Tag_Name': tag.tag_name,
                'Tag_Scale': tag.scale,
                'Tag_Rotation': tag.rotation,
                'Icon_Tag_Distance': np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            })
        else:
            # No matched tag - fill with None
            row.update({
                'Tag_Symbol_ID': None,
                'Tag_BBox_X': None,
                'Tag_BBox_Y': None,
                'Tag_BBox_Width': None,
                'Tag_BBox_Height': None,
                'Tag_Center_X': None,
                'Tag_Center_Y': None,
                'Tag_Confidence': None,
                'Tag_Name': None,
                'Tag_Scale': None,
                'Tag_Rotation': None,
                'Icon_Tag_Distance': None
            })
        
        all_rows.append(row)
    
    # Add all unassigned tags
    for tag in unassigned_tags:
        tx, ty, tw, th = tag.bbox
        tcx, tcy = tag.center
        
        row = {
            'Symbol_ID': tag.symbol_id,
            'Detection_Type': 'tag',
            'Match_Status': 'unassigned_tag',
            
            # No icon information
            'Icon_BBox_X': None,
            'Icon_BBox_Y': None,
            'Icon_BBox_Width': None,
            'Icon_BBox_Height': None,
            'Icon_Center_X': None,
            'Icon_Center_Y': None,
            'Icon_Confidence': None,
            'Icon_Original_Name': None,
            'Icon_Scale': None,
            'Icon_Rotation': None,
            
            # Matching Results
            'Assigned_Label': 'N/A',
            'Match_Confidence': 0.0,
            
            # Tag Information
            'Tag_Symbol_ID': tag.symbol_id,
            'Tag_BBox_X': tx,
            'Tag_BBox_Y': ty,
            'Tag_BBox_Width': tw,
            'Tag_BBox_Height': th,
            'Tag_Center_X': tcx,
            'Tag_Center_Y': tcy,
            'Tag_Confidence': tag.confidence,
            'Tag_Name': tag.tag_name,
            'Tag_Scale': None,
            'Tag_Rotation': None,
            'Icon_Tag_Distance': None
        }
        
        all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    
    # Reorder columns for better readability
    column_order = [
        'Symbol_ID', 'Detection_Type', 'Match_Status', 'Assigned_Label', 'Match_Confidence',
        'Icon_BBox_X', 'Icon_BBox_Y', 'Icon_BBox_Width', 'Icon_BBox_Height',
        'Icon_Center_X', 'Icon_Center_Y', 'Icon_Confidence', 'Icon_Original_Name',
        'Icon_Scale', 'Icon_Rotation',
        'Tag_Symbol_ID', 'Tag_BBox_X', 'Tag_BBox_Y', 'Tag_BBox_Width', 'Tag_BBox_Height',
        'Tag_Center_X', 'Tag_Center_Y', 'Tag_Confidence', 'Tag_Name',
        'Tag_Scale', 'Tag_Rotation', 'Icon_Tag_Distance'
    ]
    
    return df[column_order]


def match_icons_to_tags_df(
    df: pd.DataFrame,
    image_shape: Tuple[int, int],
    confidence_threshold: float = 0.60,
    enable_debug: bool = True,
    return_comprehensive: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convenience function for icon-tag matching with DataFrame input/output.
    
    Args:
        df: pandas DataFrame with detection data (icons and tags)
        image_shape: (height, width) tuple
        confidence_threshold: Minimum confidence for assignment (0-1)
        enable_debug: Print detailed information
        return_comprehensive: If True, returns comprehensive combined DataFrame.
                             If False, returns separate labeled/unassigned DataFrames.
    
    Returns:
        If return_comprehensive=True:
            comprehensive_df: Single DataFrame with ALL detections (icons + tags)
            unassigned_tags_df: Separate DataFrame with just unassigned tags (for convenience)
            debug_info: Dictionary with matching details
        
        If return_comprehensive=False:
            labeled_icons_df: DataFrame with labeled icons (original format)
            unassigned_tags_df: DataFrame with unassigned tags
            debug_info: Dictionary with matching details
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv('detections.csv')
        >>> 
        >>> # Get comprehensive output (recommended)
        >>> comprehensive_df, unassigned_df, debug = match_icons_to_tags_df(
        ...     df=df,
        ...     image_shape=(7200, 10800),
        ...     confidence_threshold=0.60,
        ...     return_comprehensive=True
        ... )
        >>> 
        >>> # comprehensive_df contains ALL information:
        >>> # - Matched icons with tag bbox info
        >>> # - Unmatched icons
        >>> # - Unassigned tags
    """
    # Convert DataFrame to Detection objects
    detections = df_to_detections(df)
    
    # Create matcher and perform matching
    matcher = AdvancedIconTagMatcher(
        confidence_threshold=confidence_threshold,
        high_density_threshold=confidence_threshold + 0.10,
        min_votes_required=4,
        enable_debug=enable_debug
    )
    
    labeled_icons, unassigned_tags, debug_info = matcher.match_icons_to_tags(
        detections, image_shape
    )
    
    if return_comprehensive:
        # Return comprehensive DataFrame with all detections
        comprehensive_df = create_comprehensive_output_df(labeled_icons, unassigned_tags)
        unassigned_tags_df = unassigned_tags_to_df(unassigned_tags)
        return comprehensive_df, unassigned_tags_df, debug_info
    else:
        # Return separate DataFrames (original behavior)
        labeled_icons_df = labeled_icons_to_df(labeled_icons)
        unassigned_tags_df = unassigned_tags_to_df(unassigned_tags)
        return labeled_icons_df, unassigned_tags_df, debug_info


# ============================================================================
# OUTPUT FORMATTING HELPERS
# ============================================================================

def print_summary_df(comprehensive_df: pd.DataFrame = None, 
                     labeled_df: pd.DataFrame = None, 
                     unassigned_df: pd.DataFrame = None):
    """
    Print a summary of matching results from DataFrames.
    Can accept either:
    - comprehensive_df (new format with all detections)
    - labeled_df + unassigned_df (old format, separate DataFrames)
    
    Args:
        comprehensive_df: Comprehensive DataFrame with all detections
        labeled_df: DataFrame with labeled icons (old format)
        unassigned_df: DataFrame with unassigned tags (old format)
    """
    print("\n" + "=" * 80)
    print("MATCHING SUMMARY")
    print("=" * 80)
    
    if comprehensive_df is not None:
        # New format - comprehensive DataFrame
        total_icons = len(comprehensive_df[comprehensive_df['Detection_Type'] == 'icon'])
        matched_icons = len(comprehensive_df[comprehensive_df['Match_Status'] == 'matched'])
        unmatched_icons = len(comprehensive_df[comprehensive_df['Match_Status'] == 'unmatched_icon'])
        unassigned_tags = len(comprehensive_df[comprehensive_df['Match_Status'] == 'unassigned_tag'])
        
        print(f"\nTotal icons: {total_icons}")
        print(f"  • Matched: {matched_icons} ({matched_icons/total_icons*100:.1f}%)")
        print(f"  • Unmatched: {unmatched_icons} ({unmatched_icons/total_icons*100:.1f}%)")
        
        # Label distribution for matched icons
        if matched_icons > 0:
            print(f"\nLabel distribution:")
            matched_data = comprehensive_df[comprehensive_df['Match_Status'] == 'matched']
            label_counts = matched_data['Assigned_Label'].value_counts()
            
            for label, count in sorted(label_counts.items()):
                avg_conf = matched_data[matched_data['Assigned_Label'] == label]['Match_Confidence'].mean()
                print(f"  • {label}: {count} (avg confidence: {avg_conf:.3f})")
        
        print(f"\nUnassigned tags: {unassigned_tags}")
        if unassigned_tags > 0:
            tag_data = comprehensive_df[comprehensive_df['Match_Status'] == 'unassigned_tag']
            for _, tag in tag_data.iterrows():
                print(f"  • {tag['Tag_Name']}")
    
    elif labeled_df is not None and unassigned_df is not None:
        # Old format - separate DataFrames
        label_counts = labeled_df['Assigned_Label'].value_counts()
        
        print(f"\nTotal icons: {len(labeled_df)}")
        for label, count in sorted(label_counts.items()):
            if label == "unknown":
                print(f"  • {label}: {count}")
            else:
                avg_conf = labeled_df[labeled_df['Assigned_Label'] == label]['Match_Confidence'].mean()
                print(f"  • {label}: {count} (avg confidence: {avg_conf:.3f})")
        
        print(f"\nUnassigned tags: {len(unassigned_df)}")
        if len(unassigned_df) > 0:
            for _, tag in unassigned_df.iterrows():
                print(f"  • {tag['Tag_Name']} - {tag['Unassigned_Reason']}")
    else:
        print("Error: Must provide either comprehensive_df or both labeled_df and unassigned_df")
        return
    
    print("=" * 80)


# ============================================================================
# VISUALIZATION FUNCTION (Updated for DataFrame)
# ============================================================================

def draw_labeled_results_df(
    image_path: str,
    comprehensive_df: pd.DataFrame = None,
    labeled_df: pd.DataFrame = None,
    unassigned_df: pd.DataFrame = None,
    output_path: str = 'labeled_result.png',
    show_confidence: bool = True,
    show_unassigned_tags: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes with labels and confidence scores on the image.
    Can accept either comprehensive_df or separate labeled_df + unassigned_df.
    
    Args:
        image_path: Path to the input image
        comprehensive_df: Comprehensive DataFrame with all detections (new format)
        labeled_df: DataFrame with labeled icons (old format)
        unassigned_df: DataFrame with unassigned tags (old format)
        output_path: Path to save the annotated image
        show_confidence: Whether to show confidence scores
        show_unassigned_tags: Whether to draw unassigned tags
    
    Returns:
        Annotated image as numpy array
    """
    import cv2
    
    # Convert comprehensive_df to old format if provided
    if comprehensive_df is not None:
        # Extract icon data
        icon_data = comprehensive_df[comprehensive_df['Detection_Type'] == 'icon'].copy()
        labeled_df = icon_data
        
        # Extract unassigned tag data
        tag_data = comprehensive_df[comprehensive_df['Match_Status'] == 'unassigned_tag'].copy()
        unassigned_df = tag_data
    
    if labeled_df is None:
        raise ValueError("Must provide either comprehensive_df or labeled_df")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    result = image.copy()
    img_h, img_w = image.shape[:2]
    
    # Calculate adaptive sizes
    line_thickness = max(1, int((img_h + img_w) / 3000))
    dot_radius = max(3, int((img_h + img_w) / 4000))
    font_scale = max(0.2, line_thickness * 0.125)
    font_thickness = max(1, line_thickness // 2)
    
    # Define color palette
    color_palette = {
        'unknown': (0, 0, 255),       # Red
        'default': (0, 255, 0),       # Green
        'unassigned_tag': (255, 0, 255),  # Magenta
    }
    
    # Build unique label colors
    if 'Assigned_Label' in labeled_df.columns:
        unique_labels = labeled_df[labeled_df['Assigned_Label'] != 'unknown']['Assigned_Label'].unique()
    else:
        unique_labels = []
    
    label_specific_colors = [
        (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (255, 255, 0), (128, 255, 0),
    ]
    
    label_colors = {}
    for i, label in enumerate(sorted(unique_labels)):
        label_colors[label] = label_specific_colors[i % len(label_specific_colors)]
    
    # Determine column names based on format
    if 'Icon_BBox_X' in labeled_df.columns:
        # New comprehensive format
        bbox_x_col, bbox_y_col = 'Icon_BBox_X', 'Icon_BBox_Y'
        bbox_w_col, bbox_h_col = 'Icon_BBox_Width', 'Icon_BBox_Height'
        center_x_col, center_y_col = 'Icon_Center_X', 'Icon_Center_Y'
        conf_col = 'Icon_Confidence'
        label_col = 'Assigned_Label'
        match_conf_col = 'Match_Confidence'
        status_col = 'Match_Status'
    else:
        # Old format
        bbox_x_col, bbox_y_col = 'BBox_X', 'BBox_Y'
        bbox_w_col, bbox_h_col = 'BBox_Width', 'BBox_Height'
        center_x_col, center_y_col = 'Center_X', 'Center_Y'
        conf_col = 'BBox_Confidence'
        label_col = 'Assigned_Label'
        match_conf_col = 'Label_Confidence'
        status_col = None
    
    # Draw icons
    for _, row in labeled_df.iterrows():
        x, y = int(row[bbox_x_col]), int(row[bbox_y_col])
        w, h = int(row[bbox_w_col]), int(row[bbox_h_col])
        cx, cy = int(row[center_x_col]), int(row[center_y_col])
        label = row[label_col]
        bbox_conf = row[conf_col]
        label_conf = row[match_conf_col]
        
        # Determine if matched or unmatched
        if status_col and row[status_col] == 'unmatched_icon':
            label = 'unknown'
        
        # Choose color
        if label == "unknown":
            color = color_palette['unknown']
        else:
            if label_conf >= 0.7:
                color = label_colors[label]
            elif label_conf >= 0.5:
                base_color = label_colors[label]
                color = tuple(int(c * 0.8) for c in base_color)
            else:
                base_color = label_colors[label]
                color = tuple(int(c * 0.6) for c in base_color)
        
        # Draw bounding box with transparency
        alpha1 = 0.2
        overlay = result.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha1, result, 1 - alpha1, 0, result)
        
        # Draw center point
        cv2.circle(result, (cx, cy), dot_radius, color, -1)
        cv2.circle(result, (cx, cy), dot_radius + 1, (0, 0, 0), 1)
        
        # Prepare label text
        if label == "unknown":
            text = "?" if not show_confidence else f"? ({bbox_conf:.2f})"
        else:
            text = label if not show_confidence else f"{label} ({label_conf:.2f})"
        
        # Draw text
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        text_x = x
        text_y = y - baseline - 5
        if text_y - text_h < 0:
            text_y = y + h + text_h + baseline + 5
        
        overlay = result.copy()
        cv2.putText(
            overlay, text, (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
            font_thickness, cv2.LINE_AA
        )
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Draw unassigned tags
    if show_unassigned_tags and unassigned_df is not None and len(unassigned_df) > 0:
        # Determine column names for tags
        if 'Tag_BBox_X' in unassigned_df.columns:
            tag_bbox_x_col = 'Tag_BBox_X'
            tag_bbox_y_col = 'Tag_BBox_Y'
            tag_bbox_w_col = 'Tag_BBox_Width'
            tag_bbox_h_col = 'Tag_BBox_Height'
            tag_center_x_col = 'Tag_Center_X'
            tag_center_y_col = 'Tag_Center_Y'
            tag_conf_col = 'Tag_Confidence'
            tag_name_col = 'Tag_Name'
        else:
            tag_bbox_x_col = 'BBox_X'
            tag_bbox_y_col = 'BBox_Y'
            tag_bbox_w_col = 'BBox_Width'
            tag_bbox_h_col = 'BBox_Height'
            tag_center_x_col = 'Center_X'
            tag_center_y_col = 'Center_Y'
            tag_conf_col = 'Confidence'
            tag_name_col = 'Tag_Name'
        
        for _, row in unassigned_df.iterrows():
            x = int(row[tag_bbox_x_col]) if pd.notna(row[tag_bbox_x_col]) else 0
            y = int(row[tag_bbox_y_col]) if pd.notna(row[tag_bbox_y_col]) else 0
            w = int(row[tag_bbox_w_col]) if pd.notna(row[tag_bbox_w_col]) else 0
            h = int(row[tag_bbox_h_col]) if pd.notna(row[tag_bbox_h_col]) else 0
            cx = int(row[tag_center_x_col]) if pd.notna(row[tag_center_x_col]) else 0
            cy = int(row[tag_center_y_col]) if pd.notna(row[tag_center_y_col]) else 0
            tag_name = row[tag_name_col]
            conf = row[tag_conf_col] if pd.notna(row[tag_conf_col]) else 0.0
            color = color_palette['unassigned_tag']
            
            # Draw dashed box
            dash_length = line_thickness * 3
            for i in range(x, x + w, dash_length * 2):
                cv2.line(result, (i, y), (min(i + dash_length, x + w), y), color, line_thickness)
            for i in range(x, x + w, dash_length * 2):
                cv2.line(result, (i, y + h), (min(i + dash_length, x + w), y + h), color, line_thickness)
            for i in range(y, y + h, dash_length * 2):
                cv2.line(result, (x, i), (x, min(i + dash_length, y + h)), color, line_thickness)
            for i in range(y, y + h, dash_length * 2):
                cv2.line(result, (x + w, i), (x + w, min(i + dash_length, y + h)), color, line_thickness)
            
            cv2.circle(result, (cx, cy), dot_radius, color, -1)
            cv2.circle(result, (cx, cy), dot_radius + 1, (0, 0, 0), 1)
            
            text = f"[{tag_name}]" if not show_confidence else f"[{tag_name}] ({conf:.2f})"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            text_x = x
            text_y = y - baseline - 5
            if text_y - text_h < 0:
                text_y = y + h + text_h + baseline + 5
            
            overlay = result.copy()
            cv2.rectangle(
                overlay, (text_x - 2, text_y - text_h - baseline - 2),
                (text_x + text_w + 4, text_y + baseline + 2), color, -1
            )
            cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
            cv2.putText(
                result, text, (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA
            )
    
    # Add legend
    legend_y = 30
    legend_x = 20
    legend_font_scale = font_scale * 1.2
    legend_thickness = font_thickness + 1
    
    legend_items = []
    if len(unique_labels) > 0:
        legend_items.append(("Labeled Icons:", (255, 255, 255)))
        for label in sorted(unique_labels):
            count = len(labeled_df[labeled_df[label_col] == label])
            legend_items.append((f"  {label}: {count}", label_colors[label]))
    
    unknown_count = len(labeled_df[labeled_df[label_col] == 'unknown']) if status_col is None else len(labeled_df[labeled_df[status_col] == 'unmatched_icon'])
    if unknown_count > 0:
        legend_items.append((f"Unknown: {unknown_count}", color_palette['unknown']))
    
    if show_unassigned_tags and unassigned_df is not None and len(unassigned_df) > 0:
        legend_items.append((f"Unassigned Tags: {len(unassigned_df)}", color_palette['unassigned_tag']))
    
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
            overlay, (legend_x - 10, legend_y - 20),
            (legend_x + max_text_w + 20, legend_y + total_h + 10),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        current_y = legend_y
        for text, color in legend_items:
            cv2.putText(
                result, text, (legend_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, color,
                legend_thickness, cv2.LINE_AA
            )
            current_y += int((legend_font_scale * 30) + 10)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"\n✅ Annotated image saved: {output_path}")
    print(f"   Resolution: {img_w}x{img_h}")
    if comprehensive_df is not None:
        print(f"   Total icons: {len(labeled_df)}")
        print(f"   Unassigned tags: {len(unassigned_df) if unassigned_df is not None else 0}")
    else:
        print(f"   Labeled icons: {len(labeled_df)}")
        print(f"   Unassigned tags: {len(unassigned_df) if unassigned_df is not None else 0}")
    
    return result