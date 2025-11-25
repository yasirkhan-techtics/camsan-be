"""
Gemini-Based Icon-Tag Matcher for Unmatched Detections
Uses Google Gemini 2.5-flash to match icons/tags with structured JSON output
"""

import pandas as pd
import numpy as np
import cv2
import base64
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai
from pathlib import Path
import time
import matplotlib.pyplot as plt


@dataclass
class MatchingStats:
    """Statistics for LLM matching"""
    total_unmatched_icons: int = 0
    total_unassigned_tags: int = 0
    icons_matched: int = 0
    tags_matched: int = 0
    api_calls: int = 0
    api_errors: int = 0
    matches: List[Dict] = None
    
    def __post_init__(self):
        if self.matches is None:
            self.matches = []


class GeminiIconTagMatcher:
    """
    Uses Google Gemini 2.5-flash to match unassigned icons and tags
    with structured JSON output for reliable parsing
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        enable_debug: bool = True
    ):
        """
        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use (gemini-2.0-flash-exp recommended)
            enable_debug: Print detailed information
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.enable_debug = enable_debug
        
    def _calculate_padding_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate average distance and size statistics from successfully matched pairs
        
        Args:
            df: DataFrame with matched icons and tags
        
        Returns:
            Dictionary with statistics
        """
        # Filter only matched pairs
        matched = df[df['Match_Status'] == 'matched'].copy()
        
        if len(matched) == 0:
            # Default values if no matches
            return {
                'avg_icon_width': 150,
                'avg_icon_height': 150,
                'avg_distance': 120,
                'padding_multiplier': 2.5
            }
        
        # Calculate statistics
        avg_icon_width = matched['Icon_BBox_Width'].mean()
        avg_icon_height = matched['Icon_BBox_Height'].mean()
        avg_distance = matched['Icon_Tag_Distance'].mean()
        
        # Padding multiplier to capture surrounding context
        padding_multiplier = 0.5
        
        stats = {
            'avg_icon_width': avg_icon_width,
            'avg_icon_height': avg_icon_height,
            'avg_distance': avg_distance,
            'padding_multiplier': padding_multiplier,
            'num_matched': len(matched)
        }
        
        if self.enable_debug:
            print("\n" + "="*80)
            print("PADDING STATISTICS FROM MATCHED PAIRS")
            print("="*80)
            print(f"Matched pairs analyzed: {stats['num_matched']}")
            print(f"Average icon width:  {stats['avg_icon_width']:.1f}px")
            print(f"Average icon height: {stats['avg_icon_height']:.1f}px")
            print(f"Average icon-tag distance: {stats['avg_distance']:.1f}px")
            print(f"Padding multiplier: {stats['padding_multiplier']}x")
        
        return stats
    
    def _create_padded_crop(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        padding_stats: Dict,
        is_icon: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create a padded crop around an item with context
        
        Args:
            image: Full image
            center: Center coordinates of the item
            padding_stats: Statistics for padding calculation
            is_icon: True if cropping around icon, False for tag
        
        Returns:
            Cropped image and crop info dictionary
        """
        cx, cy = center
        img_h, img_w = image.shape[:2]
        
        # Calculate padding based on average distance and size
        if is_icon:
            # For icons: use distance + icon size
            padding = int(
                (padding_stats['avg_distance'] + 
                 max(padding_stats['avg_icon_width'], padding_stats['avg_icon_height'])) * 
                padding_stats['padding_multiplier']
            )
        else:
            # For tags: larger window to see more icons
            padding = int(
                (padding_stats['avg_distance'] + 
                 max(padding_stats['avg_icon_width'], padding_stats['avg_icon_height'])) * 
                padding_stats['padding_multiplier'] * 1.5
            )
        
        # Calculate crop bounds
        x1 = max(0, cx - padding)
        y1 = max(0, cy - padding)
        x2 = min(img_w, cx + padding)
        y2 = min(img_h, cy + padding)
        
        # Crop image
        crop = image[y1:y2, x1:x2].copy()
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        # Calculate center in crop coordinates
        center_in_crop = (cx - x1, cy - y1)
        
        # Draw marker at center
        color = (0, 255, 0) if is_icon else (255, 0, 255)
        cv2.drawMarker(crop, center_in_crop, color, cv2.MARKER_CROSS, 30, 3)
        
        crop_info = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'center_in_crop': center_in_crop,
            'crop_size': (x2 - x1, y2 - y1),
            'padding': padding
        }
        
        return crop, crop_info
    
    def _overlay_matched_icons(
        self,
        crop: np.ndarray,
        crop_info: Dict,
        df: pd.DataFrame,
        full_image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Overlay semi-transparent boxes on already matched icons in the crop
        
        Args:
            crop: Cropped image
            crop_info: Information about the crop
            df: Full DataFrame with all icons
            full_image_shape: (height, width) of full image
        
        Returns:
            Crop with overlaid matched icons
        """
        x1, y1 = crop_info['x1'], crop_info['y1']
        
        # Get matched icons
        matched = df[df['Match_Status'] == 'matched'].copy()
        
        overlay = crop.copy()
        
        for _, row in matched.iterrows():
            icon_cx = row['Icon_Center_X']
            icon_cy = row['Icon_Center_Y']
            icon_w = row['Icon_BBox_Width']
            icon_h = row['Icon_BBox_Height']
            
            # Check if icon is within crop bounds
            if (x1 <= icon_cx <= x1 + crop_info['crop_size'][0] and
                y1 <= icon_cy <= y1 + crop_info['crop_size'][1]):
                
                # Convert to crop coordinates
                icon_x_crop = int(icon_cx - x1 - icon_w/2)
                icon_y_crop = int(icon_cy - y1 - icon_h/2)
                
                # Draw semi-transparent gray box
                cv2.rectangle(
                    overlay,
                    (icon_x_crop, icon_y_crop),
                    (icon_x_crop + int(icon_w), icon_y_crop + int(icon_h)),
                    (0, 0, 0),
                    -1
                )
        
        # Blend with original
        alpha = 0.5
        result = cv2.addWeighted(overlay, alpha, crop, 1 - alpha, 0)
        
        return result
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 string"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _create_icon_matching_prompt(
        self,
        icon_name: str,
        available_tags: List[str],
        icon_template_path: Optional[str] = None
    ) -> str:
        """
        Create prompt for asking Gemini to find matching tag for an icon
        
        Args:
            icon_name: Name/type of the icon
            available_tags: List of available unassigned tags
            icon_template_path: Path to reference template image (optional)
        
        Returns:
            Prompt string
        """
        tags_list = "\n".join([f"  - {tag}" for tag in available_tags])
        
        prompt = f"""You are analyzing an electrical drawing to match symbols with their labels.

**TASK:** Find the label/tag that belongs to the ICON marked with a GREEN crosshair (+).

**Icon Type:** {icon_name}

**Available Tags:**
{tags_list}

**Instructions:**
1. Look at the icon marked with the GREEN crosshair
2. Search the surrounding area for a text label that identifies THIS specific icon
3. The tag should be near the icon (typically within a few icon-widths)
4. Choose ONLY from the available tags listed above
5. For small benefit of the doubt like I as 1 or similar then match the closest icon


**Response Format:**
You MUST respond with VALID JSON in this exact format:

{{
  "match_found": true or false,
  "matched_tag": "tag_name" or null,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}}

**Examples:**

If you find a match:
{{
  "match_found": true,
  "matched_tag": "<tag_name>",
  "confidence": "high",
  "reasoning": "<tag_name> label is directly adjacent to the marked icon"
}}

If NO match found:
{{
  "match_found": false,
  "matched_tag": null,
  "confidence": "low",
  "reasoning": "No clear tag visible near the marked icon"
}}

**IMPORTANT:**
- Respond ONLY with valid JSON
- Use "true"/"false" (lowercase) for match_found
- Use null (not "null" or "None") if no match
- Keep reasoning brief (one sentence)
"""
        return prompt
    
    def _create_tag_matching_prompt(
        self,
        tag_name: str,
        available_icon_types: List[str]
    ) -> str:
        """
        Create prompt for asking Gemini to find matching icon for a tag
        
        Args:
            tag_name: Name of the tag
            available_icon_types: List of available unmatched icon types
        
        Returns:
            Prompt string
        """
        icons_list = "\n".join([f"  - {icon}" for icon in available_icon_types])
        
        prompt = f"""You are analyzing an electrical drawing to match labels with their symbols.

**TASK:** Find the ICON that this label belongs to.

**Label/Tag:** {tag_name} (marked with MAGENTA crosshair +)

**Note:** Already matched icons are shown with GRAY semi-transparent overlay. Look for icons WITHOUT gray overlay.

**Available Icon Types:**
{icons_list}

**Instructions:**
1. Look at the tag/label marked with the MAGENTA crosshair
2. Search for a nearby icon (symbol) that this label identifies
3. IGNORE icons with gray overlay (already matched)
4. Choose ONLY from the available icon types listed above

**Response Format:**
You MUST respond with VALID JSON in this exact format:

{{
  "match_found": true or false,
  "matched_icon_type": "icon_type" or null,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}}

**Examples:**

If you find a match:
{{
  "match_found": true,
  "matched_icon_type": "chandlier",
  "confidence": "high",
  "reasoning": "Chandelier icon is directly above the marked tag"
}}

If NO match found:
{{
  "match_found": false,
  "matched_icon_type": null,
  "confidence": "low",
  "reasoning": "No unmatched icons visible near the tag"
}}

**IMPORTANT:**
- Respond ONLY with valid JSON
- Use "true"/"false" (lowercase) for match_found
- Use null (not "null" or "None") if no match
- Keep reasoning brief (one sentence)
- IGNORE gray-overlaid icons (already matched)
"""
        return prompt
    
    def _query_gemini(
        self,
        prompt: str,
        image_base64: str,
        max_retries: int = 3
    ) -> Dict:
        """
        Query Gemini API with image and prompt, returns parsed JSON
        
        Args:
            prompt: Text prompt
            image_base64: Base64 encoded image
            max_retries: Maximum number of retry attempts
        
        Returns:
            Parsed JSON response as dictionary
        """
        for attempt in range(max_retries):
            try:
                # Decode base64 to bytes for Gemini
                image_bytes = base64.b64decode(image_base64)
                
                # Create message with image and text
                response = self.model.generate_content([
                    {
                        'mime_type': 'image/png',
                        'data': image_bytes
                    },
                    prompt
                ])
                
                # Extract text response
                response_text = response.text.strip()
                
                # Try to extract JSON from response
                # Remove markdown code blocks if present
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                # Parse JSON
                result = json.loads(response_text)
                
                # Validate required fields
                if 'match_found' not in result:
                    raise ValueError("Response missing 'match_found' field")
                
                return result
                
            except json.JSONDecodeError as e:
                if self.enable_debug:
                    print(f"   ⚠️  JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"   Raw response: {response_text[:200]}...")
                
                if attempt == max_retries - 1:
                    return {
                        'match_found': False,
                        'matched_tag': None,
                        'matched_icon_type': None,
                        'confidence': 'low',
                        'reasoning': 'Failed to parse response'
                    }
                
                time.sleep(1)  # Wait before retry
                
            except Exception as e:
                if self.enable_debug:
                    print(f"   ❌ API Error (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    return {
                        'match_found': False,
                        'matched_tag': None,
                        'matched_icon_type': None,
                        'confidence': 'low',
                        'reasoning': f'API error: {str(e)}'
                    }
                
                time.sleep(1)  # Wait before retry
        
        return {
            'match_found': False,
            'matched_tag': None,
            'matched_icon_type': None,
            'confidence': 'low',
            'reasoning': 'Max retries exceeded'
        }
    
    def match_unmatched_items(
        self,
        image_path: str,
        df: pd.DataFrame,
        save_crops: bool = False,
        crops_dir: str = "gemini_crops"
    ) -> Tuple[pd.DataFrame, MatchingStats]:
        """
        Match unmatched icons and tags using Gemini
        
        Args:
            image_path: Path to the electrical drawing image
            df: DataFrame with columns matching your format
            save_crops: Whether to save crop images for inspection
            crops_dir: Directory to save crops
        
        Returns:
            Updated DataFrame and statistics
        """
        if self.enable_debug:
            print("\n" + "="*80)
            print("GEMINI-BASED ICON-TAG MATCHING")
            print("="*80)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img_h, img_w = image.shape[:2]
        
        if self.enable_debug:
            print(f"✅ Image loaded: {img_w}x{img_h}")
        
        # Create crops directory if needed
        if save_crops:
            Path(crops_dir).mkdir(exist_ok=True)
        
        # Calculate padding statistics
        padding_stats = self._calculate_padding_stats(df)
        
        # Get unmatched icons and unassigned tags
        unmatched_icons = df[df['Match_Status'] == 'unmatched_icon'].copy()
        unassigned_tags = df[df['Match_Status'] == 'unassigned_tag'].copy()
        
        # Get available tags and icons
        available_tags = unassigned_tags['Tag_Name'].unique().tolist()
        available_icon_types = unmatched_icons['Icon_Original_Name'].unique().tolist()
        
        if self.enable_debug:
            print("\n" + "-"*80)
            print("MATCHING CANDIDATES")
            print("-"*80)
            print(f"Unmatched icons: {len(unmatched_icons)}")
            print(f"Unassigned tags: {len(unassigned_tags)}")
            print(f"Available tag types: {available_tags}")
            print(f"Available icon types: {available_icon_types}")
        
        # Initialize statistics
        stats = MatchingStats(
            total_unmatched_icons=len(unmatched_icons),
            total_unassigned_tags=len(unassigned_tags)
        )
        
        # Make a copy of the dataframe for updates
        df_updated = df.copy()
        
        # Track used tags and icons
        used_tags = set()
        used_icon_indices = set()
        
        # ====================================================================
        # PHASE 1: Match unmatched icons to unassigned tags
        # ====================================================================
        
        if len(unmatched_icons) > 0 and len(available_tags) > 0:
            if self.enable_debug:
                print("\n" + "="*80)
                print("PHASE 1: MATCHING UNMATCHED ICONS TO TAGS")
                print("="*80)
            
            for idx, row in unmatched_icons.iterrows():
                if len(available_tags) == 0:
                    break
                
                symbol_id = row['Symbol_ID']
                center = (int(row['Icon_Center_X']), int(row['Icon_Center_Y']))
                icon_name = row['Icon_Original_Name']
                
                if self.enable_debug:
                    print(f"\n🔍 Icon ID={symbol_id} ({icon_name}) at {center}")
                
                # Create padded crop
                crop, crop_info = self._create_padded_crop(
                    image, center, padding_stats, is_icon=True
                )
                
                # Save crop if requested
                if save_crops:
                    crop_path = f"{crops_dir}/icon_{symbol_id}_crop.png"
                    cv2.imwrite(crop_path, crop)
                
                # Encode image
                image_base64 = self._encode_image_to_base64(crop)
                
                # Create prompt
                prompt = self._create_icon_matching_prompt(icon_name, available_tags)
                
                # Query Gemini
                if self.enable_debug:
                    print(f"   📡 Querying Gemini API...")
                
                stats.api_calls += 1
                result = self._query_gemini(prompt, image_base64)
                
                if not result.get('match_found', False):
                    if self.enable_debug:
                        print(f"   ✗ No match - {result.get('reasoning', 'Unknown')}")
                    continue
                
                matched_tag = result.get('matched_tag')
                
                if not matched_tag or matched_tag not in available_tags:
                    if self.enable_debug:
                        print(f"   ✗ Invalid tag: {matched_tag}")
                    continue
                
                if matched_tag in used_tags:
                    if self.enable_debug:
                        print(f"   ⚠️  Tag '{matched_tag}' already used")
                    continue
                
                # Find the tag row
                tag_row = unassigned_tags[unassigned_tags['Tag_Name'] == matched_tag].iloc[0]
                
                # Calculate distance
                tag_cx = tag_row['Tag_Center_X']
                tag_cy = tag_row['Tag_Center_Y']
                distance = np.sqrt((center[0] - tag_cx)**2 + (center[1] - tag_cy)**2)
                
                # Valid match!
                if self.enable_debug:
                    confidence = result.get('confidence', 'unknown')
                    reasoning = result.get('reasoning', '')
                    print(f"   ✅ Matched to '{matched_tag}' (conf={confidence}, dist={distance:.1f}px)")
                    print(f"      Reasoning: {reasoning}")
                
                # Update DataFrame - change from unmatched_icon to matched
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Match_Status'] = 'matched'
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Assigned_Label'] = matched_tag
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Match_Confidence'] = 0.85
                # df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Detection_Type'] = 'llm_matched'

                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Matcher_Type'] = 'llm_matched'

                
                # Add tag information
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Symbol_ID'] = tag_row['Symbol_ID']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_BBox_X'] = tag_row['Tag_BBox_X']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_BBox_Y'] = tag_row['Tag_BBox_Y']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_BBox_Width'] = tag_row['Tag_BBox_Width']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_BBox_Height'] = tag_row['Tag_BBox_Height']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Center_X'] = tag_cx
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Center_Y'] = tag_cy
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Confidence'] = tag_row['Tag_Confidence']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Name'] = matched_tag
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Scale'] = tag_row['Tag_Scale']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Tag_Rotation'] = tag_row['Tag_Rotation']
                df_updated.loc[df_updated['Symbol_ID'] == symbol_id, 'Icon_Tag_Distance'] = distance
                
                # Track usage
                used_tags.add(matched_tag)
                used_icon_indices.add(idx)
                available_tags.remove(matched_tag)
                
                stats.icons_matched += 1
                stats.matches.append({
                    'type': 'icon_to_tag',
                    'symbol_id': symbol_id,
                    'matched_to': matched_tag,
                    'confidence': result.get('confidence', 'unknown'),
                    'distance': distance
                })
        
        # ====================================================================
        # PHASE 2: Match unassigned tags to unmatched icons
        # ====================================================================
        
        # Refresh available icons (exclude those matched in Phase 1)
        remaining_unmatched = df_updated[df_updated['Match_Status'] == 'unassigned_tag'].copy()
        available_icon_types = remaining_unmatched['Icon_Original_Name'].unique().tolist()
        
        # Get remaining unassigned tags
        remaining_tags = df_updated[
            (df_updated['Match_Status'] == 'unassigned_tag') &
            (~df_updated['Tag_Name'].isin(used_tags))
        ].copy()
        
        if len(remaining_tags) > 0 and len(remaining_unmatched) > 0:
            if self.enable_debug:
                print("\n" + "="*80)
                print("PHASE 2: MATCHING UNASSIGNED TAGS TO ICONS")
                print("="*80)
            
            for idx, row in remaining_tags.iterrows():
                if len(available_icon_types) == 0:
                    break
                
                symbol_id = row['Symbol_ID']
                center = (int(row['Tag_Center_X']), int(row['Tag_Center_Y']))
                tag_name = row['Tag_Name']
                
                if self.enable_debug:
                    print(f"\n🏷️  Tag ID={symbol_id} ('{tag_name}') at {center}")
                
                # Create padded crop with matched icons overlaid
                crop, crop_info = self._create_padded_crop(
                    image, center, padding_stats, is_icon=False
                )
                
                # Overlay matched icons
                crop = self._overlay_matched_icons(crop, crop_info, df_updated, (img_h, img_w))
                
                # Save crop if requested
                if save_crops:
                    crop_path = f"{crops_dir}/tag_{symbol_id}_crop.png"
                    cv2.imwrite(crop_path, crop)
                
                # Encode image
                image_base64 = self._encode_image_to_base64(crop)
                
                # Create prompt
                prompt = self._create_tag_matching_prompt(tag_name, available_icon_types)
                
                # Query Gemini
                if self.enable_debug:
                    print(f"   📡 Querying Gemini API...")
                
                stats.api_calls += 1
                result = self._query_gemini(prompt, image_base64)
                
                if not result.get('match_found', False):
                    if self.enable_debug:
                        print(f"   ✗ No match - {result.get('reasoning', 'Unknown')}")
                    continue
                
                matched_icon_type = result.get('matched_icon_type')
                
                if not matched_icon_type or matched_icon_type not in available_icon_types:
                    if self.enable_debug:
                        print(f"   ✗ Invalid icon type: {matched_icon_type}")
                    continue
                
                # Find closest unmatched icon of this type
                candidate_icons = remaining_unmatched[
                    remaining_unmatched['Icon_Original_Name'] == matched_icon_type
                ].copy()
                
                if len(candidate_icons) == 0:
                    if self.enable_debug:
                        print(f"   ⚠️  No available icons of type '{matched_icon_type}'")
                    continue
                
                # Calculate distances
                tag_center = np.array(center)
                candidate_icons['temp_distance'] = candidate_icons.apply(
                    lambda r: np.sqrt(
                        (r['Icon_Center_X'] - tag_center[0])**2 + 
                        (r['Icon_Center_Y'] - tag_center[1])**2
                    ),
                    axis=1
                )
                
                # Get closest icon
                closest_icon_idx = candidate_icons['temp_distance'].idxmin()
                closest_icon = candidate_icons.loc[closest_icon_idx]
                closest_icon_id = closest_icon['Symbol_ID']
                distance = closest_icon['temp_distance']
                
                # Valid match!
                if self.enable_debug:
                    confidence = result.get('confidence', 'unknown')
                    reasoning = result.get('reasoning', '')
                    print(f"   ✅ Matched to icon ID={closest_icon_id} (conf={confidence}, dist={distance:.1f}px)")
                    print(f"      Reasoning: {reasoning}")
                
                # Update DataFrame - change from unmatched_icon to matched
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Match_Status'] = 'matched'
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Assigned_Label'] = tag_name
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Match_Confidence'] = 0.80
                # df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Detection_Type'] = 'llm_matched'

                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Matcher_Type'] = 'llm_matched'
                
                # Add tag information
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Tag_Symbol_ID'] = symbol_id
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Tag_BBox_X'] = row['Tag_BBox_X']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Tag_BBox_Y'] = row['Tag_BBox_Y']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Tag_BBox_Width'] = row['Tag_BBox_Width']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Tag_BBox_Height'] = row['Tag_BBox_Height']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id, 'Tag_Center_X'] = center[0]
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id['Symbol_ID'], 'Tag_Center_Y'] = row['Tag_Center_Y']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id['Symbol_ID'], 'Tag_Confidence'] = row['Tag_Confidence']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id['Symbol_ID'], 'Tag_Name'] = tag_name
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id['Symbol_ID'], 'Tag_Scale'] = row['Tag_Scale']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id['Symbol_ID'], 'Tag_Rotation'] = row['Tag_Rotation']
                df_updated.loc[df_updated['Symbol_ID'] == closest_icon_id['Symbol_ID'], 'Icon_Tag_Distance'] = distance

                # Track usage
                available_icon_types.remove(matched_icon_type)
                used_tags.add(tag_name)

                stats.tags_matched += 1
                stats.matches.append({
                    'type': 'tag_to_icon',
                    'symbol_id': symbol_id,
                    'matched_to': matched_icon_type,
                    'confidence': result.get('confidence', 'unknown'),
                    'distance': distance
                })

        # Final statistics
        if self.enable_debug:
            print("\n" + "="*80)
            print("MATCHING COMPLETE")
            print("="*80)
            print(f"Icons matched: {stats.icons_matched}")
            print(f"Tags matched : {stats.tags_matched}")
            print(f"API calls    : {stats.api_calls}")

        return df_updated, stats