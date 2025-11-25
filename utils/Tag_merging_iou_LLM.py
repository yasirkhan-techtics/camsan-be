import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Union
import google.generativeai as genai
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class TagOverlapResolver:
    """
    Module to detect and resolve overlapping tag bounding boxes using Gemini LLM in batches.
    """
    
    def __init__(self, api_key: str, overlap_threshold: float = 0.9, batch_size: int = 10):
        """
        Initialize the Tag Overlap Resolver.
        
        Args:
            api_key: Google Gemini API key
            overlap_threshold: Overlap threshold (default: 0.9 for 90% overlap)
            batch_size: Maximum overlaps per LLM call (default: 10)
        """
        self.overlap_threshold = overlap_threshold
        self.batch_size = batch_size
        genai.configure(api_key=api_key)
        
        # Configure Gemini model with JSON output
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config
        )
    
    def calculate_overlap_ratio(self, bbox1: Dict, bbox2: Dict) -> Tuple[float, float]:
        """
        Calculate overlap ratio for both boxes.
        Returns (overlap_ratio_box1, overlap_ratio_box2)
        """
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['w'], bbox2['x'] + bbox2['w'])
        y2 = min(bbox1['y'] + bbox1['h'], bbox2['y'] + bbox2['h'])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = bbox1['w'] * bbox1['h']
        area2 = bbox2['w'] * bbox2['h']
        
        ratio1 = intersection / area1 if area1 > 0 else 0
        ratio2 = intersection / area2 if area2 > 0 else 0
        
        return ratio1, ratio2
    
    def find_overlapping_pairs(self, tag_df: pd.DataFrame) -> List[Tuple[int, int, float, float]]:
        """
        Find all pairs of rows where bboxes overlap >= threshold.
        
        Returns:
            List of (idx1, idx2, overlap_ratio1, overlap_ratio2)
        """
        overlapping_pairs = []
        
        for i in range(len(tag_df)):
            bbox1 = {
                'x': tag_df.iloc[i]['BBox_X'],
                'y': tag_df.iloc[i]['BBox_Y'],
                'w': tag_df.iloc[i]['BBox_Width'],
                'h': tag_df.iloc[i]['BBox_Height']
            }
            
            for j in range(i + 1, len(tag_df)):
                bbox2 = {
                    'x': tag_df.iloc[j]['BBox_X'],
                    'y': tag_df.iloc[j]['BBox_Y'],
                    'w': tag_df.iloc[j]['BBox_Width'],
                    'h': tag_df.iloc[j]['BBox_Height']
                }
                
                ratio1, ratio2 = self.calculate_overlap_ratio(bbox1, bbox2)
                
                # Check if either box has >= threshold overlap
                if ratio1 >= self.overlap_threshold or ratio2 >= self.overlap_threshold:
                    overlapping_pairs.append((i, j, ratio1, ratio2))
        
        return overlapping_pairs
    
    def get_union_crop(self, image: Image.Image, row1: pd.Series, row2: pd.Series) -> Image.Image:
        """
        Get the union bounding box crop of two overlapping boxes with padding.
        Padding: 20% horizontal, 10% vertical
        """
        # Calculate union bbox (outer bounds of both boxes)
        min_x = min(row1['BBox_X'], row2['BBox_X'])
        min_y = min(row1['BBox_Y'], row2['BBox_Y'])
        max_x = max(row1['BBox_X'] + row1['BBox_Width'], 
                    row2['BBox_X'] + row2['BBox_Width'])
        max_y = max(row1['BBox_Y'] + row1['BBox_Height'], 
                    row2['BBox_Y'] + row2['BBox_Height'])
        
        # Calculate union dimensions
        union_width = max_x - min_x
        union_height = max_y - min_y
        
        # Add padding: 20% horizontal, 10% vertical
        h_padding = int(union_width * 0.20)
        v_padding = int(union_height * 0.10)
        
        # Apply padding with image boundary checks
        padded_min_x = max(0, min_x - h_padding)
        padded_min_y = max(0, min_y - v_padding)
        padded_max_x = min(image.width, max_x + h_padding)
        padded_max_y = min(image.height, max_y + v_padding)
        
        return image.crop((padded_min_x, padded_min_y, padded_max_x, padded_max_y))
    
    def create_batch_table_image(self, batch_data: List[Dict]) -> Image.Image:
        """
        Create a visual table with all overlapping pairs in the batch.
        
        Args:
            batch_data: List of dicts with keys: 'sr_no', 'crop', 'tag1', 'tag2'
            
        Returns:
            Combined table image
        """
        # Table parameters
        cell_height = 200
        sr_col_width = 80
        img_col_width = 250
        tags_col_width = 300
        total_width = sr_col_width + img_col_width + tags_col_width
        
        header_height = 60
        total_height = header_height + (cell_height * len(batch_data))
        
        # Create white canvas
        table_img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(table_img)
        
        try:
            font_header = ImageFont.truetype("arial.ttf", 20)
            font_cell = ImageFont.truetype("arial.ttf", 16)
        except:
            font_header = ImageFont.load_default()
            font_cell = ImageFont.load_default()
        
        # Draw header
        draw.rectangle([0, 0, total_width, header_height], fill='#2c3e50', outline='black', width=2)
        draw.text((sr_col_width//2, header_height//2), "Sr. No", fill='white', 
                 font=font_header, anchor='mm')
        draw.text((sr_col_width + img_col_width//2, header_height//2), "Overlapping Image", 
                 fill='white', font=font_header, anchor='mm')
        draw.text((sr_col_width + img_col_width + tags_col_width//2, header_height//2), 
                 "Tags to Classify", fill='white', font=font_header, anchor='mm')
        
        # Draw rows
        for i, data in enumerate(batch_data):
            y_offset = header_height + (i * cell_height)
            
            # Draw cell borders
            draw.rectangle([0, y_offset, sr_col_width, y_offset + cell_height], 
                          outline='black', width=2)
            draw.rectangle([sr_col_width, y_offset, sr_col_width + img_col_width, 
                          y_offset + cell_height], outline='black', width=2)
            draw.rectangle([sr_col_width + img_col_width, y_offset, total_width, 
                          y_offset + cell_height], outline='black', width=2)
            
            # Sr. No
            draw.text((sr_col_width//2, y_offset + cell_height//2), str(data['sr_no']), 
                     fill='black', font=font_header, anchor='mm')
            
            # Crop image (resize to fit)
            crop = data['crop'].copy()
            crop.thumbnail((img_col_width - 20, cell_height - 20), Image.Resampling.LANCZOS)
            crop_x = sr_col_width + (img_col_width - crop.width) // 2
            crop_y = y_offset + (cell_height - crop.height) // 2
            table_img.paste(crop, (crop_x, crop_y))
            
            # Tags text
            tags_text = f"Option 1: {data['tag1']}\n\nOption 2: {data['tag2']}"
            text_x = sr_col_width + img_col_width + 10
            text_y = y_offset + cell_height // 2 - 20
            draw.text((text_x, text_y), tags_text, fill='black', font=font_cell)
        
        return table_img
    
    def classify_batch_with_llm(self, batch_data: List[Dict], table_image: Image.Image) -> Dict[int, str]:
        """
        Send batch of overlapping pairs to LLM for classification.
        
        Args:
            batch_data: List of dicts with overlap info
            table_image: Visual table image
            
        Returns:
            Dictionary mapping sr_no to selected tag
        """
        # Build the prompt
        pairs_info = []
        for data in batch_data:
            pairs_info.append(f"  {data['sr_no']}. Options: \"{data['tag1']}\" OR \"{data['tag2']}\"")
        
        pairs_text = "\n".join(pairs_info)
        
        prompt = f"""You are analyzing a table with {len(batch_data)} overlapping tag/icon pairs.

Each row shows:
- Sr. No: Identification number
- Overlapping Image: The union crop of two overlapping detections
- Tags to Classify: Two possible tag names (Option 1 and Option 2)

Your task: For each row, determine which tag option correctly identifies the content in the image.

Here are the pairs:
{pairs_text}

Provide your answer as JSON with this exact structure:
{{
  "classifications": [
    {{"sr_no": 1, "selected_tag": "exact_tag_name", "confidence": "high/medium/low"}},
    {{"sr_no": 2, "selected_tag": "exact_tag_name", "confidence": "high/medium/low"}},
    ...
  ]
}}

Rules:
1. Classify ALL {len(batch_data)} pairs in the table
2. For each pair, choose ONLY from its two given options
3. Use EXACT tag names (case-sensitive)
4. Base decisions on visual content only
"""
        
        # Call Gemini API
        print(f"    Sending batch of {len(batch_data)} pairs to LLM...")
        response = self.model.generate_content([prompt, table_image])
        
        try:
            result = json.loads(response.text)
            classifications = {}
            
            for item in result.get("classifications", []):
                sr_no = item.get("sr_no")
                selected_tag = item.get("selected_tag")
                confidence = item.get("confidence", "unknown")
                
                if sr_no and selected_tag:
                    classifications[sr_no] = selected_tag
                    print(f"    ✓ Sr.{sr_no}: '{selected_tag}' (confidence: {confidence})")
            
            # Check if we got all classifications
            if len(classifications) != len(batch_data):
                print(f"    ⚠ Warning: Expected {len(batch_data)} classifications, got {len(classifications)}")
            
            return classifications
            
        except json.JSONDecodeError as e:
            print(f"    ✗ Error parsing LLM response: {e}")
            print(f"    Raw response: {response.text[:500]}...")
            return {}
    
    def visualize_batch_results(self, batch_data: List[Dict], classifications: Dict[int, str]):
        """Display the batch table and classification results."""
        fig, axes = plt.subplots(len(batch_data), 2, figsize=(14, 4 * len(batch_data)))
        
        if len(batch_data) == 1:
            axes = [axes]
        
        for i, data in enumerate(batch_data):
            sr_no = data['sr_no']
            selected = classifications.get(sr_no, "UNKNOWN")
            
            # Show crop
            axes[i][0].imshow(data['crop'])
            axes[i][0].axis('off')
            axes[i][0].set_title(f"Sr. {sr_no}: Union Crop", fontsize=10, fontweight='bold')
            
            # Show result
            axes[i][1].axis('off')
            result_text = f"Sr. No: {sr_no}\n\n"
            result_text += f"Options:\n  1. {data['tag1']}\n  2. {data['tag2']}\n\n"
            result_text += f"✓ LLM Selected:\n  {selected}"
            
            axes[i][1].text(0.5, 0.5, result_text, fontsize=12, ha='center', va='center',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def process_dataframe(self, df: pd.DataFrame, image: Union[str, Image.Image], 
                         display_results: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main processing function: detect overlapping bbox rows and resolve using LLM in batches.
        
        Args:
            df: Input dataframe with tag data
            image: Original image (PIL Image or file path string)
            display_results: Whether to display visualizations
            
        Returns:
            Tuple of (kept_df, removed_df)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
            print(f"✓ Loaded image: {image.size}")
        
        # Filter for tags only
        tag_df = df[df['Template_Type'] == 'tag'].copy().reset_index(drop=True)
        
        if tag_df.empty:
            print("✗ No tags found in dataframe")
            return df, pd.DataFrame()
        
        print(f"\n{'='*70}")
        print(f"STEP 1: Finding Overlapping Bounding Boxes")
        print(f"{'='*70}")
        print(f"Total tags to analyze: {len(tag_df)}")
        print(f"Overlap threshold: {self.overlap_threshold*100}% (90%+)")
        print(f"Batch size: {self.batch_size} pairs per LLM call")
        
        # Find all overlapping pairs
        overlapping_pairs = self.find_overlapping_pairs(tag_df)
        
        if not overlapping_pairs:
            print(f"\n✓ No overlapping tags found! All {len(tag_df)} tags are unique.")
            return tag_df, pd.DataFrame()
        
        print(f"\n✓ Found {len(overlapping_pairs)} overlapping pairs!")
        
        # Track which indices to remove
        indices_to_remove = set()
        all_classifications = {}
        
        print(f"\n{'='*70}")
        print(f"STEP 2: Processing in Batches")
        print(f"{'='*70}")
        
        # Process in batches
        num_batches = (len(overlapping_pairs) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(num_batches):
            batch_start = batch_num * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(overlapping_pairs))
            batch_pairs = overlapping_pairs[batch_start:batch_end]
            
            print(f"\n--- Batch {batch_num + 1}/{num_batches} ---")
            print(f"Processing pairs {batch_start + 1} to {batch_end}")
            
            # Prepare batch data
            batch_data = []
            for local_idx, (idx1, idx2, overlap1, overlap2) in enumerate(batch_pairs):
                sr_no = batch_start + local_idx + 1
                row1 = tag_df.iloc[idx1]
                row2 = tag_df.iloc[idx2]
                
                # Get union crop
                union_crop = self.get_union_crop(image, row1, row2)
                
                batch_data.append({
                    'sr_no': sr_no,
                    'crop': union_crop,
                    'tag1': row1['Tag_Name'],
                    'tag2': row2['Tag_Name'],
                    'idx1': idx1,
                    'idx2': idx2,
                    'row1': row1,
                    'row2': row2
                })
                
                print(f"  {sr_no}. Row {idx1} ('{row1['Tag_Name']}') vs Row {idx2} ('{row2['Tag_Name']}')")
            
            # Create table image
            table_image = self.create_batch_table_image(batch_data)
            
            if display_results:
                plt.figure(figsize=(12, 2 * len(batch_data)))
                plt.imshow(table_image)
                plt.axis('off')
                plt.title(f"Batch {batch_num + 1}/{num_batches} - Overlapping Pairs Table", 
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
            
            # Classify batch
            classifications = self.classify_batch_with_llm(batch_data, table_image)
            all_classifications.update(classifications)
            
            # Process results
            for data in batch_data:
                sr_no = data['sr_no']
                selected_tag = classifications.get(sr_no)
                
                if selected_tag:
                    # Determine which to keep
                    if selected_tag == data['tag1']:
                        keep_idx = data['idx1']
                        remove_idx = data['idx2']
                    else:
                        keep_idx = data['idx2']
                        remove_idx = data['idx1']
                    
                    indices_to_remove.add(remove_idx)
                    print(f"  → Sr.{sr_no}: Keeping row {keep_idx}, removing row {remove_idx}")
                else:
                    print(f"  ⚠ Sr.{sr_no}: No classification received, keeping row {data['idx1']}")
            
            # Display batch results
            if display_results:
                self.visualize_batch_results(batch_data, classifications)
        
        # Create output dataframes
        # kept_df: All tags EXCEPT the ones marked for removal (includes non-overlapping + resolved overlapping)
        kept_df = tag_df[~tag_df.index.isin(indices_to_remove)].copy()
        
        # removed_df: Only the duplicate tags that were removed
        removed_df = tag_df[tag_df.index.isin(indices_to_remove)].copy()
        
        # Calculate statistics
        all_overlapping_indices = set()
        for idx1, idx2, _, _ in overlapping_pairs:
            all_overlapping_indices.add(idx1)
            all_overlapping_indices.add(idx2)
        
        non_overlapping_count = len(tag_df) - len(all_overlapping_indices)
        resolved_overlapping_count = len(all_overlapping_indices) - len(removed_df)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"  Total tags analyzed: {len(tag_df)}")
        print(f"  Non-overlapping tags (kept as-is): {non_overlapping_count}")
        print(f"  Overlapping pairs found: {len(overlapping_pairs)}")
        print(f"  Batches processed: {num_batches}")
        print(f"  LLM calls made: {num_batches} (vs {len(overlapping_pairs)} if done individually)")
        print(f"  Resolved overlapping tags (kept): {resolved_overlapping_count}")
        print(f"  Duplicate tags (removed): {len(removed_df)}")
        print(f"  ---")
        print(f"  FINAL kept_df size: {len(kept_df)} = {non_overlapping_count} non-overlapping + {resolved_overlapping_count} resolved")
        print(f"  Token savings: ~{((len(overlapping_pairs) - num_batches) / len(overlapping_pairs) * 100):.1f}%")
        print(f"{'='*70}\n")
        
        return kept_df, removed_df


# Example usage
def main():
    """Example usage of the TagOverlapResolver module."""
    
    # Initialize
    API_KEY = "your-gemini-api-key-here"
    resolver = TagOverlapResolver(
        api_key=API_KEY, 
        overlap_threshold=0.9,
        batch_size=10  # Process 10 pairs per LLM call
    )
    
    # Load your dataframe
    df = pd.read_csv("your_data.csv")
    
    # Load your image (can pass path string or PIL Image)
    image_path = "your_image.png"
    
    # Process with visualizations
    kept_df, removed_df = resolver.process_dataframe(
        df=df,
        image=image_path,
        display_results=True  # Shows batch tables and results
    )
    
    # Save results
    kept_df.to_csv("kept_tags.csv", index=False)
    removed_df.to_csv("removed_duplicate_tags.csv", index=False)
    
    print("\n✓ Results saved!")
    print(f"  Kept tags: kept_tags.csv ({len(kept_df)} rows)")
    print(f"  Removed duplicates: removed_duplicate_tags.csv ({len(removed_df)} rows)")


if __name__ == "__main__":
    main()