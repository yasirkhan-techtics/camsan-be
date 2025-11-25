import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import google.generativeai as genai
from typing import List, Dict, Tuple
import json
import re
import matplotlib.pyplot as plt
from IPython.display import display

class DetectionVerificationPipeline:
    """
    Enhanced Pipeline for verifying detection results using LLM.
    Handles both tags and icons with separate confidence thresholds.
    Now with improved debugging, visualization, and index tracking.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the pipeline.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temp_folder = "temp_crops"
        self.table_folder = "verification_tables"
        self.approved_folder = "auto_approved_visuals"
        os.makedirs(self.temp_folder, exist_ok=True)
        os.makedirs(self.table_folder, exist_ok=True)
        os.makedirs(self.approved_folder, exist_ok=True)
        
        # Track processing for debugging
        self.processing_log = []
    
    def calculate_confidence_threshold(self, confidences: List[float], 
                                      approaches: List[str] = ['iqr', 'percentile', 'std', 'kmeans']) -> float:
        """
        Calculate dynamic confidence threshold using multiple approaches.
        Uses MINIMUM of thresholds to be more conservative (fewer false negatives).
        
        Args:
            confidences: List of confidence scores
            approaches: List of approaches to use
            
        Returns:
            Combined threshold value
        """
        if len(confidences) < 3:
            return min(confidences) * 0.9 if confidences else 0.5
        
        confidences = np.array(confidences)
        thresholds = []
        
        # IQR-based threshold
        if 'iqr' in approaches:
            q1 = np.percentile(confidences, 25)
            q3 = np.percentile(confidences, 75)
            iqr = q3 - q1
            threshold_iqr = q1 - 1.5 * iqr
            thresholds.append(max(threshold_iqr, 0.0))
        
        # Percentile-based (15th percentile)
        if 'percentile' in approaches:
            threshold_percentile = np.percentile(confidences, 15)
            thresholds.append(threshold_percentile)
        
        # Standard deviation-based
        if 'std' in approaches:
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            threshold_std = mean_conf - 1.5 * std_conf
            thresholds.append(max(threshold_std, 0.0))
        
        # Simple K-means clustering (2 clusters)
        if 'kmeans' in approaches and len(confidences) >= 5:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(confidences.reshape(-1, 1))
                centers = sorted(kmeans.cluster_centers_.flatten())
                threshold_kmeans = (centers[0] + centers[1]) / 2
                thresholds.append(threshold_kmeans)
            except:
                pass
        
        # CRITICAL FIX: Use MINIMUM to be more conservative (fewer false negatives)
        # This ensures we only auto-approve the most confident detections
        if thresholds:
            combined_threshold = np.max(thresholds)
            print(f"         Individual thresholds: {[f'{t:.4f}' for t in thresholds]}")
            print(f"         Selected threshold (min): {combined_threshold:.4f}")
            return float(combined_threshold)
        
        return 0.5
    
    def filter_detections(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Filter detections into high and low confidence groups.
        PRESERVES ORIGINAL INDICES throughout the process.
        
        Args:
            df: DataFrame with detection results
            
        Returns:
            high_conf_df, low_conf_df, thresholds_dict
        """
        print("\n" + "="*80)
        print("STEP 1: FILTERING DETECTIONS BY TEMPLATE TYPE")
        print("="*80)
        
        # Check required columns
        required_cols = ['Template_Type', 'Tag_Name', 'Confidence', 'Symbol_ID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # CRITICAL: Work with original indices throughout
        # Create boolean masks instead of copying dataframes
        is_tag = df['Template_Type'].str.lower().str.contains('tag|cf', case=False, na=False)
        is_icon = df['Template_Type'].str.lower().str.contains('icon', case=False, na=False)
        
        tag_indices = df.index[is_tag].tolist()
        icon_indices = df.index[is_icon].tolist()
        
        print(f"\n📊 Initial Separation by Template_Type:")
        print(f"   Total detections: {len(df)}")
        print(f"   Tags (containing 'tag' or 'CF'): {len(tag_indices)}")
        print(f"   Icons (containing 'icon'): {len(icon_indices)}")
        print(f"   Tag indices: {tag_indices[:20]}{'...' if len(tag_indices) > 20 else ''}")
        print(f"   Icon indices: {icon_indices[:20]}{'...' if len(icon_indices) > 20 else ''}")
        
        thresholds = {}
        high_conf_indices = []
        low_conf_indices = []
        
        print(f"\n🎯 Calculating Dynamic Thresholds...")
        
        # Process tags by Tag_Name
        if len(tag_indices) > 0:
            df_tags = df.loc[tag_indices]
            unique_tags = df_tags['Tag_Name'].unique()
            print(f"\n   Found {len(unique_tags)} unique Tag_Names: {list(unique_tags)}")
            
            for tag_name in unique_tags:
                # Get indices for this specific tag
                tag_mask = (df['Tag_Name'] == tag_name) & is_tag
                tag_group_indices = df.index[tag_mask].tolist()
                tag_group = df.loc[tag_group_indices]
                
                confidences = tag_group['Confidence'].tolist()
                
                threshold = self.calculate_confidence_threshold(confidences)
                thresholds[f'tag_{tag_name}'] = threshold
                
                # Separate by threshold while keeping original indices
                high_mask = tag_group['Confidence'] >= threshold
                low_mask = tag_group['Confidence'] < threshold
                
                high_indices = tag_group.index[high_mask].tolist()
                low_indices = tag_group.index[low_mask].tolist()
                
                print(f"\n   📌 Tag_Name: '{tag_name}'")
                print(f"      Total detections: {len(tag_group_indices)}")
                print(f"      Original indices: {tag_group_indices}")
                print(f"      Calculated threshold: {threshold:.4f}")
                print(f"      Confidence range: [{min(confidences):.4f}, {max(confidences):.4f}]")
                print(f"      High confidence (>= {threshold:.4f}): {len(high_indices)} detections")
                print(f"      High conf indices: {high_indices}")
                print(f"      Low confidence (< {threshold:.4f}): {len(low_indices)} detections")
                print(f"      Low conf indices: {low_indices}")
                
                high_conf_indices.extend(high_indices)
                low_conf_indices.extend(low_indices)
        
        # Process icons (all together)
        if len(icon_indices) > 0:
            df_icons = df.loc[icon_indices]
            confidences = df_icons['Confidence'].tolist()
            threshold = self.calculate_confidence_threshold(confidences)
            thresholds['icon_all'] = threshold
            
            # Separate by threshold while keeping original indices
            high_mask = df_icons['Confidence'] >= threshold
            low_mask = df_icons['Confidence'] < threshold
            
            high_indices = df_icons.index[high_mask].tolist()
            low_indices = df_icons.index[low_mask].tolist()
            
            print(f"\n   📌 Icons (all types):")
            print(f"      Total detections: {len(icon_indices)}")
            print(f"      Original indices: {icon_indices}")
            print(f"      Calculated threshold: {threshold:.4f}")
            print(f"      Confidence range: [{min(confidences):.4f}, {max(confidences):.4f}]")
            print(f"      High confidence (>= {threshold:.4f}): {len(high_indices)} detections")
            print(f"      High conf indices: {high_indices}")
            print(f"      Low confidence (< {threshold:.4f}): {len(low_indices)} detections")
            print(f"      Low conf indices: {low_indices}")
            
            high_conf_indices.extend(high_indices)
            low_conf_indices.extend(low_indices)
        
        # Create final dataframes using original indices
        high_conf_df = df.loc[high_conf_indices].copy() if high_conf_indices else pd.DataFrame()
        low_conf_df = df.loc[low_conf_indices].copy() if low_conf_indices else pd.DataFrame()
        
        print(f"\n" + "-"*80)
        print(f"✅ FILTERING COMPLETE")
        print(f"   Total High Confidence: {len(high_conf_df)} (Auto-approved)")
        print(f"   High conf original indices: {high_conf_df.index.tolist()}")
        print(f"   Total Low Confidence: {len(low_conf_df)} (Need LLM verification)")
        print(f"   Low conf original indices: {low_conf_df.index.tolist()}")
        print("-"*80)
        
        # Verification check
        total_separated = len(high_conf_indices) + len(low_conf_indices)
        total_processed = len(tag_indices) + len(icon_indices)
        print(f"\n🔍 Sanity Check:")
        print(f"   Total tag+icon detections: {total_processed}")
        print(f"   Total separated (high+low): {total_separated}")
        print(f"   Match: {'✓ PASS' if total_separated == total_processed else '✗ FAIL - DATA LOSS!'}")
        
        # Display filtered dataframes with MORE details
        if len(high_conf_df) > 0:
            print(f"\n📋 High Confidence Detections (Auto-Approved) - First 10:")
            display_df = high_conf_df[['Symbol_ID', 'Tag_Name', 'Template_Type', 'Confidence']].head(10)
            display_df_with_idx = display_df.copy()
            display_df_with_idx['Original_Index'] = display_df.index
            print(display_df_with_idx.to_string(index=False))
            
            if len(high_conf_df) > 10:
                print(f"   ... and {len(high_conf_df) - 10} more")
        
        if len(low_conf_df) > 0:
            print(f"\n📋 Low Confidence Detections (Need Verification) - First 10:")
            display_df = low_conf_df[['Symbol_ID', 'Tag_Name', 'Template_Type', 'Confidence']].head(10)
            display_df_with_idx = display_df.copy()
            display_df_with_idx['Original_Index'] = display_df.index
            print(display_df_with_idx.to_string(index=False))
            
            if len(low_conf_df) > 10:
                print(f"   ... and {len(low_conf_df) - 10} more")
        
        return high_conf_df, low_conf_df, thresholds
    
    def crop_detection(self, image_path: str, row: pd.Series, 
                      padding_percent: float, output_path: str) -> str:
        """
        Crop detection from image with padding.
        
        Args:
            image_path: Path to original image
            row: DataFrame row with bbox info
            padding_percent: Padding percentage (e.g., 0.4 for 40%)
            output_path: Path to save cropped image
            
        Returns:
            Path to cropped image
        """
        img = Image.open(image_path)
        
        # Calculate bbox with padding
        x = int(row['BBox_X'])
        y = int(row['BBox_Y'])
        w = int(row['BBox_Width'])
        h = int(row['BBox_Height'])
        
        pad_x = int(w * padding_percent)
        pad_y = int(h * padding_percent)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.width, x + w + pad_x)
        y2 = min(img.height, y + h + pad_y)
        
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(output_path)
        
        return output_path
    
    def create_verification_table(self, crop_paths: List[str], 
                                  indices: List[int], 
                                  output_path: str,
                                  template_path: str = None,
                                  original_indices: List[int] = None,
                                  metadata: List[Dict] = None) -> str:
        """
        Create verification table image with crops and dynamic cell sizing.
        Now includes original index and metadata for debugging.
        
        Args:
            crop_paths: List of paths to cropped images
            indices: List of serial numbers (1, 2, 3...)
            output_path: Path to save table
            template_path: Optional template image path for icons
            original_indices: Original dataframe indices
            metadata: Additional metadata for each detection
            
        Returns:
            Path to table image
        """
        border_width = 3
        cell_padding = 20
        font_size = 24
        small_font_size = 18
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            font_bold = ImageFont.truetype("arialbd.ttf", font_size)
            small_font = ImageFont.truetype("arial.ttf", small_font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", small_font_size)
            except:
                font = ImageFont.load_default()
                font_bold = font
                small_font = font
        
        # Load crops
        crops = [Image.open(p) for p in crop_paths]
        
        # Resize crops to have consistent maximum dimensions while maintaining aspect ratio
        max_crop_width = 400
        max_crop_height = 300
        resized_crops = []
        
        for crop in crops:
            # Calculate scaling factor
            width_scale = max_crop_width / crop.width if crop.width > max_crop_width else 1.0
            height_scale = max_crop_height / crop.height if crop.height > max_crop_height else 1.0
            scale = min(width_scale, height_scale)
            
            if scale < 1.0:
                new_width = int(crop.width * scale)
                new_height = int(crop.height * scale)
                resized_crop = crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_crops.append(resized_crop)
            else:
                resized_crops.append(crop)
        
        # Calculate column widths based on actual content
        col1_width = 100  # Sr. No
        col2_width = 120  # Original Index (NEW)
        col3_width = max(img.width for img in resized_crops) + 2 * cell_padding
        col3_width = max(col3_width, 200)  # Minimum width
        
        if template_path:
            template_img = Image.open(template_path)
            # Resize template to fit nicely
            if template_img.width > 350 or template_img.height > 300:
                width_scale = 350 / template_img.width if template_img.width > 350 else 1.0
                height_scale = 300 / template_img.height if template_img.height > 300 else 1.0
                scale = min(width_scale, height_scale)
                new_width = int(template_img.width * scale)
                new_height = int(template_img.height * scale)
                template_img = template_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            col4_width = template_img.width + 2 * cell_padding
            col4_width = max(col4_width, 200)  # Minimum width
            col5_width = 280  # Verification result
            total_width = col1_width + col2_width + col3_width + col4_width + col5_width + (6 * border_width)
            headers = ["Sr. No", "DF Index", "Detection", "Template", "Same or Not"]
            col_widths = [col1_width, col2_width, col3_width, col4_width, col5_width]
        else:
            col4_width = 280  # Verification result
            total_width = col1_width + col2_width + col3_width + col4_width + (5 * border_width)
            headers = ["Sr. No", "DF Index", "Image", "Valid or Not"]
            col_widths = [col1_width, col2_width, col3_width, col4_width]
        
        # Calculate row heights dynamically for each row
        header_height = 70
        row_heights = []
        for crop in resized_crops:
            if template_path:
                row_height = max(crop.height, template_img.height) + 2 * cell_padding
            else:
                row_height = crop.height + 2 * cell_padding
            row_height = max(row_height, 120)  # Increased minimum for metadata
            row_heights.append(row_height)
        
        total_height = header_height + sum(row_heights) + ((len(resized_crops) + 1) * border_width)
        
        # Create canvas
        table_img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(table_img)
        
        # Draw header
        draw.rectangle([0, 0, total_width, header_height], fill='#E8E8E8')
        draw.line([(0, 0), (total_width, 0)], fill='black', width=border_width)
        draw.line([(0, header_height), (total_width, header_height)], fill='black', width=border_width)
        
        # Header borders and text
        x = 0
        for i, (header, col_width) in enumerate(zip(headers, col_widths)):
            draw.line([(x, 0), (x, header_height)], fill='black', width=border_width)
            draw.text((x + border_width + col_width//2, header_height//2), 
                     header, fill='black', font=font_bold, anchor='mm')
            x += col_width + border_width
        
        draw.line([(x, 0), (x, header_height)], fill='black', width=border_width)
        
        # Draw data rows with dynamic heights
        y = header_height + border_width
        
        for idx, crop, row_height, orig_idx, meta in zip(indices, resized_crops, row_heights, 
                                                          original_indices or indices, 
                                                          metadata or [{}]*len(indices)):
            # Row borders
            draw.line([(0, y), (0, y + row_height)], fill='black', width=border_width)
            
            x = 0
            for col_width in col_widths:
                x += col_width + border_width
                draw.line([(x, y), (x, y + row_height)], fill='black', width=border_width)
            
            draw.line([(0, y + row_height), (total_width, y + row_height)], 
                     fill='black', width=border_width)
            
            # Sr. No
            draw.text((border_width + col1_width//2, y + row_height//2), 
                     str(idx), fill='black', font=font, anchor='mm')
            
            # Original Index
            draw.text((border_width + col1_width + border_width + col2_width//2, y + row_height//2), 
                     str(orig_idx), fill='blue', font=font_bold, anchor='mm')
            
            # Detection image - centered in cell
            img_x = border_width + col1_width + border_width + col2_width + border_width
            img_x += (col3_width - crop.width) // 2
            img_y = y + (row_height - crop.height) // 2
            table_img.paste(crop, (img_x, img_y))
            
            # Add metadata text below image if provided
            if meta:
                meta_text = f"Conf: {meta.get('confidence', 'N/A')}"
                meta_y = img_y + crop.height + 5
                draw.text((img_x + crop.width//2, meta_y), 
                         meta_text, fill='#666', font=small_font, anchor='mt')
            
            # Template image (if provided) - centered in cell
            if template_path:
                img_x = border_width + col1_width + border_width + col2_width + border_width + col3_width + border_width
                img_x += (col4_width - template_img.width) // 2
                img_y = y + (row_height - template_img.height) // 2
                table_img.paste(template_img, (img_x, img_y))
            
            y += row_height + border_width
        
        table_img.save(output_path)
        return output_path
    
    def visualize_auto_approved(self, df: pd.DataFrame, high_conf_df: pd.DataFrame, 
                               image_path: str, thresholds: Dict):
        """
        Create visualization of auto-approved high confidence detections.
        Shows what was automatically approved without LLM verification.
        """
        if len(high_conf_df) == 0:
            print("\n   No auto-approved detections to visualize")
            return
        
        print(f"\n📸 Creating visualizations for {len(high_conf_df)} auto-approved detections...")
        
        # Group by Template_Type and Tag_Name
        is_tag = high_conf_df['Template_Type'].str.lower().str.contains('tag|cf', case=False, na=False)
        
        # Process tags
        tags_df = high_conf_df[is_tag]
        if len(tags_df) > 0:
            unique_tags = tags_df['Tag_Name'].unique()
            for tag_name in unique_tags:
                tag_group = tags_df[tags_df['Tag_Name'] == tag_name]
                threshold = thresholds.get(f'tag_{tag_name}', 0.5)
                
                print(f"\n   Visualizing auto-approved tag '{tag_name}': {len(tag_group)} detections")
                
                # Create crops
                crop_paths = []
                original_indices = []
                metadata_list = []
                
                for orig_idx, row in tag_group.iterrows():
                    crop_path = os.path.join(self.approved_folder, f"approved_tag_{tag_name}_{orig_idx}.png")
                    self.crop_detection(image_path, row, 0.4, crop_path)
                    crop_paths.append(crop_path)
                    original_indices.append(orig_idx)
                    metadata_list.append({
                        'confidence': f"{row['Confidence']:.4f}",
                        'symbol_id': row['Symbol_ID']
                    })
                
                # Create table (max 15 per table)
                for batch_start in range(0, len(crop_paths), 15):
                    batch_end = min(batch_start + 15, len(crop_paths))
                    batch_crops = crop_paths[batch_start:batch_end]
                    batch_indices = list(range(batch_start + 1, batch_end + 1))
                    batch_orig_indices = original_indices[batch_start:batch_end]
                    batch_metadata = metadata_list[batch_start:batch_end]
                    
                    table_name = f"AUTO_APPROVED_tag_{tag_name}_batch_{batch_start//15 + 1}.png"
                    table_path = os.path.join(self.approved_folder, table_name)
                    
                    self.create_verification_table(
                        batch_crops, batch_indices, table_path,
                        original_indices=batch_orig_indices,
                        metadata=batch_metadata
                    )
                    
                    # Display
                    img = Image.open(table_path)
                    plt.figure(figsize=(18, 12))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"AUTO-APPROVED: Tag '{tag_name}' (Threshold: {threshold:.4f})\nBatch {batch_start//15 + 1}", 
                             fontsize=16, fontweight='bold', color='green')
                    plt.tight_layout()
                    plt.show()
        
        # Process icons
        icons_df = high_conf_df[~is_tag]
        if len(icons_df) > 0:
            threshold = thresholds.get('icon_all', 0.5)
            print(f"\n   Visualizing auto-approved icons: {len(icons_df)} detections")
            
            # Create crops
            crop_paths = []
            original_indices = []
            metadata_list = []
            
            for orig_idx, row in icons_df.iterrows():
                crop_path = os.path.join(self.approved_folder, f"approved_icon_{orig_idx}.png")
                self.crop_detection(image_path, row, 0.3, crop_path)
                crop_paths.append(crop_path)
                original_indices.append(orig_idx)
                metadata_list.append({
                    'confidence': f"{row['Confidence']:.4f}",
                    'symbol_id': row['Symbol_ID']
                })
            
            # Create table (max 15 per table)
            for batch_start in range(0, len(crop_paths), 15):
                batch_end = min(batch_start + 15, len(crop_paths))
                batch_crops = crop_paths[batch_start:batch_end]
                batch_indices = list(range(batch_start + 1, batch_end + 1))
                batch_orig_indices = original_indices[batch_start:batch_end]
                batch_metadata = metadata_list[batch_start:batch_end]
                
                table_name = f"AUTO_APPROVED_icons_batch_{batch_start//15 + 1}.png"
                table_path = os.path.join(self.approved_folder, table_name)
                
                self.create_verification_table(
                    batch_crops, batch_indices, table_path,
                    original_indices=batch_orig_indices,
                    metadata=batch_metadata
                )
                
                # Display
                img = Image.open(table_path)
                plt.figure(figsize=(18, 12))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"AUTO-APPROVED: Icons (Threshold: {threshold:.4f})\nBatch {batch_start//15 + 1}", 
                         fontsize=16, fontweight='bold', color='green')
                plt.tight_layout()
                plt.show()
        
        print(f"   ✓ Auto-approved visualizations complete")
    
    def display_table(self, table_path: str):
        """Display table image using matplotlib."""
        img = Image.open(table_path)
        plt.figure(figsize=(18, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Verification Table: {os.path.basename(table_path)}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def verify_with_llm(self, table_path: str, verification_type: str, 
                       tag_name: str = None, num_items: int = 0) -> List[bool]:
        """
        Verify detections using LLM.
        
        Args:
            table_path: Path to verification table
            verification_type: 'tag' or 'icon'
            tag_name: Tag name if verifying tags
            num_items: Number of items in the batch
            
        Returns:
            List of verification results (True = valid, False = invalid)
        """
        if verification_type == 'tag':
            prompt = f"""You are verifying detection results for '{tag_name}' tags in a technical drawing.

The table shows {num_items} detected instances with serial numbers. Your task is to verify if each detection is a VALID instance of '{tag_name}'.

Context:
- These are technical P&ID (Piping and Instrumentation Diagram) symbols
- '{tag_name}' represents a specific type of tag/label in the drawing
- Valid detections should clearly show the tag text '{tag_name}' or similar variations
- Invalid detections might be: partial captures, wrong text, noise, or completely different elements
- Only treat items as a continuation if their labels match exactly; similar-looking labels must be considered entirely separate and not part of the same sequence.

Instructions:
1. Examine each row carefully (Serial numbers 1 through {num_items})
2. For EACH serial number, determine if it's a valid '{tag_name}' detection
3. Respond with ONLY a JSON array of exactly {num_items} boolean values, one per row in order

Example response format for {num_items} items:
[true, true, false, true, false, true, true]

Where:
- true = Valid detection of '{tag_name}'
- false = Invalid detection (wrong, partial, or not '{tag_name}')

CRITICAL: You MUST provide exactly {num_items} boolean values in the array, no more, no less.

Respond with ONLY the JSON array, no other text or explanation."""

        else:  # icon
            prompt = f"""You are verifying icon detection results in a technical drawing by comparing detected instances with a reference template.

The table shows {num_items} detected instances:
- Column 1: Serial number (1 through {num_items})
- Column 2: DF Index (dataframe index for tracking)
- Column 3: Detected icon instance
- Column 4: Reference template icon
- Column 5: Your verification (to be filled)

Context:
- These are technical symbols/icons from P&ID diagrams
- The template (Column 4) shows what the correct icon should look like
- Detected instances (Column 3) should closely match the template in shape, structure, and key features
- Colors, line thickness, and other minor variations are acceptable, as long as the overall shape and structure match
- They are be noise as line passing through the icon or texture of the background, etc.

Instructions:
1. Compare each detected instance with the reference template
2. Focus on: overall shape, key structural elements, symbol characteristics
3. Minor variations in size, rotation, or line thickness are acceptable
4. For EACH serial number (1 through {num_items}), determine if the detected instance MATCHES the template

Example response format for {num_items} items:
[true, true, false, true, true, false, true]

Where:
- true = Detection matches template (valid)
- false = Detection does NOT match template (invalid)

CRITICAL: You MUST provide exactly {num_items} boolean values in the array, no more, no less.

Be strict but reasonable - the detection should clearly represent the same symbol type as the template.

Respond with ONLY the JSON array, no other text or explanation."""
        
        # Send to LLM
        print(f"      🤖 Sending to LLM for verification...")
        
        try:
            image = Image.open(table_path)
            response = self.model.generate_content([prompt, image])
            
            # Get response text
            response_text = response.text.strip()
            
            print(f"\n      📝 LLM Raw Response:")
            print(f"         {response_text}")
            
            # Parse response - look for JSON array
            json_match = re.search(r'\[[\s\w,]+\]', response_text)
            if json_match:
                results = json.loads(json_match.group())
                
                print(f"\n      ✅ Parsed Results: {results}")
                print(f"         Expected {num_items} items, got {len(results)} items")
                
                if len(results) != num_items:
                    print(f"         ⚠️  WARNING: Count mismatch!")
                    return []
                
                # Display individual results
                print(f"\n      📊 Item-by-Item Results:")
                approved_count = 0
                for i, result in enumerate(results, 1):
                    status = "✓ VALID" if result else "✗ INVALID"
                    if result:
                        approved_count += 1
                    print(f"         Item {i}: {status}")
                
                print(f"\n      Summary: {approved_count}/{num_items} approved, {num_items - approved_count}/{num_items} rejected")
                
                return results
            else:
                print(f"         ❌ ERROR: Could not parse JSON from response")
                return []
                
        except Exception as e:
            print(f"         ❌ ERROR in LLM verification: {e}")
            return []
    
    def process_pipeline(self, df: pd.DataFrame, image_path: str, 
                        template_path: str = None, batch_size: int = 10) -> pd.DataFrame:
        """
        Main pipeline to process and verify detections.
        
        Args:
            df: DataFrame with detection results
            image_path: Path to original image
            template_path: Path to template image for icons
            batch_size: Number of detections per batch
            
        Returns:
            Updated DataFrame with LLM_Approved column
        """
        print("\n" + "🔷"*40)
        print("DETECTION VERIFICATION PIPELINE STARTED")
        print("🔷"*40)
        print(f"\n📊 Input DataFrame Info:")
        print(f"   Total rows: {len(df)}")
        print(f"   Index range: {df.index.min()} to {df.index.max()}")
        print(f"   Index type: {type(df.index)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Initialize verification columns
        df['LLM_Approved'] = pd.Series([None] * len(df), dtype=object)
        df['Auto_Approved'] = False  # Track which were auto-approved based on high confidence
        df['Verification_Method'] = 'Pending'  # Track how each was verified
        
        # Log initial state
        print(f"\n   LLM_Approved column initialized:")
        print(f"   - None count: {df['LLM_Approved'].isna().sum()}")
        
        # Step 1: Filter detections
        high_conf_df, low_conf_df, thresholds = self.filter_detections(df)
        
        # CRITICAL: Visualize auto-approved detections
        print(f"\n" + "="*80)
        print(f"STEP 1.5: VISUALIZING AUTO-APPROVED DETECTIONS")
        print(f"="*80)
        self.visualize_auto_approved(df, high_conf_df, image_path, thresholds)
        
        # Mark high confidence as approved (using original indices)
        if len(high_conf_df) > 0:
            high_conf_original_indices = high_conf_df.index.tolist()
            print(f"\n✅ Marking {len(high_conf_df)} high-confidence detections as approved")
            print(f"   Indices being marked: {high_conf_original_indices}")
            
            for orig_idx in high_conf_original_indices:
                if orig_idx in df.index:
                    df.at[orig_idx, 'LLM_Approved'] = True
                    df.at[orig_idx, 'Auto_Approved'] = True  # Mark as auto-approved
                    df.at[orig_idx, 'Verification_Method'] = 'Auto_High_Confidence'
                    
                    # Log for verification
                    self.processing_log.append({
                        'index': orig_idx,
                        'action': 'auto_approved',
                        'symbol_id': df.at[orig_idx, 'Symbol_ID'],
                        'tag_name': df.at[orig_idx, 'Tag_Name'],
                        'confidence': df.at[orig_idx, 'Confidence'],
                        'template_type': df.at[orig_idx, 'Template_Type']
                    })
                else:
                    print(f"   ⚠️  WARNING: Index {orig_idx} not found in original df!")
            
            # Verify update
            approved_count = (df['LLM_Approved'] == True).sum()
            auto_approved_count = (df['Auto_Approved'] == True).sum()
            print(f"   ✓ Verification: {approved_count} rows now marked as True")
            print(f"   ✓ Auto-approved count: {auto_approved_count}")
            
            # CRITICAL: Show which specific detections were auto-approved
            print(f"\n   📋 Auto-Approved Detections Detail:")
            for orig_idx in high_conf_original_indices[:10]:  # Show first 10
                if orig_idx in df.index:
                    row = df.loc[orig_idx]
                    print(f"      Index {orig_idx}: Symbol_ID={row['Symbol_ID']}, "
                          f"Tag={row['Tag_Name']}, Conf={row['Confidence']:.4f}, "
                          f"LLM_Approved={row['LLM_Approved']}, Auto_Approved={row['Auto_Approved']}")
            if len(high_conf_original_indices) > 10:
                print(f"      ... and {len(high_conf_original_indices) - 10} more")
        
        # Process low confidence detections
        if len(low_conf_df) == 0:
            print("\n✅ No low-confidence detections to verify")
            self._print_final_summary(df)
            return df
        
        print(f"\n" + "="*80)
        print(f"STEP 2: LLM VERIFICATION OF LOW CONFIDENCE DETECTIONS")
        print(f"="*80)
        
        # Separate tags and icons from low confidence
        is_tag_low = low_conf_df['Template_Type'].str.lower().str.contains('tag|cf', case=False, na=False)
        low_tags = low_conf_df[is_tag_low]
        low_icons = low_conf_df[~is_tag_low]
        
        print(f"\n📋 Low confidence breakdown:")
        print(f"   Tags: {len(low_tags)}")
        print(f"   Icons: {len(low_icons)}")
        
        # Process tags - SEPARATE BATCH FOR EACH TAG_NAME
        if len(low_tags) > 0:
            print(f"\n" + "-"*80)
            print(f"PROCESSING TAGS (SEPARATE BATCHES PER TAG_NAME)")
            print("-"*80)
            
            unique_tags = low_tags['Tag_Name'].unique()
            for tag_name in unique_tags:
                # Use .loc to get the right rows from low_tags
                tag_mask = low_tags['Tag_Name'] == tag_name
                tag_group = low_tags[tag_mask]
                
                print(f"\n🏷️  Processing Tag_Name: '{tag_name}' ({len(tag_group)} detections)")
                print(f"   Indices: {tag_group.index.tolist()}")
                
                self._process_group(df, tag_group, image_path, 'tag', 
                                  tag_name, batch_size, padding=0.4, template_path=None)
        
        # Process icons
        if len(low_icons) > 0:
            print(f"\n" + "-"*80)
            print(f"PROCESSING ICONS")
            print("-"*80)
            print(f"\n🎨 Processing all icons ({len(low_icons)} detections)")
            print(f"   Indices: {low_icons.index.tolist()}")
            
            self._process_group(df, low_icons, image_path, 'icon', 
                              None, batch_size, padding=0.3, 
                              template_path=template_path)
        
        # Final summary
        self._print_final_summary(df)
        
        # Clean up temp files
        self._cleanup_temp_files()
        
        print("\n" + "🔷"*40)
        print("PIPELINE COMPLETE")
        print("🔷"*40)
        
        return df
    
    def _process_group(self, df: pd.DataFrame, group: pd.DataFrame, 
                      image_path: str, verification_type: str, 
                      tag_name: str, batch_size: int, padding: float,
                      template_path: str = None):
        """
        Process a group of detections in batches.
        CRITICAL: Preserves original indices throughout.
        CRITICAL: Only processes detections that are NOT already auto-approved.
        """
        
        # CRITICAL FIX: Filter out any detections that are already auto-approved
        # This prevents overwriting high-confidence approvals
        group_original_indices = group.index.tolist()
        
        # Check which indices are already approved
        already_approved = []
        indices_to_process = []
        
        for idx in group_original_indices:
            if idx in df.index and df.at[idx, 'Auto_Approved'] == True:
                already_approved.append(idx)
                print(f"   ⚠️  Skipping index {idx} - already auto-approved (high confidence)")
            else:
                indices_to_process.append(idx)
        
        if already_approved:
            print(f"\n   🛡️  Protection: Skipped {len(already_approved)} already-approved detections")
            print(f"   Already approved indices: {already_approved}")
        
        if not indices_to_process:
            print(f"\n   ✓ All detections in this group were already auto-approved. Nothing to process.")
            return
        
        # Continue with only the indices that need processing
        group_original_indices = indices_to_process
        group = group.loc[indices_to_process]
        
        num_batches = (len(group_original_indices) + batch_size - 1) // batch_size
        
        print(f"\n   Creating {num_batches} batch(es) of up to {batch_size} items each...")
        print(f"   Processing indices: {group_original_indices}")
        
        # Verify group data
        print(f"\n   🔍 Group Verification:")
        print(f"   - Group shape: {group.shape}")
        print(f"   - Group index: {group.index.tolist()}")
        print(f"   - Unique Template_Types: {group['Template_Type'].unique().tolist()}")
        print(f"   - Unique Tag_Names: {group['Tag_Name'].unique().tolist()}")

        consecutive_all_false = 0  
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(group_original_indices))
            batch_original_indices = group_original_indices[batch_start:batch_end]
            
            # CRITICAL: Use group.loc to get the correct rows
            batch_df = group.loc[batch_original_indices]
            
            print(f"\n   {'='*70}")
            print(f"   BATCH {batch_num + 1}/{num_batches} ({len(batch_original_indices)} items)")
            print(f"   Original indices in this batch: {batch_original_indices}")
            print(f"   Verification type: {verification_type}")
            if verification_type == 'tag':
                print(f"   Tag name: {tag_name}")
            else:
                print(f"   Icon detection - will compare with template")
            print(f"   {'='*70}")
            
            # Debug: Print what we're processing
            print(f"\n      🔍 Batch Content Debug:")
            for idx, (orig_idx, row) in enumerate(batch_df.iterrows()):
                print(f"      Item {idx+1}: DF_Index={orig_idx}, Symbol_ID={row['Symbol_ID']}, "
                      f"Tag={row['Tag_Name']}, Type={row['Template_Type']}, Conf={row['Confidence']:.4f}")
            
            # Crop images
            print(f"\n      📸 Cropping {len(batch_original_indices)} detections with {int(padding*100)}% padding...")
            crop_paths = []
            metadata_list = []
            
            for idx, (orig_idx, row) in enumerate(batch_df.iterrows()):
                crop_path = os.path.join(self.temp_folder, f"crop_{verification_type}_{orig_idx}_{batch_num}.png")
                self.crop_detection(image_path, row, padding, crop_path)
                crop_paths.append(crop_path)
                
                metadata_list.append({
                    'confidence': f"{row['Confidence']:.4f}",
                    'symbol_id': row['Symbol_ID']
                })
                
                print(f"         ✓ Cropped item {idx+1}: Index={orig_idx}, Symbol_ID={row['Symbol_ID']}")
            
            print(f"      ✓ Cropping complete")
            
            # Create table
            if verification_type == 'tag':
                type_suffix = tag_name
            else:
                type_suffix = 'all_icons'
            
            table_name = f"{verification_type}_{type_suffix}_batch_{batch_num + 1}.png"
            table_path = os.path.join(self.table_folder, table_name)
            
            print(f"\n      📋 Creating verification table...")
            serial_numbers = list(range(1, len(batch_original_indices) + 1))
            
            self.create_verification_table(
                crop_paths, serial_numbers, table_path, template_path,
                original_indices=batch_original_indices,
                metadata=metadata_list
            )
            
            print(f"      ✓ Table saved to: {table_path}")
            
            # Display table
            print(f"\n      🖼️  Displaying table...")
            self.display_table(table_path)
            
            # Verify with LLM
            print(f"\n      🔍 Starting LLM verification for {len(batch_original_indices)} items...")

            results = self.verify_with_llm(table_path, verification_type, 
                                          tag_name, len(batch_original_indices))
                                          
            if all(r == False for r in results):
                consecutive_all_false += 1
                print(f"\n      ⚠️ LLM returned all False for the {consecutive_all_false} time(s) in a row")

                if consecutive_all_false >= 5:
                    print("\n      ❌ LLM consistently rejected 5 batches in a row.")
                    print("      🚨 Exiting remaining verification loops early.\n")
                    break  # EXIT the batch loop early
            else:
                consecutive_all_false = 0


            # Update DataFrame with original indices
            if len(results) == len(batch_original_indices):
                print(f"\n      💾 Updating DataFrame with results...")
                print(f"      Mapping: Serial -> DF_Index -> Result")
                
                for serial, orig_idx, result in zip(serial_numbers, batch_original_indices, results):
                    if orig_idx in df.index:
                        # CRITICAL: Only update if NOT already auto-approved
                        if df.at[orig_idx, 'Auto_Approved'] == True:
                            print(f"      🛡️  Serial {serial} -> Index {orig_idx}: SKIPPED (already auto-approved)")
                            continue
                        
                        # Store old value for comparison
                        old_value = df.at[orig_idx, 'LLM_Approved']
                        df.at[orig_idx, 'LLM_Approved'] = result
                        df.at[orig_idx, 'Auto_Approved'] = False  # Mark as NOT auto-approved
                        df.at[orig_idx, 'Verification_Method'] = 'LLM_Verified'
                        
                        status_symbol = "✓" if result else "✗"
                        print(f"      {status_symbol} Serial {serial} -> Index {orig_idx}: "
                              f"{old_value} -> {result} ({'APPROVED' if result else 'REJECTED'})")
                        
                        # Log for debugging
                        self.processing_log.append({
                            'index': orig_idx,
                            'action': 'llm_verified',
                            'result': result,
                            'batch': batch_num + 1,
                            'serial': serial,
                            'symbol_id': df.at[orig_idx, 'Symbol_ID'],
                            'tag_name': df.at[orig_idx, 'Tag_Name'],
                            'confidence': df.at[orig_idx, 'Confidence']
                        })
                    else:
                        print(f"      ⚠️  WARNING: Index {orig_idx} not found in df!")
                
                # Verification check
                approved_in_batch = sum(results)
                print(f"\n      ✓ Batch {batch_num + 1} complete!")
                print(f"      Approved: {approved_in_batch}/{len(results)}, "
                      f"Rejected: {len(results) - approved_in_batch}/{len(results)}")
            else:
                print(f"      ❌ ERROR: Could not update DataFrame - result count mismatch")
                print(f"      Expected {len(batch_original_indices)} results, got {len(results)}")
                # Mark as False (rejected) if verification failed
                for orig_idx in batch_original_indices:
                    if orig_idx in df.index and df.at[orig_idx, 'Auto_Approved'] != True:
                        df.at[orig_idx, 'LLM_Approved'] = False
                        df.at[orig_idx, 'Verification_Method'] = 'LLM_Verification_Failed'
                        print(f"      ✗ Index {orig_idx}: Marked as REJECTED (verification failed)")
    
    def _print_final_summary(self, df: pd.DataFrame):
        """Print comprehensive final summary with verification."""
        print(f"\n" + "="*80)
        print(f"STEP 3: FINAL RESULTS & VERIFICATION")
        print("="*80)
        
        # Count results
        approved = (df['LLM_Approved'] == True).sum()
        rejected = (df['LLM_Approved'] == False).sum()
        pending = df['LLM_Approved'].isna().sum()
        
        # Count by verification method
        auto_approved = (df['Auto_Approved'] == True).sum()
        llm_verified = ((df['Verification_Method'] == 'LLM_Verified') | 
                       (df['Verification_Method'] == 'LLM_Verification_Failed')).sum()
        
        print(f"\n📊 Verification Summary:")
        print(f"   Total detections: {len(df)}")
        print(f"   ✅ Approved: {approved} ({approved/len(df)*100:.1f}%)")
        print(f"      - Auto-approved (high confidence): {auto_approved}")
        print(f"      - LLM approved: {approved - auto_approved}")
        print(f"   ❌ Rejected: {rejected} ({rejected/len(df)*100:.1f}%)")
        if pending > 0:
            print(f"   ⏳ Pending/Error: {pending} ({pending/len(df)*100:.1f}%)")
        
        print(f"\n📋 Verification Method Breakdown:")
        for method in df['Verification_Method'].unique():
            count = (df['Verification_Method'] == method).sum()
            print(f"   {method}: {count}")
        
        # Sanity check
        total_processed = approved + rejected + pending
        if total_processed != len(df):
            print(f"\n   ⚠️  WARNING: Count mismatch!")
            print(f"   Sum of categories: {total_processed}")
            print(f"   Total rows: {len(df)}")
        else:
            print(f"\n   ✓ Sanity check PASSED: All {len(df)} detections accounted for")
        
        # CRITICAL: Show breakdown by Tag_Name to verify CF1/CF2
        print(f"\n📊 Results by Tag_Name:")
        for tag_name in df['Tag_Name'].unique():
            tag_df = df[df['Tag_Name'] == tag_name]
            tag_approved = (tag_df['LLM_Approved'] == True).sum()
            tag_rejected = (tag_df['LLM_Approved'] == False).sum()
            tag_auto = (tag_df['Auto_Approved'] == True).sum()
            
            print(f"\n   Tag: '{tag_name}' (Total: {len(tag_df)})")
            print(f"      ✅ Approved: {tag_approved} (Auto: {tag_auto}, LLM: {tag_approved - tag_auto})")
            print(f"      ❌ Rejected: {tag_rejected}")
            
            # Show specific indices
            if tag_auto > 0:
                auto_indices = tag_df[tag_df['Auto_Approved'] == True].index.tolist()
                print(f"      Auto-approved indices: {auto_indices}")
            
            llm_approved_indices = tag_df[(tag_df['LLM_Approved'] == True) & (tag_df['Auto_Approved'] == False)].index.tolist()
            if llm_approved_indices:
                print(f"      LLM-approved indices: {llm_approved_indices}")
            
            rejected_indices = tag_df[tag_df['LLM_Approved'] == False].index.tolist()
            if rejected_indices:
                print(f"      Rejected indices: {rejected_indices[:10]}{'...' if len(rejected_indices) > 10 else ''}")
        
        # Show sample of results with new columns
        print(f"\n📋 Sample Results:")
        
        if approved > 0:
            print(f"\n   ✅ Approved Detections (first 5):")
            approved_df = df[df['LLM_Approved'] == True][['Symbol_ID', 'Tag_Name', 'Template_Type', 
                                                           'Confidence', 'Auto_Approved', 'Verification_Method']].head(5)
            approved_with_idx = approved_df.copy()
            approved_with_idx['DF_Index'] = approved_df.index
            print(approved_with_idx.to_string(index=False))
        
        if rejected > 0:
            print(f"\n   ❌ Rejected Detections (first 5):")
            rejected_df = df[df['LLM_Approved'] == False][['Symbol_ID', 'Tag_Name', 'Template_Type', 
                                                            'Confidence', 'Auto_Approved', 'Verification_Method']].head(5)
            rejected_with_idx = rejected_df.copy()
            rejected_with_idx['DF_Index'] = rejected_df.index
            print(rejected_with_idx.to_string(index=False))
        
        if pending > 0:
            print(f"\n   ⏳ Pending/Error Detections:")
            pending_df = df[df['LLM_Approved'].isna()][['Symbol_ID', 'Tag_Name', 'Template_Type', 
                                                         'Confidence', 'Auto_Approved', 'Verification_Method']]
            pending_with_idx = pending_df.copy()
            pending_with_idx['DF_Index'] = pending_df.index
            print(pending_with_idx.to_string(index=False))
        
        # Processing log summary with more detail
        if self.processing_log:
            print(f"\n📝 Processing Log Summary:")
            auto_approved_logs = [log for log in self.processing_log if log['action'] == 'auto_approved']
            llm_verified_logs = [log for log in self.processing_log if log['action'] == 'llm_verified']
            llm_approved_logs = [log for log in llm_verified_logs if log.get('result')]
            llm_rejected_logs = [log for log in llm_verified_logs if not log.get('result')]
            
            print(f"   Auto-approved (high confidence): {len(auto_approved_logs)}")
            if auto_approved_logs:
                print(f"      Indices: {[log['index'] for log in auto_approved_logs[:10]]}"
                      f"{'...' if len(auto_approved_logs) > 10 else ''}")
            
            print(f"   LLM verified: {len(llm_verified_logs)}")
            print(f"   - LLM approved: {len(llm_approved_logs)}")
            if llm_approved_logs:
                print(f"      Indices: {[log['index'] for log in llm_approved_logs[:10]]}"
                      f"{'...' if len(llm_approved_logs) > 10 else ''}")
            
            print(f"   - LLM rejected: {len(llm_rejected_logs)}")
            if llm_rejected_logs:
                print(f"      Indices: {[log['index'] for log in llm_rejected_logs[:10]]}"
                      f"{'...' if len(llm_rejected_logs) > 10 else ''}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary crop files."""
        print(f"\n🧹 Cleaning up temporary files...")
        count = 0
        for file in os.listdir(self.temp_folder):
            os.remove(os.path.join(self.temp_folder, file))
            count += 1
        print(f"   ✓ Removed {count} temporary crop files from {self.temp_folder}")
        
        # Note: Keep approved_folder and table_folder for reference