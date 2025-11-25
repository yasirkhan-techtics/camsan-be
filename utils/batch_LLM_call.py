"""
Image Comparison Module using Google Gemini API with Pydantic validation.
"""

import os
import time
from typing import List, Optional
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from PIL import Image
from pydantic import BaseModel, Field, field_validator


class ComparisonResult(BaseModel):
    """Single comparison result with serial number and match status."""
    serial_number: str = Field(..., description="Serial number or identifier")
    matches: bool = Field(..., description="Whether the shape matches the reference")
    
    @field_validator('serial_number')
    @classmethod
    def validate_serial_number(cls, v: str) -> str:
        """Ensure serial number is not empty."""
        if not v or not v.strip():
            raise ValueError("Serial number cannot be empty")
        return v.strip()


class BatchComparisonResults(BaseModel):
    """Collection of comparison results from a single API call."""
    results: List[ComparisonResult] = Field(
        ..., 
        description="List of comparison results for each table/symbol in the image"
    )


class ImageComparator:
    """
    Compare electrical symbols in images against a reference image.
    
    Uses Google Gemini API to perform shape-invariant comparison.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 5,
        retry_delay: int = 5
    ):
        """
        Initialize the Image Comparator.
        
        Args:
            api_key: Google API key (if None, uses environment variable)
            model: Gemini model to use
            max_retries: Maximum number of retry attempts on server error
            retry_delay: Delay in seconds between retries
        """
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _upload_reference_image(self, reference_path: str) -> any:
        """Upload reference image to Gemini."""
        return self.client.files.upload(file=reference_path)
    
    def _compare_single_image(
        self,
        uploaded_reference: any,
        compare_path: str,
        filename: str
    ) -> BatchComparisonResults:
        """
        Compare a single image containing multiple tables/symbols against the reference.
        
        Args:
            uploaded_reference: Uploaded reference file object
            compare_path: Path to comparison image
            filename: Name of the comparison file
            
        Returns:
            BatchComparisonResults with multiple comparison results
        """
        with open(compare_path, 'rb') as f:
            img_bytes = f.read()
        
        # Construct prompt for structured output
        prompt = """Compare EACH electrical symbol/table in the image with the reference electrical symbol.
Ignore rotation, scaling, thickness, and boldness—only the exact geometric shape must match.

For each symbol/table found in the image, return a comparison result with:
- serial_number: the identifier or serial number visible in the image
- matches: true if the shape matches the reference, false otherwise

Return your response in this exact JSON format:
{
    "results": [
        {"serial_number": "1", "matches": true},
        {"serial_number": "2", "matches": false},
        {"serial_number": "3", "matches": true}
    ]
}

Analyze ALL symbols/tables in the image and return a result for each one.
Only respond with the JSON, no additional text."""
        
        # Retry logic for server errors
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        prompt,
                        uploaded_reference,
                        types.Part.from_bytes(data=img_bytes, mime_type='image/png')
                    ]
                )
                
                # Parse response with Pydantic
                result_text = response.text.strip()
                print(f"  Raw LLM response: {result_text}")  # Debug output
                
                # Try to parse JSON, handling potential markdown code blocks
                import json
                import re
                
                # Remove markdown code blocks if present
                clean_text = re.sub(r'```json\s*|\s*```', '', result_text)
                clean_text = clean_text.strip()
                
                # Parse JSON
                json_data = json.loads(clean_text)
                
                # Validate with Pydantic
                batch_results = BatchComparisonResults.model_validate(json_data)
                
                return batch_results
                
            except ServerError as e:
                print(f"Server busy for {filename}, retry {attempt+1}/{self.max_retries} in {self.retry_delay}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return BatchComparisonResults(results=[])
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()  # Debug output
                return BatchComparisonResults(results=[])
        
        # Fallback if all retries failed
        return BatchComparisonResults(results=[])
    
    def compare_folder(
        self,
        reference_image_path: str,
        compare_folder: str,
        image_extensions: tuple = ('.png', '.jpg', '.jpeg'),
        verbose: bool = True
    ) -> dict:
        """
        Compare all images in a folder against a reference image.
        
        Args:
            reference_image_path: Path to reference electrical symbol image
            compare_folder: Folder containing images to compare
            image_extensions: Tuple of valid image extensions
            verbose: Print progress information
            
        Returns:
            Dictionary mapping filenames to their BatchComparisonResults
        """
        # Validate paths
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        
        if not os.path.exists(compare_folder):
            raise FileNotFoundError(f"Compare folder not found: {compare_folder}")
        
        # Upload reference image
        if verbose:
            print(f"Uploading reference image: {reference_image_path}")
        
        uploaded_reference = self._upload_reference_image(reference_image_path)
        
        # Process all images
        all_results = {}
        image_files = [
            f for f in os.listdir(compare_folder)
            if f.lower().endswith(image_extensions)
        ]
        
        if verbose:
            print(f"Found {len(image_files)} images to compare\n")
        
        for idx, filename in enumerate(image_files, 1):
            if verbose:
                print(f"[{idx}/{len(image_files)}] Processing: {filename}")
            
            compare_path = os.path.join(compare_folder, filename)
            batch_results = self._compare_single_image(uploaded_reference, compare_path, filename)
            all_results[filename] = batch_results
            
            if verbose:
                print(f"  → Found {len(batch_results.results)} symbols/tables")
                for result in batch_results.results:
                    print(f"     • Serial {result.serial_number}: {'✓ MATCH' if result.matches else '✗ NO MATCH'}")
                print()
        
        return all_results
    
    def display_comparison(
        self,
        reference_image_path: str,
        compare_image_path: str
    ) -> Image.Image:
        """
        Create a side-by-side comparison image.
        
        Args:
            reference_image_path: Path to reference image
            compare_image_path: Path to comparison image
            
        Returns:
            PIL Image with both images side by side
        """
        ref_img = Image.open(reference_image_path)
        comp_img = Image.open(compare_image_path)
        
        # Create combined image
        max_height = max(ref_img.height, comp_img.height)
        combined = Image.new('RGB', (ref_img.width + comp_img.width, max_height), color='white')
        combined.paste(ref_img, (0, 0))
        combined.paste(comp_img, (ref_img.width, 0))
        
        return combined