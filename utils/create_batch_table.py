from PIL import Image, ImageDraw, ImageFont
import os

def create_image_table(image_folder, output_folder="tables"):
    """
    Creates table images with 5 images per table from a folder.
    
    Args:
        image_folder: Path to folder containing images
        output_folder: Path to save table images (default: "tables")
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(valid_extensions)]
    image_files.sort()
    
    if not image_files:
        print("No images found in the folder!")
        return
    
    # Process images in batches of 10
    batch_num = 1
    for i in range(0, len(image_files), 10):
        batch = image_files[i:i+10]
        create_single_table(image_folder, batch, output_folder, batch_num)
        batch_num += 1
    
    print(f"Created {batch_num-1} table(s) in '{output_folder}' folder")


def create_single_table(image_folder, image_files, output_folder, batch_num):
    """Creates a single table image for a batch of images."""
    
    # Settings
    border_width = 2
    padding = 20
    cell_padding = 15
    font_size = 24
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        font_bold = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        font = ImageFont.load_default()
        font_bold = font
    
    # Load and resize images
    images = []
    max_img_height = 0
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path)
        
        # Resize if too large (max width 400px)
        if img.width > 400:
            ratio = 400 / img.width
            new_size = (400, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        images.append(img)
        max_img_height = max(max_img_height, img.height)
    
    # Calculate dimensions
    # Column widths
    col1_width = 120  # Sr. No
    col2_width = max(img.width for img in images) + 2 * cell_padding
    col3_width = 200  # Same or not
    
    total_width = col1_width + col2_width + col3_width + (4 * border_width)
    
    # Row height based on image height + padding
    row_height = max_img_height + int(max_img_height * 0.2) + 2 * cell_padding
    header_height = 60
    
    total_height = header_height + (len(images) * row_height) + ((len(images) + 1) * border_width)
    
    # Create white canvas
    table_img = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(table_img)
    
    # Draw header row
    y = 0
    
    # Header background
    draw.rectangle([0, 0, total_width, header_height], fill='#E8E8E8')
    
    # Header borders
    draw.line([(0, 0), (total_width, 0)], fill='black', width=border_width)
    draw.line([(0, header_height), (total_width, header_height)], fill='black', width=border_width)
    
    # Vertical borders for header
    x = 0
    draw.line([(x, 0), (x, header_height)], fill='black', width=border_width)
    x += col1_width + border_width
    draw.line([(x, 0), (x, header_height)], fill='black', width=border_width)
    x += col2_width + border_width
    draw.line([(x, 0), (x, header_height)], fill='black', width=border_width)
    x += col3_width + border_width
    draw.line([(x, 0), (x, header_height)], fill='black', width=border_width)
    
    # Header text
    draw.text((border_width + col1_width//2, header_height//2), "Sr. No", 
              fill='black', font=font_bold, anchor='mm')
    draw.text((border_width + col1_width + border_width + col2_width//2, header_height//2), 
              "Image", fill='black', font=font_bold, anchor='mm')
    draw.text((border_width + col1_width + border_width + col2_width + border_width + col3_width//2, 
              header_height//2), "Same or Not", fill='black', font=font_bold, anchor='mm')
    
    # Draw data rows
    y = header_height + border_width
    
    for idx, img in enumerate(images, 1):
        # Row borders
        # Top border (already drawn as bottom of previous row or header)
        
        # Left vertical border
        draw.line([(0, y), (0, y + row_height)], fill='black', width=border_width)
        
        # Column separators
        x = col1_width + border_width
        draw.line([(x, y), (x, y + row_height)], fill='black', width=border_width)
        
        x += col2_width + border_width
        draw.line([(x, y), (x, y + row_height)], fill='black', width=border_width)
        
        # Right border
        x += col3_width
        draw.line([(x, y), (x, y + row_height)], fill='black', width=border_width)
        
        # Bottom border
        draw.line([(0, y + row_height), (total_width, y + row_height)], 
                  fill='black', width=border_width)
        
        # Sr. No text
        draw.text((border_width + col1_width//2, y + row_height//2), 
                  str(idx), fill='black', font=font, anchor='mm')
        
        # Paste image
        img_x = border_width + col1_width + border_width + cell_padding
        img_y = y + (row_height - img.height) // 2
        table_img.paste(img, (img_x, img_y))
        
        # Move to next row
        y += row_height + border_width
    
    # Save table
    output_path = os.path.join(output_folder, f"table_{batch_num}.png")
    table_img.save(output_path)
    print(f"Created: {output_path}")


# Example usage
if __name__ == "__main__":
    # Replace 'images' with your folder path
    create_image_table("images")