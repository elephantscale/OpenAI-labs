# Add your utilities or helper functions to this file.
from pathlib import Path
from IPython.display import display, Image as DisplayImage, IFrame
from PIL import Image as PILImage, ImageDraw
import pymupdf
from typing import Union, Dict, List, Any
from functools import lru_cache
import fitz
import io

def print_document(file_path: str):
    """
    Displays a PDF or image file in the notebook.

    Args:
        file_path: The path to the document file.
    """
    path = Path(file_path)
    if path.exists():
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            display(DisplayImage(filename=str(path)))
        elif path.suffix.lower() == '.pdf':
            display(IFrame(src=str(path), width=800, height=600))
        else:
            print(f"Unsupported file type: {path.suffix}")
    else:
        print(f"File not found: {file_path}")

## Based on the code sample provided at https://docs.landing.ai/ade/ade-python#visualize-parsed-chunks:-draw-bounding-boxes

# Define colors for each chunk type
CHUNK_TYPE_COLORS = {
    "chunkText": (40, 167, 69),        # Green
    "chunkTable": (0, 123, 255),       # Blue
    "chunkMarginalia": (111, 66, 193), # Purple
    "chunkFigure": (255, 0, 255),      # Magenta
    "chunkLogo": (144, 238, 144),      # Light green
    "chunkCard": (255, 165, 0),        # Orange
    "chunkAttestation": (0, 255, 255), # Cyan
    "chunkScanCode": (255, 193, 7),    # Yellow
    "chunkForm": (220, 20, 60),        # Red
    "tableCell": (173, 216, 230),      # Light blue
    "table": (70, 130, 180),           # Steel blue
}

def draw_bounding_boxes_2(groundings, document_path, base_path="."):
    """
    Draw bounding boxes on document images to visualize parsed chunks.

    Args:
        groundings: Dictionary of grounding objects with chunk locations
        document_path: Path to the original document
        base_path: Directory to save annotated images
    """
    def create_annotated_image(image, groundings, page_num=0):
        """Create an annotated image with grounding boxes and labels."""
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)

        img_width, img_height = image.size

        groundings_found = 0
        for gid, grounding in groundings.items():
            # Check if grounding belongs to this page (for multi-page PDFs)
            if grounding.page != page_num:
                continue

            groundings_found += 1
            box = grounding.box

            # Extract normalized coordinates from box
            left, top, right, bottom = box.left, box.top, box.right, box.bottom

            # Convert normalized coordinates to pixel coordinates
            x1 = int(left * img_width)
            y1 = int(top * img_height)
            x2 = int(right * img_width)
            y2 = int(bottom * img_height)

            # Draw bounding box with color based on chunk type
            color = CHUNK_TYPE_COLORS.get(grounding.type, (128, 128, 128))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background and text
            label = f"{grounding.type}:{gid}"
            label_y = max(0, y1 - 20)
            draw.rectangle([x1, label_y, x1 + len(label) * 8, y1], fill=color)
            draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))

        if groundings_found == 0:
            return None
        return annotated_img

    # Handle PDF documents
    if document_path.suffix.lower() == '.pdf':
        pdf = pymupdf.open(document_path)
        total_pages = len(pdf)

        for page_num in range(total_pages):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scaling
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Create and save annotated image for this page
            annotated_img = create_annotated_image(img, groundings, page_num)
            if annotated_img is not None:
                annotated_path = f"{base_path}/page_{page_num + 1}_annotated.png"
                annotated_img.save(annotated_path)
                print(f"Annotated image saved to: {annotated_path}")

        pdf.close()
    else:
        # Handle image files directly
        img = PILImage.open(document_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Create and save annotated image
        annotated_img = create_annotated_image(img, groundings)
        annotated_path = "page_annotated.png"
        annotated_img.save(annotated_path)
        print(f"Annotated image saved to: {annotated_path}")

    return img


def draw_bounding_boxes(parse_response, document_path):
    """
    Draw bounding boxes for all grounded chunks on the given document.

    Args:
        parse_response: ADE ParseResponse containing grounding info.
        document_path: Path to the original PDF or image.

    Returns:
        The last annotated PIL Image object (for single-page docs,
        this is the only page).
    """
    
    document_path = Path(document_path)
    
    def create_annotated_image(image, groundings, page_num=0):
        """Create an annotated image with grounding boxes and labels."""
        
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)

        img_width, img_height = image.size

        for gid, grounding in groundings.items():
            # Check if grounding belongs to this page (for PDFs)
            if grounding.page != page_num:
                continue

            box = grounding.box

            # Extract coordinates from box
            left, top, right, bottom = box.left, box.top, box.right, box.bottom

            # Convert to pixel coordinates
            x1 = int(left * img_width)
            y1 = int(top * img_height)
            x2 = int(right * img_width)
            y2 = int(bottom * img_height)

            # Draw bounding box
            color = CHUNK_TYPE_COLORS.get(grounding.type, (128, 128, 128))  # Default to gray
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background and text
            label = f"{grounding.type}:{gid}"
            label_y = max(0, y1 - 20)
            draw.rectangle([x1, label_y, x1 + len(label) * 8, y1], fill=color)
            draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))
        
        return annotated_img

    if document_path.suffix.lower() == '.pdf':
        pdf = pymupdf.open(document_path)
        total_pages = len(pdf)
        base_name = document_path.stem

        for page_num in range(total_pages):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scaling
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Create and save annotated image
            annotated_img = create_annotated_image(img, parse_response.grounding, page_num)
            annotated_path = f"page_{page_num + 1}_annotated.png"
            # annotated_img.save(annotated_path) #uncomment if you want to save the output
            # print(f"Annotated image saved to: {annotated_path}") #uncomment if you want to save the output

        pdf.close()
    else:
        # Load image file directly
        img = PILImage.open(document_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Create and save annotated image
        annotated_img = create_annotated_image(img, parse_response.grounding)
        annotated_path = "page_annotated.png"
        # annotated_img.save(annotated_path) #uncomment if you want to save the output
        # print(f"Annotated image saved to: {annotated_path}")
        
    display(annotated_img)
    #return annotated_img


## Based on the code sample provided at https://github.com/landing-ai/ade-helper-scripts/blob/main/Events/Deeplearning_Event_FY25/ade_utils.py

def create_cropped_chunk_images(parse_result, extraction_metadata, document_path, first_page, doc_name):
    """
    Create cropped images of individual chunks and full page with chunks outlined.

    Returns dict mapping field_name -> {"crop": Image, "outlined": Image}
    """
    pdf = pymupdf.open(document_path)
    page = pdf[first_page]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    full_page_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf.close()

    img_width, img_height = full_page_img.size
    field_images = {}

    for field_name, metadata in extraction_metadata.items():
        # Get the first chunk reference
        chunk_id = metadata['references'][0]

        if chunk_id not in parse_result.grounding:
            continue

        grounding = parse_result.grounding[chunk_id]

        # Only process if it's on the first page
        if grounding.page != first_page:
            continue

        box = grounding.box
        left, top, right, bottom = box.left, box.top, box.right, box.bottom

        # Convert normalized coordinates to pixels
        x1 = int(left * img_width)
        y1 = int(top * img_height)
        x2 = int(right * img_width)
        y2 = int(bottom * img_height)

        # Add padding for better visibility
        padding = 10
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(img_width, x2 + padding)
        y2_crop = min(img_height, y2 + padding)

        # Create cropped image
        cropped = full_page_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))

        # Create outlined version (full page with just this chunk highlighted)
        outlined = full_page_img.copy()
        draw = ImageDraw.Draw(outlined)
        color = (231, 76, 60)  # Red
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        # Add label
        label = field_name
        label_y = max(0, y1 - 25)
        draw.rectangle([x1, label_y, x1 + len(label) * 10, y1], fill=color)
        draw.text((x1 + 5, label_y + 5), label, fill=(255, 255, 255))

        field_images[field_name] = {
            "crop": cropped,
            "outlined": outlined
        }

    return field_images

@lru_cache(maxsize=20)
def get_pdf_page_cached(pdf_path_str, page_num, dpi=150):
    """
    Cache PDF pages for faster repeated access.
    """
    doc = fitz.open(pdf_path_str)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    page_width, page_height = page.rect.width, page.rect.height
    doc.close()
    return img, page_width, page_height


def extract_chunk_image(pdf_path, page_num, bbox=None, highlight=True, padding=10):
    """
    Dynamically extract and crop a specific chunk from PDF.
    
    Args:
        pdf_path: Path to PDF
        page_num: Page number (0-indexed)
        bbox: [x0, y0, x1, y1] in NORMALIZED coordinates (0-1 range) or 
        None for full page
        highlight: Add red border around chunk
        padding: Extra pixels around bbox (default 10)
    
    Returns:
        PNG image bytes or None
    """
    
    # Get cached page
    img, page_width, page_height = get_pdf_page_cached(str(pdf_path), page_num)
    
    if img is None:
        return None
    
    # If no bbox, return full page
    if not bbox or len(bbox) != 4:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    # Extract normalized bbox coordinates (0-1 range)
    norm_x0, norm_y0, norm_x1, norm_y1 = bbox
    
    # Convert normalized coordinates to PDF points
    pdf_x0 = norm_x0 * page_width
    pdf_y0 = norm_y0 * page_height
    pdf_x1 = norm_x1 * page_width
    pdf_y1 = norm_y1 * page_height
    
    # Scale PDF points to image pixels
    scale_x = img.width / page_width
    scale_y = img.height / page_height
    
    # Apply scaling and padding
    crop_x0 = max(0, int(pdf_x0 * scale_x) - padding)
    crop_y0 = max(0, int(pdf_y0 * scale_y) - padding)
    crop_x1 = min(img.width, int(pdf_x1 * scale_x) + padding)
    crop_y1 = min(img.height, int(pdf_y1 * scale_y) + padding)
    
    # Crop to chunk region
    chunk_img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    
    # Add red border highlight
    if highlight:
        draw = ImageDraw.Draw(chunk_img)
        draw.rectangle(
            [padding, padding, 
             chunk_img.width - padding - 1, 
             chunk_img.height - padding - 1],
            outline="red",
            width=3
        )
    
    # Convert to PNG bytes
    img_bytes = io.BytesIO()
    chunk_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()