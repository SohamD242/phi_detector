"""
PDF Redaction Module for CADIS PHI Detector

This module handles direct PDF redaction by:
1. Extracting text with character-to-coordinate mapping
2. Mapping PHI entity offsets to PDF rectangles
3. Applying permanent redactions using PyMuPDF's redaction API
"""

import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import sys
import os

# Import configuration
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
except ImportError:
    class config:
        VERBOSE = False


def extract_text_with_positions(pdf_path: str) -> Dict:
    """
    Extract text from PDF with character-level position mapping.
    
    This function builds a mapping between character offsets in the extracted text
    and their physical coordinates in the PDF, enabling accurate redaction placement.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing:
            - 'text': Full concatenated text from all pages
            - 'char_map': List of dicts with character position info
            - 'pages': Total number of pages
            
    Example char_map entry:
        {
            'char': 'J',
            'offset': 45,
            'page': 0,
            'bbox': (123.4, 456.7, 134.2, 470.3)  # (x0, y0, x1, y1)
        }
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    char_map = []
    char_offset = 0
    
    verbose = getattr(config, 'VERBOSE', False)
    
    if verbose:
        print(f"Extracting text with positions from {pdf_path}")
        print(f"Total pages: {len(doc)}")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        
        if verbose:
            print(f"  Processing page {page_num + 1}...")
        
        for block in page_dict["blocks"]:
            if block["type"] != 0:  # Skip non-text blocks (images, etc.)
                continue
            
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    bbox = span["bbox"]  # (x0, y0, x1, y1)
                    
                    if not text:
                        continue
                    
                    # Calculate approximate character width within span
                    span_width = bbox[2] - bbox[0]
                    char_width = span_width / len(text) if len(text) > 0 else 0
                    
                    # Map each character to its approximate position
                    for i, char in enumerate(text):
                        char_x0 = bbox[0] + (i * char_width)
                        char_x1 = char_x0 + char_width
                        
                        char_map.append({
                            'char': char,
                            'offset': char_offset,
                            'page': page_num,
                            'bbox': (char_x0, bbox[1], char_x1, bbox[3])
                        })
                        
                        char_offset += 1
                    
                    full_text += text
                
                # Add newline after each line
                full_text += "\n"
                char_map.append({
                    'char': '\n',
                    'offset': char_offset,
                    'page': page_num,
                    'bbox': (0, 0, 0, 0)  # Newlines have no visual position
                })
                char_offset += 1
    
    # Store page count before closing document
    total_pages = len(doc)
    doc.close()
    
    if verbose:
        print(f"Extracted {len(full_text)} characters with position mapping")
    
    return {
        'text': full_text,
        'char_map': char_map,
        'pages': total_pages
    }


def merge_char_bboxes(char_list: List[Dict]) -> fitz.Rect:
    """
    Merge character bounding boxes into a single rectangle.
    
    Args:
        char_list: List of character info dicts with 'bbox' keys
        
    Returns:
        fitz.Rect covering all characters with padding
    """
    if not char_list:
        return fitz.Rect(0, 0, 0, 0)
    
    # Get min/max coordinates
    x0 = min(c['bbox'][0] for c in char_list)
    y0 = min(c['bbox'][1] for c in char_list)
    x1 = max(c['bbox'][2] for c in char_list)
    y1 = max(c['bbox'][3] for c in char_list)
    
    # Add small padding to ensure complete coverage (±2 points)
    padding = 2
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = x1 + padding
    y1 = y1 + padding
    
    return fitz.Rect(x0, y0, x1, y1)


def map_entity_to_coordinates(entity: Dict, char_map: List[Dict]) -> List[Dict]:
    """
    Map PHI entity character offsets to PDF coordinates.
    
    Handles multi-line and multi-page entities by detecting line breaks
    (y-coordinate changes) and page transitions.
    
    Args:
        entity: Dict with 'start', 'end', 'label', 'text' keys
        char_map: Character position map from extract_text_with_positions()
        
    Returns:
        List of redaction rectangles, one per line/page:
            [
                {
                    'page': 0,
                    'rect': fitz.Rect(x0, y0, x1, y1),
                    'label': 'NAME'
                },
                ...
            ]
    """
    start_idx = entity['start']
    end_idx = entity['end']
    
    verbose = getattr(config, 'VERBOSE', False)
    
    # Validate indices
    if start_idx < 0 or end_idx > len(char_map) or start_idx >= end_idx:
        if verbose:
            print(f"  Warning: Invalid entity offsets {start_idx}-{end_idx} (text: '{entity.get('text', '')}')")
        return []
    
    # Get character positions for this entity
    entity_chars = char_map[start_idx:end_idx]
    
    if not entity_chars:
        return []
    
    # Filter out newlines and invalid positions
    entity_chars = [c for c in entity_chars if c['char'] != '\n' and c['bbox'] != (0, 0, 0, 0)]
    
    if not entity_chars:
        return []
    
    redactions = []
    current_page = entity_chars[0]['page']
    current_line_chars = []
    prev_y = entity_chars[0]['bbox'][1]
    
    # Group characters by page and line
    for char_info in entity_chars:
        # Check if we're on a new page or new line (y-coordinate change > 5 pixels)
        if char_info['page'] != current_page or abs(char_info['bbox'][1] - prev_y) > 5:
            # Flush current line
            if current_line_chars:
                rect = merge_char_bboxes(current_line_chars)
                redactions.append({
                    'page': current_page,
                    'rect': rect,
                    'label': entity['label']
                })
                
                if verbose:
                    print(f"    Mapped '{entity.get('text', '')}' [{entity['label']}] to page {current_page}: {rect}")
            
            # Start new line
            current_page = char_info['page']
            current_line_chars = [char_info]
            prev_y = char_info['bbox'][1]
        else:
            current_line_chars.append(char_info)
    
    # Flush last line
    if current_line_chars:
        rect = merge_char_bboxes(current_line_chars)
        redactions.append({
            'page': current_page,
            'rect': rect,
            'label': entity['label']
        })
        
        if verbose:
            print(f"    Mapped '{entity.get('text', '')}' [{entity['label']}] to page {current_page}: {rect}")
    
    return redactions


def redact_pdf(input_pdf_path: str, output_pdf_path: str, 
               entities: List[Dict], char_map: List[Dict]) -> None:
    """
    Create redacted copy of PDF with PHI permanently blacked out.
    
    Uses PyMuPDF's official redaction API (add_redact_annot + apply_redactions)
    to permanently remove PHI text from the PDF structure, ensuring HIPAA compliance.
    
    Args:
        input_pdf_path: Path to original PDF
        output_pdf_path: Path for redacted PDF output
        entities: List of PHI entities with 'start', 'end', 'label', 'text' keys
        char_map: Character position map from extract_text_with_positions()
    """
    verbose = getattr(config, 'VERBOSE', False)
    redact_headers_footers = getattr(config, 'REDACT_HEADERS_FOOTERS', True)
    header_height = getattr(config, 'HEADER_REDACTION_HEIGHT', 72)
    footer_height = getattr(config, 'FOOTER_REDACTION_HEIGHT', 72)
    
    if verbose:
        print(f"\nApplying redactions to PDF: {input_pdf_path}")
        print(f"Total entities to redact: {len(entities)}")
        if redact_headers_footers:
            print(f"Header/footer redaction: top {header_height}pt, bottom {footer_height}pt")
    
    doc = fitz.open(input_pdf_path)
    
    # STEP 1: Redact headers and footers on ALL pages (prevents PHI leakage)
    if redact_headers_footers:
        if verbose:
            print(f"\n[Step 1/2] Redacting headers and footers on all {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Define header rectangle (top portion of page)
            header_rect = fitz.Rect(0, 0, page_width, header_height)
            
            # Define footer rectangle (bottom portion of page)
            footer_rect = fitz.Rect(0, page_height - footer_height, page_width, page_height)
            
            # Add redaction annotations (solid black boxes, no text)
            page.add_redact_annot(header_rect, fill=(0, 0, 0))
            page.add_redact_annot(footer_rect, fill=(0, 0, 0))
            
            # Apply header/footer redactions immediately
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=True,
                text=True
            )
            
            if verbose and page_num == 0:
                print(f"  Example (page 1): Header area={header_rect}, Footer area={footer_rect}")
        
        if verbose:
            print(f"  ✓ Headers and footers redacted on all pages")
            print(f"\n[Step 2/2] Applying entity-specific redactions...")
    
    # STEP 2: Group entity-based redactions by page for efficiency
    page_redactions = {}
    
    for entity in entities:
        redaction_rects = map_entity_to_coordinates(entity, char_map)
        
        for redaction in redaction_rects:
            page_num = redaction['page']
            if page_num not in page_redactions:
                page_redactions[page_num] = []
            page_redactions[page_num].append(redaction)
    
    if verbose:
        if redact_headers_footers:
            print(f"Entity redactions distributed across {len(page_redactions)} pages")
        else:
            print(f"Redactions distributed across {len(page_redactions)} pages")
    
    # Apply entity-specific redactions page by page
    for page_num, redactions in page_redactions.items():
        page = doc[page_num]
        
        if verbose:
            print(f"  Page {page_num + 1}: {len(redactions)} redactions")
        
        for redaction in redactions:
            rect = redaction['rect']
            label = redaction['label']
            
            # Add redaction annotation with replacement text
            page.add_redact_annot(
                rect,
                text=f"[{label}]",         # Replacement text shown after redaction
                fill=(0, 0, 0),            # Black background
                text_color=(1, 1, 1),      # White text
                fontsize=9,                # Smaller font to fit in original space
                align=fitz.TEXT_ALIGN_LEFT
            )
        
        # Apply all redactions on this page (permanent removal)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,  # Don't redact images
            graphics=True,                       # Remove vector graphics in redacted areas
            text=True                            # Remove text (main goal)
        )
    
    # Save redacted PDF with optimization and content stream cleanup
    doc.save(
        output_pdf_path, 
        garbage=4,      # Maximum garbage collection
        deflate=True,   # Compress content streams
        clean=True,     # CRITICAL: Rebuild content streams to remove original text
        pretty=False,   # Don't prettify (not needed)
        linear=False,   # Don't linearize (not needed for security)
        no_new_id=False # Allow new document ID
    )
    doc.close()
    
    # Verify redaction by checking if PHI text is still extractable
    if verbose:
        print(f"  Verifying redaction...")
        verify_doc = fitz.open(output_pdf_path)
        verify_text = ""
        for page in verify_doc:
            verify_text += page.get_text()
        verify_doc.close()
        
        # Check if any entity text still appears in output
        found_phi = []
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            if entity_text and len(entity_text) > 3 and entity_text in verify_text:
                found_phi.append(entity_text)
        
        if found_phi:
            print(f"  ⚠ WARNING: {len(found_phi)} PHI terms still present in output: {found_phi[:3]}")
        else:
            print(f"  ✓ Verification passed: PHI text removed from PDF")
    
    if verbose:
        print(f"✓ Redacted PDF saved to: {output_pdf_path}")


def redact_pdf_search_based(input_pdf_path: str, output_pdf_path: str, 
                            entities: List[Dict]) -> None:
    """
    Alternative simpler redaction approach using text search.
    
    This is a fallback method that searches for entity text strings directly
    in the PDF. Simpler but less accurate if text appears multiple times.
    
    Args:
        input_pdf_path: Path to original PDF
        output_pdf_path: Path for redacted PDF output
        entities: List of PHI entities with 'text' and 'label' keys
    """
    verbose = getattr(config, 'VERBOSE', False)
    
    if verbose:
        print(f"\nApplying search-based redactions to PDF: {input_pdf_path}")
    
    doc = fitz.open(input_pdf_path)
    redaction_count = 0
    
    for page_num, page in enumerate(doc):
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            if not entity_text:
                continue
            
            # Search for exact text match (case-sensitive)
            rects = page.search_for(entity_text, quads=False)
            
            for rect in rects:
                page.add_redact_annot(
                    rect,
                    text=f"[{entity['label']}]",
                    fill=(0, 0, 0),
                    text_color=(1, 1, 1),
                    fontsize=9
                )
                redaction_count += 1
        
        # Apply redactions for this page
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=True,
            text=True
        )
    
    # Save with clean=True to ensure text removal
    doc.save(
        output_pdf_path, 
        garbage=4, 
        deflate=True,
        clean=True,
        pretty=False,
        linear=False,
        no_new_id=False
    )
    doc.close()
    
    if verbose:
        print(f"✓ Applied {redaction_count} redactions using search-based method")
        print(f"✓ Redacted PDF saved to: {output_pdf_path}")


def convert_pdf_to_images(input_pdf_path: str, output_pdf_path: str, dpi: int = 300) -> None:
    verbose = getattr(config, 'VERBOSE', False)
    
    if verbose:
        print(f"\nConverting PDF to image-based format (DPI: {dpi})...")
        print(f"  Input: {input_pdf_path}")
    
    # Open source document
    src_doc = fitz.open(input_pdf_path)
    
    # Create new empty document
    img_doc = fitz.open()
    
    # Process each page
    for page_num in range(len(src_doc)):
        if verbose:
            print(f"  Converting page {page_num + 1}/{len(src_doc)} to image...")
        
        # Get page from source
        page = src_doc[page_num]
        
        # Render page to high-resolution image (pixmap)
        # zoom factor: 300 DPI = 4.17x zoom, 150 DPI = 2.08x zoom
        zoom = dpi / 72.0  # 72 DPI is the default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Get original page dimensions
        rect = page.rect
        
        # Create new page with same dimensions as original
        new_page = img_doc.new_page(width=rect.width, height=rect.height)
        
        # Insert the rendered image to fill the entire page
        new_page.insert_image(rect, pixmap=pix)
        
        # Clean up pixmap
        pix = None
    
    # Close source document
    src_doc.close()
    
    # Save image-based PDF with compression
    img_doc.save(
        output_pdf_path,
        garbage=4,      # Maximum garbage collection
        deflate=True,   # Compress streams
        clean=True      # Remove unused objects
    )
    img_doc.close()
    
    if verbose:
        print(f"  ✓ Image-based PDF saved to: {output_pdf_path}")
        print(f"  ✓ Text extraction is now impossible")


def secure_redact_pdf(input_pdf_path: str, output_pdf_path: str,
                      entities: List[Dict], char_map: List[Dict],
                      convert_to_images: bool = True, image_dpi: int = 300) -> None:
    """
    Perform secure redaction with optional image conversion.
    
    This function combines standard redaction with image conversion for maximum security.
    First applies redactions using PyMuPDF's API, then optionally converts the entire
    PDF to images to ensure zero text extraction possibility.
    
    Args:
        input_pdf_path: Path to original PDF
        output_pdf_path: Path for final secure PDF output
        entities: List of PHI entities to redact
        char_map: Character position mapping
        convert_to_images: If True, convert to image-based PDF (recommended)
        image_dpi: Resolution for image conversion (300 = print quality)
    """
    verbose = getattr(config, 'VERBOSE', False)
    
    if convert_to_images:
        # Two-stage process: redact first, then convert to images
        import tempfile
        import os as os_module
        
        # Create temporary file for intermediate redacted PDF
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os_module.close(temp_fd)
        
        try:
            # Stage 1: Apply redactions
            if verbose:
                print(f"\n[Stage 1/2] Applying redactions...")
            redact_pdf(input_pdf_path, temp_path, entities, char_map)
            
            # Stage 2: Convert to images
            if verbose:
                print(f"\n[Stage 2/2] Converting to image-based PDF...")
            convert_pdf_to_images(temp_path, output_pdf_path, dpi=image_dpi)
            
            if verbose:
                print(f"\n✓ Secure redaction complete (2-stage process)")
                print(f"  → Redactions applied: {len(entities)} entities")
                print(f"  → PDF converted to images at {image_dpi} DPI")
                print(f"  → Text extraction: IMPOSSIBLE")
        
        finally:
            # Clean up temporary file
            if os_module.path.exists(temp_path):
                os_module.remove(temp_path)
    else:
        # Standard redaction only (less secure)
        if verbose:
            print(f"\nApplying standard redaction (image conversion disabled)...")
        redact_pdf(input_pdf_path, output_pdf_path, entities, char_map)
