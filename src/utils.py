import os
import fitz  # PyMuPDF for PDF generation
import re

def get_pdf_files(input_folder: str):
    """List all PDF files from input folder."""
    return [os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(".pdf")]

def parse_markdown_formatting(text: str):
    """Parse markdown text and return segments with formatting info.
    Returns list of (text, is_bold, is_phi_marker) tuples.
    """
    segments = []
    
    # Handle PHI markers first (they should remain monospace and colored)
    if text.startswith('[') and ']' in text and text.count('[') == 1:
        segments.append((text, False, True))
        return segments
    
    # Parse bold markdown (**text**)
    pattern = r'\*\*(.*?)\*\*'
    last_end = 0
    
    for match in re.finditer(pattern, text):
        # Add text before bold
        if match.start() > last_end:
            normal_text = text[last_end:match.start()]
            if normal_text:
                segments.append((normal_text, False, False))
        
        # Add bold text
        bold_text = match.group(1)
        segments.append((bold_text, True, False))
        last_end = match.end()
    
    # Add remaining text after last bold
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            segments.append((remaining, False, False))
    
    # If no bold formatting found, return entire text as regular
    if not segments:
        segments.append((text, False, False))
    
    return segments

def render_text_with_formatting(page, x: float, y: float, text: str, fontsize: float, 
                                max_width: float, base_color=(0, 0, 0)):
    """Render text with bold markdown formatting using accurate text measurement.
    Returns the final y position after rendering.
    """
    segments = parse_markdown_formatting(text)
    current_x = x
    current_y = y
    line_height = fontsize * 1.4
    
    # Build line buffer to avoid aggressive wrapping
    line_buffer = []
    buffer_width = 0
    
    for segment_text, is_bold, is_phi in segments:
        # Choose font and color
        if is_phi:
            fontname = "cour"
            color = (0.6, 0.4, 0)  # Orange-brown for PHI markers
        elif is_bold:
            fontname = "hebo"  # Helvetica Bold
            color = base_color
        else:
            fontname = "helv"  # Helvetica Regular
            color = base_color
        
        # Build up text segments with proper wrapping using accurate measurement
        words = segment_text.split()
        for i, word in enumerate(words):
            add_space = i < len(words) - 1
            word_with_space = word + (" " if add_space else "")
            
            # Use PyMuPDF's get_text_length for accurate width
            word_width = fitz.get_text_length(word_with_space, fontname=fontname, fontsize=fontsize)
            
            # Check if adding this word would exceed the line width
            if buffer_width + word_width > max_width and line_buffer:
                # Render current line buffer
                for text_seg, fname, clr in line_buffer:
                    page.insert_text((current_x, current_y), text_seg, fontsize=fontsize, 
                                   fontname=fname, color=clr)
                    seg_width = fitz.get_text_length(text_seg, fontname=fname, fontsize=fontsize)
                    current_x += seg_width
                
                # Move to next line
                current_y += line_height
                current_x = x
                line_buffer = []
                buffer_width = 0
            
            # Add word to buffer
            line_buffer.append((word_with_space, fontname, color))
            buffer_width += word_width
    
    # Render remaining buffer
    if line_buffer:
        for text_seg, fname, clr in line_buffer:
            page.insert_text((current_x, current_y), text_seg, fontsize=fontsize, 
                           fontname=fname, color=clr)
            seg_width = fitz.get_text_length(text_seg, fontname=fname, fontsize=fontsize)
            current_x += seg_width
    
    return current_y

def markdown_to_pdf(markdown_text: str, output_path: str):
    """Convert markdown text to a styled PDF using PyMuPDF with proper bold formatting."""
    # Create a new PDF document
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size in points
    
    # Define styling
    margin = 50
    width = 595 - 2 * margin
    y_position = margin
    line_height = 14
    
    # Parse and render markdown content
    lines = markdown_text.split('\n')
    
    for line in lines:
        if y_position > 792:  # Near bottom of page
            page = doc.new_page(width=595, height=842)
            y_position = margin
        
        # Handle different markdown elements
        if line.startswith('# '):  # H1 - Main heading
            text = line[2:].strip()
            # Remove markdown bold syntax from headers
            text = text.replace('**', '')
            y_position += 5  # Small top margin
            page.insert_text((margin, y_position), text, fontsize=20, fontname="hebo", color=(0, 0, 0))
            y_position += line_height * 1.8
            
        elif line.startswith('## '):  # H2 - Subheading
            text = line[3:].strip()
            text = text.replace('**', '')
            y_position += 3
            page.insert_text((margin, y_position), text, fontsize=14, fontname="hebo", color=(0.1, 0.1, 0.1))
            y_position += line_height * 1.5
            
        elif line.startswith('### '):  # H3 - Sub-subheading
            text = line[4:].strip()
            text = text.replace('**', '')
            y_position += 2
            page.insert_text((margin, y_position), text, fontsize=12, fontname="hebo", color=(0.2, 0.2, 0.2))
            y_position += line_height * 1.3
            
        elif line.strip().startswith('- ') or line.strip().startswith('* '):  # Bullet points
            # Handle indentation
            indent_level = (len(line) - len(line.lstrip())) // 2
            bullet_x = margin + 10 + (indent_level * 20)
            text_x = bullet_x + 15
            
            # Get text after bullet
            text = line.lstrip()[2:].strip()
            
            # Render bullet with same size as text (11pt) for better visibility
            page.insert_text((bullet_x, y_position), '●', fontsize=11, fontname="helv", color=(0, 0, 0))
            
            # Render text with bold formatting
            final_y = render_text_with_formatting(page, text_x, y_position, text, 11, 
                                                  width - (text_x - margin), (0, 0, 0))
            y_position = final_y + line_height
                
        elif line.startswith('[') and ']' in line:  # PHI markers like [NAME], [MRN], [DATE]
            # Highlight deidentified content
            page.insert_text((margin, y_position), line, fontsize=11, fontname="cour", color=(0.6, 0.4, 0))
            y_position += line_height
            
        elif line.strip():  # Regular text
            # Render with bold formatting support
            final_y = render_text_with_formatting(page, margin, y_position, line, 11, 
                                                 width, (0, 0, 0))
            y_position = final_y + line_height
        else:
            # Empty line - add spacing
            y_position += line_height * 0.5
    
    # Save the PDF
    doc.save(output_path)
    doc.close()

def save_text(output_folder: str, filename: str, text: str):
    """Save deidentified markdown as styled PDF."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)  # Keep PDF extension
    
    # Convert markdown to PDF
    markdown_to_pdf(text, output_path)
    
    return output_path
