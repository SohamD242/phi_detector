import os
from tqdm import tqdm
from src.pdf_redactor import extract_text_with_positions, secure_redact_pdf
from src.deidentifier import build_medspacy_pipeline, build_bert_pipeline, detect_all_phi_entities
from src.utils import get_pdf_files
import config

def main():
    input_folder = "data/input_pdfs"
    output_folder = "data/output_pdfs"

    pdf_files = get_pdf_files(input_folder)
    if not pdf_files:
        print("No PDF files found in input_pdfs/. Please add PDFs and rerun.")
        return

    # Initialize deidentification pipelines
    print("Initializing PHI detection models...")
    print("-" * 50)
    
    # Load medSpaCy for medical entity recognition
    nlp = build_medspacy_pipeline()
    print("✓ medSpaCy pipeline loaded")
    
    # Load BERT model for contextual PHI detection
    bert_pipeline = build_bert_pipeline()
    if bert_pipeline is None:
        print("⚠ BERT pipeline failed to load - using regex + medSpaCy only")
    else:
        print("✓ BERT pipeline loaded")
    
    print("-" * 50)
    
    # Check if image conversion is enabled
    convert_to_images = getattr(config, 'CONVERT_TO_IMAGES', True)
    image_dpi = getattr(config, 'IMAGE_PDF_DPI', 300)
    
    if convert_to_images:
        print(f"\nProcessing {len(pdf_files)} PDFs with SECURE redaction...")
        print(f"(2-stage: Redaction + Image conversion at {image_dpi} DPI)")
        print("(Text extraction will be IMPOSSIBLE from output PDFs)\n")
    else:
        print(f"\nProcessing {len(pdf_files)} PDFs with standard redaction...")
        print("(PHI will be blacked out but may be extractable)\n")

    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Redacting PDFs"):
        try:
            # Extract text with character-to-coordinate mapping
            extraction = extract_text_with_positions(pdf_path)
            text = extraction['text']
            char_map = extraction['char_map']
            
            # Detect PHI entities using BERT + regex + medSpaCy
            entities = detect_all_phi_entities(text, nlp, bert_pipeline)
            
            if len(entities) == 0:
                print(f"\n  ℹ {os.path.basename(pdf_path)}: No PHI detected, copying original...")
                # Copy original if no PHI found
                import shutil
                output_path = os.path.join(output_folder, os.path.basename(pdf_path))
                os.makedirs(output_folder, exist_ok=True)
                shutil.copy2(pdf_path, output_path)
            else:
                # Apply secure redactions to original PDF
                output_path = os.path.join(output_folder, os.path.basename(pdf_path))
                os.makedirs(output_folder, exist_ok=True)
                secure_redact_pdf(
                    pdf_path, 
                    output_path, 
                    entities, 
                    char_map,
                    convert_to_images=convert_to_images,
                    image_dpi=image_dpi
                )
                if convert_to_images:
                    print(f"\n  ✓ {os.path.basename(pdf_path)}: Redacted {len(entities)} PHI entities + converted to images")
                else:
                    print(f"\n  ✓ {os.path.basename(pdf_path)}: Redacted {len(entities)} PHI entities")
            
        except Exception as e:
            print(f"\n✗ Error processing {os.path.basename(pdf_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ PDF redaction complete. Check '{output_folder}' for redacted copies.")
    if convert_to_images:
        print(f"  → All PDFs converted to image-based format (text extraction IMPOSSIBLE)")
        print(f"  → DPI: {image_dpi} (adjust in config.py for size/quality tradeoff)")
    print(f"  Original PDFs remain unchanged in '{input_folder}'.")

if __name__ == "__main__":
    main()
