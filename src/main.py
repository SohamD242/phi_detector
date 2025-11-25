import os
from tqdm import tqdm
from src.pdf_reader import extract_text_from_pdf
from src.deidentifier import build_medspacy_pipeline, deidentify_text
from src.utils import get_pdf_files, save_text

def main():
    input_folder = "data/input_pdfs"
    output_folder = "data/output_pdfs"

    pdf_files = get_pdf_files(input_folder)
    if not pdf_files:
        print("No PDF files found in input_pdfs/. Please add PDFs and rerun.")
        return

    nlp = build_medspacy_pipeline()
    print(f"Processing {len(pdf_files)} PDFs...")

    for pdf_path in tqdm(pdf_files, desc="Deidentifying"):
        text = extract_text_from_pdf(pdf_path)
        deid_text = deidentify_text(text, nlp)
        output_file = os.path.basename(pdf_path)
        save_text(output_folder, output_file, deid_text)

    print(f"\nDeidentification complete. Check '{output_folder}' for results.")

if __name__ == "__main__":
    main()
