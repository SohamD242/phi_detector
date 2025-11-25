import pymupdf4llm

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF as markdown with formatting preserved."""
    # Extract PDF content as markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text
