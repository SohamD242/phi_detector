# CADIS PHI Detector

A comprehensive Protected Health Information (PHI) deidentification tool for medical documents.

## Features

- **Automatic PHI Detection & Masking**:
  - Patient names
  - Medical Record Numbers (MRN)
  - Dates (multiple formats)
  - Phone numbers
  - Email addresses
  - SSN
  - Addresses and ZIP codes
  - Doctor names

- **PDF to PDF Pipeline**: Extracts PDF → Markdown → Deidentify → Styled PDF
- **Format Preservation**: Maintains document structure and formatting
- **Medical Context-Aware**: Uses medSpaCy for medical text understanding
- **Multi-layer Protection**: Combines AI detection with regex patterns
- **Professional Output**: Generates clean, styled PDF documents

## Quick Start

### Prerequisites

- Python 3.7 or higher
- Windows OS (for .bat file) or any OS (for manual execution)

### Installation & Usage (Windows)

1. **Place your PDF files** in the `data/input_pdfs/` folder

2. **Double-click `run_deidentifier.bat`**
   - First run will automatically:
     - Create a virtual environment
     - Install all required dependencies
     - Process your PDFs
   - Subsequent runs will just process the PDFs

3. **Find deidentified PDF files** in `data/output_pdfs/`

### Manual Installation (All Operating Systems)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add PDFs to input folder
# Place your PDF files in: data/input_pdfs/

# 5. Run deidentification
python -m src.main
```

## Project Structure

```
cadis-phi_detector/
├── run_deidentifier.bat    # One-click runner (Windows)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   ├── input_pdfs/         # Place your PDF files here
│   └── output_pdfs/        # Deidentified PDF files appear here
└── src/
    ├── main.py             # Main execution script
    ├── pdf_reader.py       # PDF to markdown extraction
    ├── deidentifier.py     # PHI detection & masking
    └── utils.py            # Markdown to PDF conversion
```

## Dependencies

- **PyMuPDF**: PDF processing, text extraction, and PDF generation
- **PyMuPDF4LLM**: PDF to markdown conversion with formatting preservation
- **medspacy**: Medical NLP and PHI detection
- **spacy**: NLP framework
- **tqdm**: Progress bars
- **regex**: Advanced pattern matching

## PHI Masking Examples

**Before:**
```
Patient: Sonney Hayes (MRN: 88911233455)
DOB: 03/12/1968
Phone: 555-123-4567
```

**After:**
```
Patient: [NAME] ([MRN])
DOB: [DATE]
Phone: [PHONE]
```

## Important Notes

- ⚠️ **Always verify** deidentified output before sharing
- 📁 Original PDF files are **not modified**
- 🔒 All processing is done **locally** on your machine
- 🎯 Designed for HIPAA compliance requirements

## Troubleshooting

### "No PDF files found"
- Ensure PDF files are in `data/input_pdfs/` folder
- Check file extensions are `.pdf` (lowercase)

### "Python is not installed"
- Download Python from: https://www.python.org/downloads/
- During installation, check "Add Python to PATH"

### "Failed to install dependencies"
- Check internet connection
- Try running: `pip install --upgrade pip`
- Manually install: `pip install PyMuPDF PyMuPDF4LLM medspacy tqdm`

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## License

This tool is provided for healthcare PHI deidentification purposes. Users are responsible for ensuring compliance with applicable regulations (HIPAA, GDPR, etc.).

## Support

For issues or questions, please review the troubleshooting section or check the code comments in the source files.
