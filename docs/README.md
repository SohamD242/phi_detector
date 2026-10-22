# HealthCare PHI Detector

A state-of-the-art Protected Health Information (PHI) deidentification tool for medical documents, powered by **ClinicalBERT** deep learning and multi-layer detection.

## Features

### 🧠 **Advanced AI-Powered Detection**
- **ClinicalBERT Integration**: Deep contextual understanding of medical text using transformer-based models
- **Token Classification**: Identifies ambiguous PHI entities that traditional methods miss
- **Doctor vs Patient Discrimination**: Distinguishes between healthcare providers and patients
- **Contextual Entity Recognition**: Understands medical context for accurate name detection

### 🛡️ **Multi-Layer Protection System**
1. **Regex Pre-Pass**: High-confidence structured identifiers (SSN, phone, email, MRN)
2. **BERT Token Classification**: Contextual entity detection (names, addresses, roles)
3. **Entity Merging**: Intelligent overlap resolution with confidence scoring
4. **medSpaCy Validation**: Medical NLP cross-verification
5. **Context-Specific Masking**: Document pattern recognition
6. **Regex Post-Pass**: Final safety net for missed patterns

### 📋 **Comprehensive PHI Coverage**
- **Names**: Patients, doctors, providers (with role disambiguation)
- **Identifiers**: MRN, SSN, account numbers, credit cards
- **Contact**: Phone numbers, email addresses, URLs, social media handles
- **Locations**: Full addresses, ZIP codes, cities, facilities
- **Dates/Times**: All date formats, timestamps, DOB
- **Ages**: HIPAA-compliant (ages >89 masked)
- **Organizations**: Hospitals, clinics, healthcare facilities

### 🎨 **Professional Output**
- **PDF to PDF Pipeline**: Extracts PDF → Markdown → Deidentify → Styled PDF
- **Format Preservation**: Maintains document structure, bold text, bullets, headers
- **Color-Coded Markers**: PHI tags displayed in distinctive orange-brown
- **Multi-Page Support**: Handles long documents with automatic pagination

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

### Core AI/ML Libraries
- **transformers** (≥4.30.0): Hugging Face library for ClinicalBERT model
- **torch** (≥2.0.0): PyTorch backend for deep learning inference
- **medspacy**: Medical NLP and PHI detection
- **spacy**: Core NLP framework

### PDF Processing
- **PyMuPDF**: PDF processing, text extraction, and PDF generation
- **PyMuPDF4LLM**: PDF to markdown conversion with formatting preservation

### Utilities
- **tqdm**: Progress bars for batch processing
- **regex**: Advanced pattern matching

## Architecture

### BERT-Based Deidentification Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  PDF Input → pymupdf4llm → Markdown Text                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 1: Regex Pre-Pass (Structured PHI)                   │
│  ├─ SSN, Phone, Email, MRN patterns                         │
│  ├─ Date formats (ISO, US, European)                        │
│  └─ Credit cards, URLs, IP addresses                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 2: ClinicalBERT Token Classification                 │
│  ├─ Text chunking (512 token limit handling)                │
│  ├─ Subword tokenization with offset mapping                │
│  ├─ Entity prediction (PERSON, LOC, ORG, DATE, etc.)        │
│  ├─ Confidence filtering (>0.7 threshold)                   │
│  └─ Label mapping to PHI categories                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 3: Entity Merging & Deduplication                    │
│  ├─ Overlap resolution (Regex > BERT priority)              │
│  ├─ Confidence-based selection                              │
│  └─ Position-based sorting                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 4: medSpaCy Validation                               │
│  └─ Medical entity recognition cross-check                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 5: Context-Specific Masking                          │
│  ├─ "Patient: Name (MRN: ...)" patterns                     │
│  └─ Title + name combinations (Dr./Mr./Mrs.)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 6: Regex Post-Pass (Safety Net)                      │
│  └─ Catch any remaining structured patterns                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Deidentified Markdown → PyMuPDF → Styled PDF Output        │
└─────────────────────────────────────────────────────────────┘
```

### Model Details

**Primary Model**: `medicalai/ClinicalBERT`
- Pre-trained on MIMIC-III clinical notes
- Fine-tuned for medical entity recognition
- Contextual understanding of medical terminology
- Token-level classification with span aggregation

**Alternative Models** (can be configured in `build_bert_pipeline()`):
- `emilyalsentzer/Bio_ClinicalBERT`
- `allenai/scibert_scivocab_uncased`

**Hardware Acceleration**:
- Automatic GPU detection (CUDA)
- Falls back to CPU if GPU unavailable
- Batch processing for efficiency

## PHI Masking Examples

### Example 1: Basic Patient Information
**Before:**
```
Patient: Sonney Hayes (MRN: 88911233455)
DOB: 03/12/1968
Phone: 555-123-4567
Email: sonney.hayes@email.com
Address: 123 Main Street, Boston, MA 02101
```

**After:**
```
Patient: [NAME] ([MRN])
DOB: [DATE]
Phone: [PHONE]
Email: [EMAIL]
Address: [ADDRESS], [ADDRESS], [ADDRESS] [ZIP]
```

### Example 2: Clinical Context (BERT Advantage)
**Before:**
```
Dr. Sarah Johnson examined the patient John Smith in the Emergency Department.
The patient's wife, Mary Smith, was contacted at 617-555-0123.
```

**After:**
```
[DOCTOR] examined the patient [NAME] in the Emergency Department.
The patient's wife, [NAME], was contacted at [PHONE].
```

### Example 3: Age Masking (HIPAA Compliant)
**Before:**
```
92-year-old male patient
Age: 94
75-year-old female patient
```

**After:**
```
[AGE>89] male patient
Age: [AGE>89]
75-year-old female patient  ← Ages ≤89 preserved per HIPAA
```

## Important Notes

- ⚠️ **Always verify** deidentified output before sharing
- 📁 Original PDF files are **not modified**
- 🔒 All processing is done **locally** on your machine
- 🎯 Designed for HIPAA compliance requirements

## Performance & Optimization

### GPU Acceleration
The system automatically detects and uses GPU if available (requires CUDA-compatible GPU):
```bash
# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

**Performance Comparison:**
- **CPU**: ~5-10 seconds per page
- **GPU**: ~1-2 seconds per page

### Memory Management
- Long documents automatically chunked (450 tokens per chunk)
- Overlap handling prevents entity splitting
- Batch processing for multiple PDFs

### Fallback Behavior
If BERT fails to load (model not found, insufficient memory, etc.):
- System automatically falls back to **regex + medSpaCy only**
- Warning displayed: "⚠ BERT pipeline failed to load"
- Processing continues with traditional detection methods

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
- For torch issues on Windows: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### "BERT model failed to load"
**Common causes:**
1. **Insufficient memory**: BERT requires ~2GB RAM
   - Close other applications
   - Consider using CPU instead of GPU if GPU has low memory
2. **Model download failed**: Hugging Face model not accessible
   - Check internet connection
   - Model will download on first run (~500MB)
3. **Torch not installed correctly**:
   ```bash
   pip uninstall torch
   pip install torch>=2.0.0
   ```

**Fallback**: System will use regex + medSpaCy if BERT unavailable

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Slow processing
- **First run**: Model downloads (~500MB) may take time
- **Subsequent runs**: Models cached locally, much faster
- **Enable GPU**: Install CUDA toolkit for 5-10x speedup
- **Long documents**: Chunking may add overhead

## License

This tool is provided for healthcare PHI deidentification purposes. Users are responsible for ensuring compliance with applicable regulations (HIPAA, GDPR, etc.).

## Support

For issues or questions, please review the troubleshooting section or check the code comments in the source files.
