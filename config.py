# Configuration for CADIS PHI Detector

# ==================== BERT Model Configuration ====================

# Primary model to use for PHI detection
# IMPORTANT: Must be a model with NER/token-classification head
# Recommended options:
#   - "dslim/bert-base-NER" (default, general purpose NER, 420MB)
#   - "Jean-Baptiste/roberta-large-ner-english" (high accuracy, 1.3GB)
#   - "Clinical-AI-Apollo/Medical-NER" (medical-specific NER)
#   - "samrawal/bert-base-uncased_clinical-ner" (clinical NER)
# 
# Note: "medicalai/ClinicalBERT" is a base model without NER head - it won't work for NER
BERT_MODEL_NAME = "dslim/bert-base-NER"

# Enable/disable BERT-based detection
# Set to False to use only regex + medSpaCy (faster, less accurate)
USE_BERT = True

# Confidence threshold for BERT predictions (0.0 - 1.0)
# Higher = fewer false positives, but may miss some PHI
# Lower = catch more PHI, but more false positives
BERT_CONFIDENCE_THRESHOLD = 0.7

# Device for BERT inference
# Options: "auto" (GPU if available, else CPU), "cpu", "cuda"
BERT_DEVICE = "auto"

# ==================== Text Chunking Configuration ====================

# Maximum tokens per chunk for BERT processing
# BERT models have 512 token limit; use slightly less to be safe
MAX_TOKENS_PER_CHUNK = 450

# Number of overlapping tokens between chunks
# Higher = better entity detection at chunk boundaries, but slower
CHUNK_OVERLAP = 50

# ==================== Regex Configuration ====================

# HIPAA age threshold (ages above this are masked)
AGE_THRESHOLD = 89

# Enable context-specific masking (Patient:, Dr., etc.)
USE_CONTEXT_MASKING = True

# ==================== Output Configuration ====================

# Enable verbose logging during processing
VERBOSE = False

# Show progress bars for batch processing
SHOW_PROGRESS = True

# ==================== Security Configuration ====================

# Convert redacted PDFs to image-based format (HIGHLY RECOMMENDED)
# This prevents any possibility of text extraction from redacted PDFs
# True = Maximum security, larger file sizes, slower processing
# False = Faster processing, relies on PyMuPDF's text removal (less secure)
CONVERT_TO_IMAGES = True

# Resolution (DPI) for image-based PDFs
# 300 = Print quality (larger files, ~4MB per page)
# 200 = High screen quality (medium files, ~2MB per page)
# 150 = Good screen quality (smaller files, ~1MB per page)
IMAGE_PDF_DPI = 300

# Redact headers and footers automatically (HIGHLY RECOMMENDED)
# Headers and footers often contain patient names and other PHI
# True = Always redact headers/footers on all pages
# False = Only redact detected PHI entities
REDACT_HEADERS_FOOTERS = True

# Height of header area to redact (in points, 72 points = 1 inch)
# Default 72 = top 1 inch of each page
HEADER_REDACTION_HEIGHT = 19

# Height of footer area to redact (in points)
# Default 72 = bottom 1 inch of each page
FOOTER_REDACTION_HEIGHT = 19

# ==================== Entity Label Mapping ====================

# Map BERT entity labels to PHI categories
# Add custom mappings here if using different models
BERT_TO_PHI_MAPPING = {
    # Person names
    'PER': 'NAME',
    'PERSON': 'NAME',
    'PATIENT': 'NAME',
    'DOCTOR': 'DOCTOR',
    'PHYSICIAN': 'DOCTOR',
    
    # Locations
    'LOC': 'ADDRESS',
    'LOCATION': 'ADDRESS',
    'GPE': 'ADDRESS',
    'FACILITY': 'ADDRESS',
    
    # Dates and times
    'DATE': 'DATE',
    'TIME': 'TIME',
    
    # Other identifiers
    'ID': 'ID',
    'PHONE': 'PHONE',
    'EMAIL': 'EMAIL',
    'AGE': 'AGE',
}

# ==================== Performance Tuning ====================

# Batch size for processing multiple documents
# Higher = faster, but uses more memory
BATCH_SIZE = 1

# Enable model caching (recommended)
CACHE_MODELS = True

# Cache directory for downloaded models
CACHE_DIR = None  # None = use default Hugging Face cache (~/.cache/huggingface)
