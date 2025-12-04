import medspacy
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import sys
import os

# Import configuration
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
except ImportError:
    # Use default configuration if config.py not found
    class config:
        BERT_MODEL_NAME = "medicalai/ClinicalBERT"
        USE_BERT = True
        BERT_CONFIDENCE_THRESHOLD = 0.7
        BERT_DEVICE = "auto"
        MAX_TOKENS_PER_CHUNK = 450
        CHUNK_OVERLAP = 50
        AGE_THRESHOLD = 89
        USE_CONTEXT_MASKING = True
        VERBOSE = False
        BERT_TO_PHI_MAPPING = {
            'PER': 'NAME', 'PERSON': 'NAME', 'PATIENT': 'NAME',
            'DOCTOR': 'DOCTOR', 'PHYSICIAN': 'DOCTOR',
            'LOC': 'ADDRESS', 'LOCATION': 'ADDRESS', 'GPE': 'ADDRESS',
            'FACILITY': 'ADDRESS',
            'DATE': 'DATE',
            'TIME': 'TIME', 'ID': 'ID', 'PHONE': 'PHONE',
            'EMAIL': 'EMAIL', 'AGE': 'AGE',
        }

# Small built-in title and common name lists to help identify person names.
# These are intentionally short to avoid overreliance; add more names offline if you want.
TITLES = ["Dr", "Dr.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Miss", "Mx", "Professor", "Prof", "Prof."]
COMMON_FIRST_NAMES = set("""
James Mary Robert Patricia John Jennifer Michael Linda William Elizabeth David Barbara Richard Susan Joseph Jessica
Thomas Sarah Charles Karen Christopher Nancy Daniel Lisa Matthew Betty Anthony Margaret Donald Sandra Paul Ashley
""".split())
COMMON_LAST_NAMES = set("""
Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson
Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young
""".split())

# Words that often appear near addresses
ADDRESS_KEYWORDS = ["street", "st", "st.", "rd", "rd.", "ave", "avenue", "blvd", "boulevard", "lane", "ln", "drive", "dr", "apt", "suite", "unit", "po box", "p.o. box"]

# HIPAA rule: ages > 89 are considered identifiable (configurable)
AGE_THRESHOLD = getattr(config, 'AGE_THRESHOLD', 89)

# -----------------------
# Regex patterns for PHI
# -----------------------

RE_PATTERNS = {
    # Protected tokens
    "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "SSN_nodash": re.compile(r'\b\d{9}\b'),
    "PHONE": re.compile(r'(\b\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'),
    "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    "MRN": re.compile(r'\b(MRN|Acct|Account|Account#|MRN#)[:\s]*[A-Za-z0-9\-_/]{3,}\b', flags=re.IGNORECASE),
    "SSN_LABEL": re.compile(r'\b(Social Security Number|SSN)[:\s]*\S+\b', flags=re.IGNORECASE),
    "DOB_LABEL": re.compile(r'\b(DOB|Date of Birth)[:\s]*\S+\b', flags=re.IGNORECASE),
    "MEDREC_LABEL": re.compile(r'\b(Medical Record Number|Medical Record #|Record #)[:\s]*[A-Za-z0-9\-_/]{3,}\b', flags=re.IGNORECASE),
    "MEDREC_HASH": re.compile(r'\bMedical Record#[:\s]*\d+\b', flags=re.IGNORECASE),
    "EMAIL_LABEL": re.compile(r'\b(Email|E-mail)[:\s]*\S+\b', flags=re.IGNORECASE),
    "URL": re.compile(r'https?://\S+|www\.\S+'),
    "IP": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    "DATE_ISO": re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),  # 2023-02-01
    # Common date patterns like 01/02/2023, 1/2/23, Jan 2, 2023
    "DATE_GENERIC": re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b', flags=re.IGNORECASE),
    # Date format: DD Month YYYY (e.g., 30 May 2003)
    "DATE_DD_MONTH_YYYY": re.compile(r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', flags=re.IGNORECASE),
    "DATE_DD_MON_YYYY": re.compile(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{4}\b', flags=re.IGNORECASE),
    "TIME": re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b'),
    "SSN_WORD": re.compile(r'\bSS#[:\s]*\d+\b', flags=re.IGNORECASE),
    # Addresses: number + street keyword
    "ADDRESS": re.compile(r'\b\d{1,6}\s+(?:[A-Za-z0-9]+\s){0,4}(?:' + r'|'.join(re.escape(k) for k in ADDRESS_KEYWORDS) + r')\b', flags=re.IGNORECASE),
    # Simple US-style zip code (5 or 5-4)
    "ZIP": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    # Names preceded by titles: Dr. John Doe, Mr Smith
    "TITLE_NAME": re.compile(r'\b(?:' + r'|'.join(re.escape(t) for t in TITLES) + r')\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
    # Uppercase multiword sequences (e.g., JOHN DOE)
    "ALLCAPS_NAME": re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b'),
    # Initials like J. D. Smith or J D Smith
    "INITIALS_NAME": re.compile(r'\b(?:[A-Z]\.\s?){1,3}[A-Z][a-z]+\b'),
    # Age mentions: "age 92", "92-year-old"
    "AGE_NUM": re.compile(r'\b(?:age|Age)[:\s]*\d{1,3}\b|\b\d{1,3}-year-old\b'),
    # Simple patient label: "Patient: John Doe"
    "PATIENT_LABEL": re.compile(r'\b(Patient|Pt|Patient Name|Name)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', flags=re.IGNORECASE),
    # Comma-separated name formats (Lastname, Firstname)
    "LASTNAME_FIRSTNAME_COMMA": re.compile(r'\b([A-Z][a-z]+),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
    "LASTNAME_FIRSTNAME_COMMA_ALLCAPS": re.compile(r'\b([A-Z]{2,}),\s*([A-Z]{2,}(?:\s+[A-Z]{2,})?)\b'),
    "LASTNAME_TITLE_FIRSTNAME": re.compile(r'\b([A-Z][a-z]+),\s*(' + r'|'.join(re.escape(t) for t in TITLES) + r')\s+([A-Z][a-z]+)\b'),
    "LASTNAME_FIRSTNAME_SUFFIX": re.compile(r'\b([A-Z][a-z]+),\s*([A-Z][a-z]+),\s*(Jr\.?|Sr\.?|III|IV|II|Esq\.?)\b', flags=re.IGNORECASE),
    "LASTNAME_FIRST_INITIAL": re.compile(r'\b([A-Z][a-z]+),\s*([A-Z]\.?)\b'),
    # Physician signatures (e.g., "Signed by Mike Boykin", "Mike Boykin N, MD")
    "SIGNED_BY_PHYSICIAN": re.compile(r'\b(?:Signed by|signed by)\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)\b', flags=re.IGNORECASE),
    "PHYSICIAN_WITH_CREDENTIALS": re.compile(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+([A-Z]\.?),?\s+(MD|M\.D\.|DO|D\.O\.|NP|PA|RN)\b', flags=re.IGNORECASE),
    # Social media handles @username
    "HANDLE": re.compile(r'@\w+'),
}

def build_medspacy_pipeline():
    """Initialize medSpaCy pipeline with deidentification components."""
    nlp = medspacy.load(enable=["medspacy_pii"])
    return nlp

def build_bert_pipeline():
    """Initialize BERT NER pipeline for token classification.
    
    Returns:
        Hugging Face NER pipeline configured for medical text, or None if loading fails.
    """
    # Check if BERT is enabled in config
    if not getattr(config, 'USE_BERT', True):
        print("BERT disabled in configuration. Using regex + medSpaCy only.")
        return None
    
    try:
        # Determine device
        bert_device_config = getattr(config, 'BERT_DEVICE', 'auto')
        if bert_device_config == 'auto':
            device = 0 if torch.cuda.is_available() else -1
        elif bert_device_config == 'cuda':
            device = 0
        else:
            device = -1
        
        # Get model name from config
        model_name = getattr(config, 'BERT_MODEL_NAME', 'medicalai/ClinicalBERT')
        
        print(f"Loading BERT NER model: {model_name}")
        print(f"Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")
        
        # Try to load as NER model first
        try:
            # Use a pre-trained NER model
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                device=device,
                aggregation_strategy="simple"  # Merge subword tokens
            )
            print("✓ BERT NER model loaded successfully")
            return ner_pipeline
            
        except Exception as model_error:
            # If the specified model doesn't have NER head, try fallback models
            print(f"⚠ Model {model_name} doesn't support NER, trying fallback models...")
            
            fallback_models = [
                "dslim/bert-base-NER",  # General NER model
                "Jean-Baptiste/roberta-large-ner-english",  # Good general NER
                "Clinical-AI-Apollo/Medical-NER",  # Medical NER
            ]
            
            for fallback_model in fallback_models:
                try:
                    print(f"  Trying: {fallback_model}...")
                    ner_pipeline = pipeline(
                        "ner",
                        model=fallback_model,
                        device=device,
                        aggregation_strategy="simple"
                    )
                    print(f"✓ Loaded fallback model: {fallback_model}")
                    return ner_pipeline
                except Exception as fallback_error:
                    continue
            
            # If all fallbacks fail, raise the original error
            raise model_error
        
    except Exception as e:
        print(f"✗ Warning: Failed to load BERT model: {e}")
        print("  Falling back to regex + medSpaCy only.")
        import traceback
        if getattr(config, 'VERBOSE', False):
            traceback.print_exc()
        return None

def chunk_text(text: str, 
               max_tokens: int = None, 
               overlap: int = None) -> List[Tuple[str, int]]:
    """Split text into chunks for BERT processing (max 512 tokens).
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (leave buffer for special tokens)
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of (chunk_text, start_offset) tuples
    """
    # Use config values if not specified
    if max_tokens is None:
        max_tokens = getattr(config, 'MAX_TOKENS_PER_CHUNK', 450)
    if overlap is None:
        overlap = getattr(config, 'CHUNK_OVERLAP', 50)
    
    # Simple sentence-based chunking to avoid breaking entities
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    current_offset = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())  # Rough token estimate
        
        if current_length + sentence_length > max_tokens and current_chunk:
            # Save current chunk
            chunk_str = '. '.join(current_chunk) + '.'
            chunks.append((chunk_str, current_offset))
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_offset += len('. '.join(current_chunk[:-len(overlap_sentences)])) + 2
            current_chunk = overlap_sentences
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_str = '. '.join(current_chunk)
        if not chunk_str.endswith('.'):
            chunk_str += '.'
        chunks.append((chunk_str, current_offset))
    
    return chunks if chunks else [(text, 0)]

def detect_phi_with_bert(text: str, bert_pipeline) -> List[Dict]:
    """Detect PHI entities using BERT token classification.
    
    Args:
        text: Input text to analyze
        bert_pipeline: Hugging Face NER pipeline
        
    Returns:
        List of entity dicts with keys: start, end, label, text
    """
    if bert_pipeline is None:
        if getattr(config, 'VERBOSE', False):
            print("DEBUG: BERT pipeline is None, skipping BERT detection")
        return []
    
    try:
        # Chunk text if too long
        chunks = chunk_text(text)
        all_entities = []
        
        if getattr(config, 'VERBOSE', False):
            print(f"DEBUG: Processing {len(chunks)} chunks")
        
        for idx, (text_chunk, chunk_offset) in enumerate(chunks):
            if not text_chunk or not text_chunk.strip():
                if getattr(config, 'VERBOSE', False):
                    print(f"DEBUG: Skipping empty chunk {idx}")
                continue  # Skip empty chunks
            
            if getattr(config, 'VERBOSE', False):
                print(f"DEBUG: Processing chunk {idx}, length={len(text_chunk)}, offset={chunk_offset}")
                
            # Run BERT NER on chunk
            predictions = bert_pipeline(text_chunk)
            
            if getattr(config, 'VERBOSE', False):
                print(f"DEBUG: Got {len(predictions)} predictions from BERT")
            
            # Convert predictions to standardized format with absolute offsets
            for pred in predictions:
                # Extract entity information safely
                entity = {
                    'start': pred.get('start', 0) + chunk_offset,
                    'end': pred.get('end', 0) + chunk_offset,
                    'label': map_bert_label_to_phi(pred.get('entity_group', '')),
                    'text': pred.get('word', ''),
                    'score': pred.get('score', 0.0)
                }
                
                # Only include high-confidence predictions (threshold from config)
                confidence_threshold = getattr(config, 'BERT_CONFIDENCE_THRESHOLD', 0.7)
                if entity['score'] > confidence_threshold and entity['label'] is not None:
                    all_entities.append(entity)
                    if getattr(config, 'VERBOSE', False):
                        print(f"DEBUG: Added entity: {entity['label']} - '{entity['text']}' (score={entity['score']:.2f})")
        
        # Remove duplicates from overlapping chunks
        all_entities = deduplicate_entities(all_entities)
        
        if getattr(config, 'VERBOSE', False):
            print(f"DEBUG: Total BERT entities after deduplication: {len(all_entities)}")
        
        return all_entities
        
    except Exception as e:
        print(f"✗ ERROR: BERT prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def map_bert_label_to_phi(bert_label: str) -> str:
    """Map BERT entity labels to PHI categories.
    
    Args:
        bert_label: Entity label from BERT model (e.g., 'PER', 'ORG', 'LOC')
        
    Returns:
        Standardized PHI label (e.g., 'NAME', 'ADDRESS') or None if not PHI
    """
    # Use mapping from config if available
    label_mapping = getattr(config, 'BERT_TO_PHI_MAPPING', {
        'PER': 'NAME', 'PERSON': 'NAME', 'PATIENT': 'NAME',
        'DOCTOR': 'DOCTOR', 'PHYSICIAN': 'DOCTOR',
        'LOC': 'ADDRESS', 'LOCATION': 'ADDRESS', 'GPE': 'ADDRESS',
        'FACILITY': 'ADDRESS',
        'DATE': 'DATE',
        'TIME': 'TIME', 'ID': 'ID', 'PHONE': 'PHONE',
        'EMAIL': 'EMAIL', 'AGE': 'AGE',
    })
    
    return label_mapping.get(bert_label.upper(), None)

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate and overlapping entities, keeping higher confidence ones.
    
    Args:
        entities: List of entity dicts with start, end, label, score
        
    Returns:
        Deduplicated list of entities
    """
    if not entities:
        return []
    
    # Sort by start position, then by score (descending)
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['score']))
    
    deduplicated = []
    
    for entity in sorted_entities:
        # Check for overlap with already added entities
        overlaps = False
        for existing in deduplicated:
            if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                overlaps = True
                break
        
        if not overlaps:
            deduplicated.append(entity)
    
    return deduplicated

def extract_regex_spans(text: str, high_confidence_only: bool = True) -> List[Dict]:
    """Extract PHI spans using regex patterns.
    
    Args:
        text: Input text
        high_confidence_only: If True, only return structured identifiers (SSN, phone, email, etc.)
        
    Returns:
        List of entity dicts with keys: start, end, label, text
    """
    spans = []
    
    # Define high-confidence structured patterns
    structured_patterns = [
        "SSN", "SSN_nodash", "SSN_LABEL", "SSN_WORD",
        "PHONE", "EMAIL", "EMAIL_LABEL", "MRN", "MEDREC_LABEL", "MEDREC_HASH",
        "URL", "IP", "CREDIT_CARD", "HANDLE"
    ]
    
    # Date patterns (medium confidence)
    date_patterns = [
        "DATE_ISO", "DATE_GENERIC", "DATE_DD_MONTH_YYYY", 
        "DATE_DD_MON_YYYY", "DOB_LABEL", "TIME"
    ]
    
    # Address patterns
    location_patterns = ["ADDRESS", "ZIP"]
    
    # Name patterns (lower confidence, BERT should handle these)
    name_patterns = [
        "TITLE_NAME", "ALLCAPS_NAME", "INITIALS_NAME", "PATIENT_LABEL",
        "LASTNAME_FIRSTNAME_COMMA", "LASTNAME_FIRSTNAME_COMMA_ALLCAPS",
        "LASTNAME_TITLE_FIRSTNAME", "LASTNAME_FIRSTNAME_SUFFIX", 
        "LASTNAME_FIRST_INITIAL",
        "SIGNED_BY_PHYSICIAN", "PHYSICIAN_WITH_CREDENTIALS"
    ]
    
    patterns_to_use = structured_patterns + date_patterns + location_patterns
    if not high_confidence_only:
        patterns_to_use += name_patterns + ["AGE_NUM"]
    
    for pattern_name in patterns_to_use:
        if pattern_name not in RE_PATTERNS:
            continue
            
        pattern = RE_PATTERNS[pattern_name]
        
        for match in pattern.finditer(text):
            # Determine output label
            if pattern_name in structured_patterns:
                if "SSN" in pattern_name:
                    label = "SSN"
                elif "PHONE" in pattern_name:
                    label = "PHONE"
                elif "EMAIL" in pattern_name:
                    label = "EMAIL"
                elif "MRN" in pattern_name or "MEDREC" in pattern_name:
                    label = "MRN"
                elif pattern_name == "URL":
                    label = "URL"
                elif pattern_name == "IP":
                    label = "IP"
                elif pattern_name == "CREDIT_CARD":
                    label = "CREDIT_CARD"
                elif pattern_name == "HANDLE":
                    label = "HANDLE"
                else:
                    label = "ID"
            elif pattern_name in date_patterns:
                label = "DATE" if pattern_name != "TIME" else "TIME"
            elif pattern_name in location_patterns:
                label = "ZIP" if pattern_name == "ZIP" else "ADDRESS"
            elif pattern_name in name_patterns:
                if pattern_name in ["SIGNED_BY_PHYSICIAN", "PHYSICIAN_WITH_CREDENTIALS"]:
                    label = "DOCTOR"
                else:
                    label = "NAME"
            # Handle comma-separated name patterns specifically
            elif pattern_name in ["LASTNAME_FIRSTNAME_COMMA", "LASTNAME_FIRSTNAME_COMMA_ALLCAPS",
                                 "LASTNAME_TITLE_FIRSTNAME", "LASTNAME_FIRSTNAME_SUFFIX",
                                 "LASTNAME_FIRST_INITIAL"]:
                label = "NAME"
            elif pattern_name == "AGE_NUM":
                # Check age threshold
                age_match = re.search(r'\d{1,3}', match.group(0))
                if age_match and int(age_match.group(0)) > AGE_THRESHOLD:
                    label = "AGE"
                else:
                    continue  # Skip ages <= 89
            else:
                label = "PHI"
            
            spans.append({
                'start': match.start(),
                'end': match.end(),
                'label': label,
                'text': match.group(0),
                'score': 1.0  # Regex matches are deterministic
            })
    
    return deduplicate_entities(spans)

def merge_bert_and_regex_spans(text: str, bert_entities: List[Dict], 
                                regex_spans: List[Dict]) -> List[Dict]:
    """Merge BERT predictions with regex matches, handling overlaps.
    
    Priority: Regex (structured) > BERT (contextual) for overlaps
    
    Args:
        text: Original text
        bert_entities: Entities detected by BERT
        regex_spans: Spans detected by regex
        
    Returns:
        Merged and deduplicated list of entities sorted by position
    """
    # Start with high-confidence regex matches
    all_entities = regex_spans.copy()
    
    # Add BERT entities that don't overlap with regex
    for bert_entity in bert_entities:
        overlaps = False
        
        for regex_span in regex_spans:
            # Check for overlap
            if (bert_entity['start'] < regex_span['end'] and 
                bert_entity['end'] > regex_span['start']):
                overlaps = True
                break
        
        if not overlaps:
            all_entities.append(bert_entity)
    
    # Sort by start position
    all_entities.sort(key=lambda x: x['start'])
    
    return all_entities

def mask_with_regex(text: str) -> str:
    """Mask PHI using comprehensive HIPAA-compliant regex patterns."""
    # Apply HIPAA regex patterns first
    for label, pattern in RE_PATTERNS.items():
        if label == "AGE_NUM":
            # Special handling for ages > 89 (HIPAA requirement)
            def replace_age(match):
                age_text = match.group(0)
                # Extract the numeric age
                age_match = re.search(r'\d{1,3}', age_text)
                if age_match:
                    age = int(age_match.group(0))
                    if age > AGE_THRESHOLD:
                        return "[AGE>89]"
                return age_text  # Keep ages <= 89
            text = pattern.sub(replace_age, text)
        elif label == "PATIENT_LABEL":
            # Replace patient names after labels
            text = pattern.sub(r'\1: [NAME]', text)
        elif label == "TITLE_NAME":
            # Replace names with titles
            text = pattern.sub('[NAME]', text)
        elif label == "ALLCAPS_NAME":
            # Check if all caps name, replace if likely a person name
            def replace_allcaps(match):
                name = match.group(0)
                words = name.split()
                # Only replace if 2-3 words and looks like a name
                if 2 <= len(words) <= 3:
                    return "[NAME]"
                return name
            text = pattern.sub(replace_allcaps, text)
        elif label == "INITIALS_NAME":
            text = pattern.sub('[NAME]', text)
        elif label in ["LASTNAME_FIRSTNAME_COMMA", "LASTNAME_FIRSTNAME_COMMA_ALLCAPS",
                       "LASTNAME_TITLE_FIRSTNAME", "LASTNAME_FIRSTNAME_SUFFIX",
                       "LASTNAME_FIRST_INITIAL"]:
            text = pattern.sub('[NAME]', text)
        elif label == "SIGNED_BY_PHYSICIAN":
            text = pattern.sub('Signed by [DOCTOR]', text)
        elif label == "PHYSICIAN_WITH_CREDENTIALS":
            text = pattern.sub('[DOCTOR]', text)
        elif label in ["SSN", "SSN_nodash", "SSN_LABEL", "SSN_WORD"]:
            text = pattern.sub('[SSN]', text)
        elif label in ["PHONE"]:
            text = pattern.sub('[PHONE]', text)
        elif label in ["EMAIL", "EMAIL_LABEL"]:
            text = pattern.sub('[EMAIL]', text)
        elif label in ["MRN", "MEDREC_LABEL"]:
            text = pattern.sub('[MRN]', text)
        elif label == "MEDREC_HASH":
            text = pattern.sub('Medical Record#: [MRN]', text)
        elif label in ["DOB_LABEL"]:
            text = pattern.sub('[DATE]', text)
        elif label in ["DATE_ISO", "DATE_GENERIC", "DATE_DD_MONTH_YYYY", "DATE_DD_MON_YYYY"]:
            text = pattern.sub('[DATE]', text)
        elif label == "TIME":
            text = pattern.sub('[TIME]', text)
        elif label == "ADDRESS":
            text = pattern.sub('[ADDRESS]', text)
        elif label == "ZIP":
            text = pattern.sub('[ZIP]', text)
        elif label == "URL":
            text = pattern.sub('[URL]', text)
        elif label == "IP":
            text = pattern.sub('[IP]', text)
        elif label == "CREDIT_CARD":
            text = pattern.sub('[CREDIT_CARD]', text)
        elif label == "HANDLE":
            text = pattern.sub('[HANDLE]', text)
    
    # Additional legacy patterns for compatibility
    legacy_patterns = [
        # Flat/Plot/House numbers with explicit keywords
        (r"\b(?:Flat|Apt|Apartment|Unit|Room|Suite|Plot|House)\s*(?:[Nn]o\.?|Number)\s*[A-Z0-9\-/]+", "[ADDRESS]"),
        
        # Specific residential complex names (only when followed by comma or "at")
        (r"\b(?:at\s+)?[\w\s]{3,30}(?:Colony|Nagar|Vihar|Enclave|Society|Complex|Park|Heights|Apartments|Residency)(?:\s*,|\s+[A-Z])", "[ADDRESS]"),
        
        # Labeled zip codes / pin codes
        (r"\b(?:PIN|Pincode|Pin|Zip|Zipcode|Postal\s*Code)[:\s]*\d{5,6}(?:-\d{4})?\b", "[ZIP]"),
    ]
    
    for pattern, repl in legacy_patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    return text

def mask_names_in_context(text: str) -> str:
    """Mask patient names that appear in specific medical document contexts."""
    # Match "Patient: Name (MRN: ...)" pattern (with optional markdown bold syntax **)
    text = re.sub(
        r"(\*\*)?Patient:(\*\*)?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+\(MRN:",
        r"\1Patient:\2 [NAME] (MRN:",
        text
    )
    
    # Match "Mr./Mrs./Ms. Lastname" references in text
    text = re.sub(
        r"\b(Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+)\b",
        r"\1 [NAME]",
        text
    )
    
    return text

def apply_deidentification_spans(text: str, entities: List[Dict]) -> str:
    """Replace text spans with PHI markers based on entity list.
    
    Args:
        text: Original text
        entities: List of entity dicts with start, end, label keys
        
    Returns:
        Text with entities replaced by [LABEL] markers
    """
    # Sort entities in reverse order to maintain correct offsets during replacement
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    deid_text = text
    
    for entity in sorted_entities:
        label = entity['label']
        start = entity['start']
        end = entity['end']
        
        # Replace entity with marker
        deid_text = deid_text[:start] + f"[{label}]" + deid_text[end:]
    
    return deid_text

def detect_all_phi_entities(text: str, nlp, bert_pipeline=None) -> List[Dict]:
    """
    Detect all PHI entities in text without replacing them.
    
    This function returns the raw entity list for use in PDF redaction
    or other applications that need entity positions rather than redacted text.
    
    Multi-layer approach:
    1. Regex pre-pass: Extract high-confidence structured identifiers
    2. BERT pass: Detect contextual entities (names, addresses, etc.)
    3. Merge BERT + regex: Combine predictions with overlap resolution
    4. medSpaCy validation: Cross-check with medical NER
    
    Args:
        text: Input text to analyze
        nlp: medSpaCy pipeline
        bert_pipeline: BERT NER pipeline (optional, falls back to regex if None)
        
    Returns:
        List of entity dicts with keys: start, end, label, text, score
        Example: [
            {'start': 45, 'end': 55, 'label': 'NAME', 'text': 'John Smith', 'score': 0.95},
            {'start': 60, 'end': 70, 'label': 'DATE', 'text': '01/15/1980', 'score': 1.0},
        ]
    """
    # ===== Layer 1: High-confidence regex patterns (structured PHI) =====
    regex_spans = extract_regex_spans(text, high_confidence_only=True)
    
    # ===== Layer 2: BERT contextual entity detection =====
    bert_entities = []
    if bert_pipeline is not None:
        bert_entities = detect_phi_with_bert(text, bert_pipeline)
    
    # ===== Layer 3: Merge BERT and regex predictions =====
    merged_entities = merge_bert_and_regex_spans(text, bert_entities, regex_spans)
    
    # ===== Layer 4: medSpaCy validation (additional detection) =====
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent._.is_pii:
                # Check if already covered by BERT/regex
                overlaps = False
                for entity in merged_entities:
                    if (ent.start_char < entity['end'] and ent.end_char > entity['start']):
                        overlaps = True
                        break
                
                if not overlaps:
                    merged_entities.append({
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_,
                        'text': ent.text,
                        'score': 0.9
                    })
    except Exception as e:
        print(f"Warning: medSpaCy processing failed: {e}")
    
    # Sort entities by position
    merged_entities.sort(key=lambda x: x['start'])
    
    return merged_entities

def deidentify_text(text: str, nlp, bert_pipeline=None) -> str:
    """Run comprehensive BERT + regex + medSpaCy PHI deidentification.
    
    Multi-layer approach:
    1. Regex pre-pass: Extract high-confidence structured identifiers
    2. BERT pass: Detect contextual entities (names, addresses, etc.)
    3. Merge BERT + regex: Combine predictions with overlap resolution
    4. medSpaCy validation: Cross-check with medical NER
    5. Context masking: Handle document-specific patterns
    6. Final replacement: Apply all PHI markers
    
    Args:
        text: Input text to deidentify
        nlp: medSpaCy pipeline
        bert_pipeline: BERT NER pipeline (optional, falls back to regex if None)
        
    Returns:
        Deidentified text with [LABEL] markers
    """
    # Detect all PHI entities using the shared detection function
    merged_entities = detect_all_phi_entities(text, nlp, bert_pipeline)
    
    # Apply deidentification
    deid_text = apply_deidentification_spans(text, merged_entities)
    
    # Context-specific masking (legacy patterns)
    deid_text = mask_names_in_context(deid_text)
    
    # Final regex pass for any remaining patterns
    final_regex_spans = extract_regex_spans(deid_text, high_confidence_only=True)
    if final_regex_spans:
        deid_text = apply_deidentification_spans(deid_text, final_regex_spans)
    
    return deid_text
