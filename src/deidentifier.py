import medspacy
import re

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

# HIPAA rule: ages > 89 are considered identifiable
AGE_THRESHOLD = 89

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
    "MEDREC_LABEL": re.compile(r'\b(Medical Record Number|Medical Record #|Record #)[:\s]*\S+\b', flags=re.IGNORECASE),
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
    # Social media handles @username
    "HANDLE": re.compile(r'@\w+'),
}

def build_medspacy_pipeline():
    """Initialize medSpaCy pipeline with deidentification components."""
    nlp = medspacy.load(enable=["medspacy_pii"])
    return nlp

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
        elif label in ["SSN", "SSN_nodash", "SSN_LABEL", "SSN_WORD"]:
            text = pattern.sub('[SSN]', text)
        elif label in ["PHONE"]:
            text = pattern.sub('[PHONE]', text)
        elif label in ["EMAIL", "EMAIL_LABEL"]:
            text = pattern.sub('[EMAIL]', text)
        elif label in ["MRN", "MEDREC_LABEL"]:
            text = pattern.sub('[MRN]', text)
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

def deidentify_text(text: str, nlp) -> str:
    """Run comprehensive PHI deidentification."""
    # First pass: medSpaCy PHI detection
    doc = nlp(text)
    deid_text = text
    
    # Replace entities detected by medSpaCy
    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        if ent._.is_pii:
            deid_text = deid_text[:ent.start_char] + f"[{ent.label_}]" + deid_text[ent.end_char:]
    
    # Second pass: context-specific name masking
    deid_text = mask_names_in_context(deid_text)
    
    # Third pass: regex-based masking for structured PHI
    deid_text = mask_with_regex(deid_text)
    
    return deid_text
