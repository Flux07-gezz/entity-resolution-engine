"""Conservative, precision-oriented normalization for business names and addresses.

Maintains both raw and normalized representations:
- Preserves address numbers, building numbers, and unit identifiers.
- Canonicalizes common legal suffixes and street abbreviations without destructive stripping.
- Unicode and whitespace canonicalization.
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple


# Standard legal suffix map (bidirectional or canonical form)
LEGAL_SUFFIX_MAP = {
    r"\bincorporated\b": "inc",
    r"\bcorporation\b": "corp",
    r"\bcompany\b": "co",
    r"\blimited liability company\b": "llc",
    r"\blimited liability partnership\b": "llp",
    r"\blimited\b": "ltd",
    r"\bpublic limited company\b": "plc",
    r"\bgmbh\b": "gmbh",
    r"\bs\.a\.\b": "sa",
    r"\bs\.l\.\b": "sl",
    r"\bp\.c\.\b": "pc",
    r"\bpvt\b": "pvt",
    r"\bprivate\b": "pvt",
}

# Standard street abbreviation map
STREET_ABBR_MAP = {
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\broad\b": "rd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bcourt\b": "ct",
    r"\bcircle\b": "cir",
    r"\bparkway\b": "pkwy",
    r"\bhighway\b": "hwy",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bfloor\b": "fl",
}

DIRECTIONAL_MAP = {
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnortheast\b": "ne",
    r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se",
    r"\bsouthwest\b": "sw",
}


def normalize_unicode(text: str) -> str:
    """Canonicalize unicode representation (NFKD)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text)


def clean_whitespace(text: str) -> str:
    """Collapse duplicate whitespace and strip margins."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(text: Optional[str]) -> str:
    """Conservative name normalization.
    
    Lowercase, unicode normalize, strip edge punctuation,
    map legal suffixes to standard abbreviations.
    """
    if not text:
        return ""
    
    text = normalize_unicode(str(text)).lower()
    
    # Replace symbols like '&' with 'and' for token consistency
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"@", " at ", text)
    
    # Remove punctuation except hyphens inside words
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s*\-\s*", " ", text)
    
    # Normalize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, replacement, text)
        
    return clean_whitespace(text)


def normalize_address(text: Optional[str]) -> str:
    """Conservative address normalization.
    
    Lowercase, preserve house numbers and unit numbers,
    map standard street abbreviations and directionals.
    """
    if not text:
        return ""
        
    text = normalize_unicode(str(text)).lower()
    
    # Standardize comma/separators
    text = re.sub(r"[,;]+", " , ", text)
    # Standardize unit symbols like '#' to 'ste '
    text = re.sub(r"#\s*", "ste ", text)
    
    # Strip unnecessary punctuation, keeping hyphens for addresses like 12-B or 123-45
    text = re.sub(r"[^\w\s\-,]", " ", text)
    
    # Normalize street abbreviations
    for pattern, replacement in STREET_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text)
        
    # Normalize directionals
    for pattern, replacement in DIRECTIONAL_MAP.items():
        text = re.sub(pattern, replacement, text)
        
    # Clean whitespace around commas
    text = re.sub(r"\s*,\s*", ", ", text)
    return clean_whitespace(text)


def extract_address_number(norm_address: str) -> Optional[str]:
    """Extract leading or prominent address/house number if present."""
    match = re.search(r"\b(\d+[a-z]?)\b", norm_address)
    if match:
        return match.group(1)
    return None
