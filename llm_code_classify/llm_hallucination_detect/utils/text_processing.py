"""
Text processing utilities for hallucination detection.
"""

import re
from typing import Optional


def clean_description(desc: str) -> str:
    """
    Clean and normalize a description text.
    
    Args:
        desc: The description text to clean
        
    Returns:
        Cleaned description text with normalized whitespace
    """
    return ' '.join(desc.strip().split()).replace('\n', ' ')


def extract_line_number(desc: str) -> Optional[int]:
    """
    Extract line number from a description text.
    
    Args:
        desc: The description text to search for line numbers
        
    Returns:
        The line number as an integer if found, None otherwise
    """
    line_match = re.search(r'line\s+(\d+)', desc, re.IGNORECASE)
    return int(line_match.group(1)) if line_match else None 