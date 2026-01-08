"""
Password hashing and verification.

Uses passlib with bcrypt for secure password handling.
"""

import logging
import re
from typing import Tuple, List

from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Bcrypt context for password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Cost factor (12 is a good balance of security/speed)
)

# Password requirements
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


def hash_password(password: str) -> str:
    """
    Hash a plain text password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string (bcrypt format)
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored password hash
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password meets security requirements.
    
    Requirements:
    - Minimum 8 characters
    - Maximum 128 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    Args:
        password: Plain text password to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []
    
    # Length checks
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
    
    if len(password) > MAX_PASSWORD_LENGTH:
        errors.append(f"Password must be no more than {MAX_PASSWORD_LENGTH} characters long")
    
    # Character class checks
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit")
    
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?~`]", password):
        errors.append("Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;':\",./<>?~`)")
    
    # Common password patterns check (basic)
    common_patterns = [
        r"^password",
        r"^123456",
        r"^qwerty",
        r"^abc123",
        r"^letmein",
        r"^welcome",
        r"^admin",
        r"^demo",
    ]
    for pattern in common_patterns:
        if re.search(pattern, password.lower()):
            errors.append("Password is too common or easily guessable")
            break
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_password_requirements() -> dict:
    """
    Get password requirements for frontend display.
    
    Returns:
        Dictionary with password requirement descriptions
    """
    return {
        "min_length": MIN_PASSWORD_LENGTH,
        "max_length": MAX_PASSWORD_LENGTH,
        "requirements": [
            f"At least {MIN_PASSWORD_LENGTH} characters",
            "At least one uppercase letter (A-Z)",
            "At least one lowercase letter (a-z)",
            "At least one digit (0-9)",
            "At least one special character (!@#$%^&*...)",
        ]
    }
