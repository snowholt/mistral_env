"""
Encryption Service for secure credential storage.

Provides Fernet-based symmetric encryption for database field encryption,
with support for key rotation and versioning.

Usage:
    from beautyai_inference.utils.encryption import get_encryption_service
    
    service = get_encryption_service()
    encrypted = service.encrypt("my_secret_token")
    decrypted = service.decrypt(encrypted)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def _get_default_key_path() -> Path:
    """Get default encryption key path."""
    # Check environment variable first
    key_path = os.getenv("ENCRYPTION_KEY_PATH")
    if key_path:
        return Path(key_path)
    
    # Default to backend/.encryption_key
    # This file is at: backend/src/beautyai_inference/utils/encryption.py
    current = Path(__file__).resolve()
    backend_dir = current.parents[3]  # Go up to backend/
    return backend_dir / ".encryption_key"


class EncryptionService:
    """
    Encryption service for securing sensitive data in the database.
    
    Features:
    - Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256)
    - Key versioning for rotation support
    - Thread-safe operations
    
    Example:
        service = EncryptionService()
        
        # Encrypt a token
        encrypted_bytes = service.encrypt("my_secret_token")
        
        # Decrypt when needed
        original = service.decrypt(encrypted_bytes)
    """
    
    def __init__(
        self,
        key_path: Optional[Path] = None,
        key: Optional[bytes] = None,
        key_version: int = 1
    ):
        """
        Initialize encryption service.
        
        Args:
            key_path: Path to encryption key file. If None, uses default.
            key: Raw encryption key bytes. If provided, key_path is ignored.
            key_version: Version number for key rotation tracking.
        """
        self._key_version = key_version
        self._cipher: Optional[Fernet] = None
        
        if key:
            self._cipher = Fernet(key)
            logger.info(f"Encryption service initialized with provided key (v{key_version})")
        else:
            key_path = key_path or _get_default_key_path()
            self._setup_from_file(key_path)
    
    def _setup_from_file(self, key_path: Path) -> None:
        """Load or generate encryption key from file."""
        if key_path.exists():
            with open(key_path, 'rb') as f:
                key = f.read().strip()
            logger.info(f"Encryption key loaded from {key_path}")
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(key_path, 'wb') as f:
                f.write(key)
            
            # Secure the key file (owner read/write only)
            os.chmod(key_path, 0o600)
            logger.warning(f"Generated new encryption key at {key_path}")
        
        self._cipher = Fernet(key)
    
    @property
    def key_version(self) -> int:
        """Current encryption key version."""
        return self._key_version
    
    def encrypt(self, plaintext: str) -> bytes:
        """
        Encrypt a string value.
        
        Args:
            plaintext: The string to encrypt.
            
        Returns:
            Encrypted bytes (includes Fernet timestamp + IV + ciphertext + HMAC).
            
        Raises:
            ValueError: If cipher not initialized.
        """
        if not self._cipher:
            raise ValueError("Encryption service not properly initialized")
        
        if not plaintext:
            raise ValueError("Cannot encrypt empty string")
        
        return self._cipher.encrypt(plaintext.encode('utf-8'))
    
    def decrypt(self, ciphertext: bytes) -> str:
        """
        Decrypt encrypted bytes back to string.
        
        Args:
            ciphertext: The encrypted bytes from encrypt().
            
        Returns:
            Original plaintext string.
            
        Raises:
            ValueError: If cipher not initialized or decryption fails.
        """
        if not self._cipher:
            raise ValueError("Encryption service not properly initialized")
        
        if not ciphertext:
            raise ValueError("Cannot decrypt empty ciphertext")
        
        try:
            decrypted = self._cipher.decrypt(ciphertext)
            return decrypted.decode('utf-8')
        except InvalidToken as e:
            logger.error(f"Decryption failed: invalid token or corrupted data")
            raise ValueError("Decryption failed: invalid token or corrupted data") from e
    
    def encrypt_with_version(self, plaintext: str) -> Tuple[bytes, int]:
        """
        Encrypt with key version for rotation tracking.
        
        Args:
            plaintext: The string to encrypt.
            
        Returns:
            Tuple of (encrypted_bytes, key_version).
        """
        return self.encrypt(plaintext), self._key_version
    
    def rotate_key(self, new_key: bytes, new_version: int) -> 'EncryptionService':
        """
        Create a new service instance with rotated key.
        
        Use this for re-encrypting data during key rotation:
        
            old_service = EncryptionService(key=old_key, key_version=1)
            new_service = old_service.rotate_key(new_key, version=2)
            
            # Re-encrypt data
            plaintext = old_service.decrypt(old_ciphertext)
            new_ciphertext = new_service.encrypt(plaintext)
        
        Args:
            new_key: New encryption key bytes.
            new_version: Version number for new key.
            
        Returns:
            New EncryptionService instance with new key.
        """
        return EncryptionService(key=new_key, key_version=new_version)
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a new Fernet encryption key."""
        return Fernet.generate_key()


# Global service instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """
    Get global encryption service instance.
    
    Lazily initializes on first call.
    """
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def initialize_encryption_service(
    key_path: Optional[Path] = None,
    key: Optional[bytes] = None,
    key_version: int = 1
) -> EncryptionService:
    """
    Initialize global encryption service with custom settings.
    
    Call this during application startup if custom configuration is needed.
    
    Args:
        key_path: Path to encryption key file.
        key: Raw encryption key bytes.
        key_version: Key version number.
        
    Returns:
        Initialized EncryptionService instance.
    """
    global _encryption_service
    _encryption_service = EncryptionService(
        key_path=key_path,
        key=key,
        key_version=key_version
    )
    return _encryption_service
