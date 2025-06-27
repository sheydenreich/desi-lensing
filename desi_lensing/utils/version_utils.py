"""
Version utilities for DESI lensing pipeline.

This module provides version handling functionality for comparing
catalogue versions and other version-dependent operations.
"""

import re
from typing import Union


class Version:
    """
    Simple version comparison class.
    
    Handles version strings like "v1.0", "v1.5", "v2.0", etc.
    """
    
    def __init__(self, version_string: str):
        """
        Initialize version object.
        
        Parameters
        ----------
        version_string : str
            Version string (e.g., "v1.0", "v1.5")
        """
        self.version_string = version_string.strip()
        self.version_parts = self._parse_version(self.version_string)
    
    def _parse_version(self, version_string: str) -> tuple:
        """
        Parse version string into comparable parts.
        
        Parameters
        ----------
        version_string : str
            Version string to parse
            
        Returns
        -------
        tuple
            Parsed version parts as integers/floats
        """
        # Remove 'v' prefix if present
        clean_version = version_string.lower().lstrip('v')
        
        # Handle different version formats
        if '.' in clean_version:
            try:
                parts = clean_version.split('.')
                return tuple(float(part) for part in parts)
            except ValueError:
                # Fallback for complex version strings
                return (clean_version,)
        else:
            try:
                return (float(clean_version),)
            except ValueError:
                return (clean_version,)
    
    def __str__(self) -> str:
        """String representation."""
        return self.version_string
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Version('{self.version_string}')"
    
    def __eq__(self, other: Union['Version', str]) -> bool:
        """Check equality."""
        if isinstance(other, str):
            other = Version(other)
        return self.version_parts == other.version_parts
    
    def __lt__(self, other: Union['Version', str]) -> bool:
        """Check less than."""
        if isinstance(other, str):
            other = Version(other)
        return self.version_parts < other.version_parts
    
    def __le__(self, other: Union['Version', str]) -> bool:
        """Check less than or equal."""
        if isinstance(other, str):
            other = Version(other)
        return self.version_parts <= other.version_parts
    
    def __gt__(self, other: Union['Version', str]) -> bool:
        """Check greater than."""
        if isinstance(other, str):
            other = Version(other)
        return self.version_parts > other.version_parts
    
    def __ge__(self, other: Union['Version', str]) -> bool:
        """Check greater than or equal."""
        if isinstance(other, str):
            other = Version(other)
        return self.version_parts >= other.version_parts
    
    def __ne__(self, other: Union['Version', str]) -> bool:
        """Check not equal."""
        return not self.__eq__(other) 