"""
Central constants and mappings for the DESI lensing pipeline.

This module consolidates survey names, tomographic bins, and other constants
that are used throughout the codebase.
"""

from typing import Dict, List

# Survey name mappings
SURVEY_NAME_MAPPINGS: Dict[str, str] = {
    "SDSS": "sdss",
    "HSCY1": "hscy1",
    "HSCY3": "hscy3",
    "DES": "desy3",
    "KiDS": "kids1000N",
    "DECADE": "decade",
    "DECADE_NGC": "decade_ngc",
    "DECADE_SGC": "decade_sgc",
}

# Valid survey names
VALID_SURVEYS: List[str] = [
    "DES", "KiDS", "HSCY1", "HSCY3", "SDSS", "DECADE", "DECADE_NGC", "DECADE_SGC"
]

# Number of tomographic bins per survey
N_TOMOGRAPHIC_BINS: Dict[str, int] = {
    "SDSS": 1,
    "HSCY1": 4,
    "HSCY3": 4,
    "DES": 4,
    "KiDS": 5,
    "DECADE": 4,
    "DECADE_NGC": 4,
    "DECADE_SGC": 4,
}

# Valid galaxy types
VALID_GALAXY_TYPES: List[str] = ["BGS_BRIGHT", "LRG", "ELG"]

# Valid releases
VALID_RELEASES: List[str] = ["iron", "loa"]

# Valid cosmologies
VALID_COSMOLOGIES: List[str] = ["planck18", "wmap9", "wcdm"]

# Valid statistics
VALID_STATISTICS: List[str] = ["deltasigma", "gammat"]

# Default redshift bins per galaxy type
DEFAULT_Z_BINS: Dict[str, List[float]] = {
    "BGS_BRIGHT": [0.1, 0.2, 0.3, 0.4],
    "LRG": [0.4, 0.6, 0.8, 1.1],
    "ELG": [0.8, 1.1, 1.6],
}

# Number of analysis bins per galaxy type (for combined plots)
N_ANALYSIS_BINS: Dict[str, int] = {
    "BGS_BRIGHT": 3,
    "LRG": 2,
    "ELG": 2,
}

# Valid weight types for lenses
VALID_WEIGHT_TYPES: List[str] = ["None", "FRACZ_TILELOCID", "PROB_OBS", "WEIGHT"]

# File extensions
FILE_EXTENSIONS: Dict[str, str] = {
    "measurement": ".fits",
    "covariance": ".npy",
    "precomputed": ".h5",
}

# Blinding random seeds per galaxy type
BLINDING_SEEDS_GALAXY: Dict[str, int] = {
    "BGS": 4654,
    "BGS_BRIGHT": 4654,
    "LRG": 89753,
    "ELG": 57354,
}

# Blinding random seeds per source survey
BLINDING_SEEDS_SURVEY: Dict[str, int] = {
    "des": 98765,
    "hscy1": 98765,
    "kids": 98765,
    "sdss": 98765,
    "hscy3": 98765,
    "decade": 98765,
    "decade_ngc": 98765,
    "decade_sgc": 98765,
}

# Color palette for plotting (tomographic bins)
TOMO_BIN_COLORS: List[str] = ['blue', 'red', 'green', 'orange', 'purple']


def get_survey_internal_name(survey: str) -> str:
    """
    Get the internal name for a survey.
    
    Parameters
    ----------
    survey : str
        Survey name (e.g., "DES", "KiDS")
        
    Returns
    -------
    str
        Internal survey name (e.g., "desy3", "kids1000N")
    """
    return SURVEY_NAME_MAPPINGS.get(survey.upper(), survey.lower())


def get_n_tomographic_bins(survey: str) -> int:
    """
    Get number of tomographic bins for a survey.
    
    Parameters
    ----------
    survey : str
        Survey name
        
    Returns
    -------
    int
        Number of tomographic bins
    """
    return N_TOMOGRAPHIC_BINS.get(survey.upper(), 1)


def get_default_z_bins(galaxy_type: str) -> List[float]:
    """
    Get default redshift bins for a galaxy type.
    
    Parameters
    ----------
    galaxy_type : str
        Galaxy type (BGS_BRIGHT, LRG, ELG)
        
    Returns
    -------
    List[float]
        Default redshift bin edges
    """
    return DEFAULT_Z_BINS.get(galaxy_type, [0.1, 0.2, 0.3, 0.4])


def is_valid_survey(survey: str) -> bool:
    """Check if a survey name is valid."""
    return survey.upper() in VALID_SURVEYS


def is_valid_galaxy_type(galaxy_type: str) -> bool:
    """Check if a galaxy type is valid."""
    return galaxy_type in VALID_GALAXY_TYPES


def is_valid_statistic(statistic: str) -> bool:
    """Check if a statistic name is valid."""
    return statistic.lower() in VALID_STATISTICS

