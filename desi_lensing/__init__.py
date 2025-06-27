"""
DESI Galaxy-Galaxy Lensing Analysis Pipeline

A modern, modular pipeline for computing galaxy-galaxy lensing signals
using DESI lens galaxies and various source surveys.
"""

__version__ = "2.0.0"
__author__ = "DESI Lensing Team"

from .config import (
    ComputationConfig,
    LensGalaxyConfig,
    SourceSurveyConfig,
    OutputConfig,
)
from .core.pipeline import LensingPipeline
from .core.computation import LensingComputation

# Import analysis module
from . import analysis

__all__ = [
    "ComputationConfig",
    "LensGalaxyConfig", 
    "SourceSurveyConfig",
    "OutputConfig",
    "LensingPipeline",
    "LensingComputation",
    "analysis",
] 