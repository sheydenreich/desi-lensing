"""Command-line interface for DESI lensing pipeline."""

from .main import main
from .config_converter import convert_config_file

__all__ = ["main", "convert_config_file"] 