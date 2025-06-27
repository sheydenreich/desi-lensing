"""Lens galaxy configuration for lensing pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np

from .base import BaseConfig


@dataclass
class LensGalaxyConfig(BaseConfig):
    """Configuration for lens galaxies."""
    
    # Galaxy type
    galaxy_type: Literal["BGS_BRIGHT", "LRG", "ELG"] = "BGS_BRIGHT"
    
    # Release selection
    release: Literal["iron", "loa"] = "iron"
    
    # Catalogue versions
    bgs_catalogue_version: str = "v1.5"
    lrg_catalogue_version: str = "v1.5"
    elg_catalogue_version: str = "v1.5"
    
    # Weight type for lenses
    weight_type: Literal["None", "FRACZ_TILELOCID", "PROB_OBS", "WEIGHT"] = "WEIGHT"
    
    # Redshift bins
    z_bins: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    
    # Random catalogues
    which_randoms: List[int] = field(default_factory=lambda: [1, 2])
    
    # Magnitude cuts
    magnitude_cuts: bool = True
    magnitude_column: str = "LOGMSTAR"  # Column to use for magnitude cuts
    
    # Mass completeness
    mstar_complete: bool = False
    
    def validate(self) -> List[str]:
        """Validate lens galaxy configuration."""
        errors = []
        
        # Validate galaxy type
        valid_types = {"BGS_BRIGHT", "LRG", "ELG"}
        if self.galaxy_type not in valid_types:
            errors.append(f"Invalid galaxy_type '{self.galaxy_type}'. Must be one of {valid_types}")
        
        # Validate release
        valid_releases = {"iron", "loa"}
        if self.release not in valid_releases:
            errors.append(f"Invalid release '{self.release}'. Must be one of {valid_releases}")
        
        # Validate weight type
        valid_weights = {"None", "FRACZ_TILELOCID", "PROB_OBS", "WEIGHT"}
        if self.weight_type not in valid_weights:
            errors.append(f"Invalid weight_type '{self.weight_type}'. Must be one of {valid_weights}")
        
        # Validate z_bins
        if len(self.z_bins) < 2:
            errors.append("z_bins must have at least 2 values")
        if not all(self.z_bins[i] < self.z_bins[i+1] for i in range(len(self.z_bins)-1)):
            errors.append("z_bins must be in increasing order")
        if any(z < 0 for z in self.z_bins):
            errors.append("All z_bins values must be non-negative")
        
        # Validate randoms
        if not self.which_randoms:
            errors.append("which_randoms cannot be empty")
        if any(r < 0 for r in self.which_randoms):
            errors.append("All random indices must be non-negative")
        
        return errors
    
    def get_catalogue_version(self) -> str:
        """Get the catalogue version for the current galaxy type."""
        version_map = {
            "BGS_BRIGHT": self.bgs_catalogue_version,
            "LRG": self.lrg_catalogue_version,
            "ELG": self.elg_catalogue_version,
        }
        return version_map[self.galaxy_type]
    
    def get_n_lens_bins(self) -> int:
        """Get the number of lens redshift bins."""
        return len(self.z_bins) - 1
    
    def get_lens_bin_centers(self) -> np.ndarray:
        """Get the centers of the lens redshift bins."""
        bins = np.array(self.z_bins)
        return (bins[:-1] + bins[1:]) / 2
    
    def get_default_z_bins(self) -> List[float]:
        """Get default redshift bins for the galaxy type."""
        defaults = {
            "BGS_BRIGHT": [0.1, 0.2, 0.3, 0.4],
            "LRG": [0.4, 0.6, 0.8, 1.1],
            "ELG": [0.8, 1.1, 1.6],
        }
        return defaults.get(self.galaxy_type, [0.1, 0.2, 0.3, 0.4])
    
    def use_default_z_bins(self) -> None:
        """Set z_bins to the default values for this galaxy type."""
        self.z_bins = self.get_default_z_bins() 